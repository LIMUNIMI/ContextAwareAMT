import math
import time
from typing import List, Any
from copy import copy

import plotly.express as px
import torch  # type: ignore
from torch import nn  # type: ignore
from pytorch_lightning import LightningModule  # type: ignore


def conv_output_size(size, dilation, kernel, stride):
    """
    Computes the output size of a convolutional layer.

    N.B. `padding` not supported for now

    Parameters
    ----------

    `size` : tuple-like[int]
        the size of each dimension processed by the convolution (e.g. for 1D
        convolution, use a tuple/list with 1 element; for 2D conv, use a
        tuple/list with 2 elements etc.)

    `dilation`, `kernel`, `stride` : tuple-like[int]
        parameters of the convolution, each for each dimension
    """
    out = []
    for dim in range(len(size)):
        out.append(
            math.floor((size[dim] - dilation[dim] *
                        (kernel[dim] - 1) - 1) / stride[dim] + 1))
    return tuple(out)


class Encoder(nn.Module):
    def __init__(self, input_size, max_layers, dropout, hyperparams):
        """
        * `hyperparams` must contains the following values:

            * k: int
            * kernel_size : tuple[int]
            * stride : tuple[int]
            * dilation : tuple[int]
            * lstm_hidden_size: int [x in 2^x]
            * lstm_layers: int
            * encoder_features: int [x in k*(2^x)]
            * middle_activation: int
            * output_features: int [x in k*2^x]

        * `input_size` is a tuple[int] containing the number of rows (features)
        and columns (frames)
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # setup the `note_level` stuffs
        k, kernel_size, stride, dilation, lstm_hidden_size,\
            lstm_layers, middle_features, middle_activation,\
            output_features = hyperparams

        middle_features = k * (2**middle_features)
        self.output_features = k * (2**output_features)
        self.middle_activation = middle_activation()

        if lstm_layers > 0:
            conv_in_size = (lstm_hidden_size, input_size[1])
            self.lstm = nn.LSTM(input_size[0],
                                2**lstm_hidden_size,
                                num_layers=lstm_layers,
                                batch_first=True)

        else:
            conv_in_size = input_size

        # make the stack
        self.stack = make_stack(self.output_features, max_layers, conv_in_size,
                                middle_features, middle_activation, dilation,
                                kernel_size, stride)

    def forward(self, x):
        if hasattr(self, 'lstm'):
            # put the frames before of the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)
            x, _ = self.lstm(x)
            # put the frames after the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)

        x = self.dropout(x)

        # unsqueeze the channels dimension
        x = x.unsqueeze(1).expand(x.shape[0], 1, x.shape[1], x.shape[2])
        # x = x.unsqueeze(1).expand(x.shape[0], self.output_features, x.shape[1],
        #                           x.shape[2])
        # apply the stack
        x = self.stack(x)

        # output has shape (batches, output_features, 1, 1)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder):
        """
        A decoder module built from an encoder
        """
        super().__init__()
        # building deconvolutional stack
        stack = []
        for layer in encoder.stack[::-1]:
            if type(layer) == nn.Conv2d:
                stack.append(
                    nn.ConvTranspose2d(layer.out_channels,
                                       layer.in_channels,
                                       kernel_size=layer.kernel_size,
                                       stride=layer.stride,
                                       bias=layer.bias is None,
                                       groups=layer.groups,
                                       dilation=layer.dilation))
                stack.append(
                    nn.InstanceNorm2d(layer.in_channels,
                                      affine=True,
                                      track_running_stats=True)
                    if layer.in_channels > 1 else nn.Identity())
                stack.append(encoder.middle_activation)
        self.stack = nn.Sequential(*stack)

        # building inverse LSTM
        if hasattr(encoder, 'lstm'):
            self.lstm = nn.LSTM(encoder.lstm.hidden_size,
                                encoder.lstm.input_size,
                                encoder.lstm.num_layers,
                                batch_first=True)

    def forward(self, x):
        x = self.stack(x)
        x = x[:, 0, ...]

        if hasattr(self, 'lstm'):
            # put the frames before of the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)
            x, _ = self.lstm(x)
            # put the frames after the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)

        return x


class AutoEncoder(LightningModule):
    def __init__(self, loss_fn, **kwargs):
        """
        encoder-decoder module

        the loss must accept 3 values: 
            * prediction
            * same target
            * different target
        """

        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(self.encoder)
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):

        latent = self.encoder(batch['x'])
        out = self.decoder(latent)
        loss = self.loss_fn(out, batch['ae_same'], batch['ae_diff'])

        return {'loss': loss, 'latent': latent}

    def validation_step(self, batch, batch_idx):

        latent = self.encoder(batch['x'])
        out = self.decoder(latent)
        loss = self.loss_fn(out, batch['ae_same'], batch['ae_diff'])

        return {'loss': loss, 'latent': latent, 'out': out}


class Performer(LightningModule):
    def __init__(self, hparams, loss_fn, avg_pred):
        """
        A stack of linear layers that transform `features` into only one
        feature.

        Accept the output of the decoder, having shape: (batches,
        features, 1, 1).

        Returns a tensor with shape (batches, 1)

        * `hyperparams` must contains the following values:

            * num_layers: int
            * k: int
            * input_features: int [x in k*(2^x)]
            * middle_features: int [x in k*(2^x)]
            * middle_activation: int
        """
        super().__init__()

        self.avg_pred = avg_pred
        self.loss_fn = loss_fn
        middle_features, num_layers, input_features,\
            middle_activation, k = hparams

        middle_features = k * (2**middle_features)
        input_features = k * (2**input_features)

        stack = []
        for i in range(num_layers - 2):
            stack.append(nn.Linear(middle_features, middle_features))
            stack.append(
                nn.BatchNorm1d(middle_features,
                               affine=True,
                               track_running_stats=True))
            stack.append(middle_activation())
        self.stack = nn.Sequential(
            nn.Linear(input_features, middle_features),
            nn.BatchNorm1d(middle_features,
                           affine=True,
                           track_running_stats=True), middle_activation(),
            *stack, nn.Linear(middle_features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.stack(x[:, :, 0, 0])

    def training_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out, batch['y'])

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out, batch['y'])
        dummy_loss = self.loss_fn(out, self.avg_pred)

        return {'loss': loss, 'dummy_loss': dummy_loss}


class EncoderDecoderPerformer(LightningModule):
    """
    An iterative transfer-learning LightningModule for
    autoencoder-performer architecture
    """
    def __init__(self, autoencoder, performer, contexts, lr=1, wd=0):
        super().__init__()
        self.autoencoder = autoencoder
        self.performers = nn.ModuleDict({c: copy(performer) for c in contexts})
        self.lr = lr
        self.wd = wd
        self.active = 0  # 0 -> both, 1 -> autoencoder only, 2 -> performers only

    def training_step(self, batch, batch_idx):

        ae_out = self.autoencoder.training_step(batch, batch_idx)
        if self.active < 2:
            self.autoencoder.freeze()
        perfm_out = self.performers[batch['c'][0]].training_step(
            {
                'x': ae_out['latent'],
                'y': batch['y']
            }, batch_idx)
        if self.active < 2:
            self.autoencoder.unfreeze()

        self.losslog('ae_train_loss', ae_out['loss'])
        self.losslog('perfm_train_loss', perfm_out['loss'])
        return {
            'loss': ae_out['loss'] + perfm_out['loss'],
            'ae_train_loss': ae_out['loss'],
            'perfm_train_loss': perfm_out['loss'],
        }

    def validation_step(self, batch, batch_idx):

        ae_out = self.autoencoder.validation_step(batch, batch_idx)
        perfm_out = self.performers[batch['c'][0]].validation_step(
            {
                'x': ae_out['latent'],
                'y': batch['y']
            }, batch_idx)

        # log an image of reconstruction
        # if batch_idx == 0:
        #     self.logger.experiment.log_figure(
        #         px.imshow(ae_out['out'][0].cpu().numpy()),
        #         'out0' + str(time.time()) + '.html')
        #     self.logger.experiment.log_figure(
        #         px.imshow(batch['x'][0].cpu().numpy()),
        #         'inp0' + str(time.time()) + '.html')

        self.losslog('ae_val_loss', ae_out['loss'])
        self.losslog('perfm_val_loss', perfm_out['loss'])
        self.losslog('dummy_loss', perfm_out['dummy_loss'])
        return {
            'loss': ae_out['loss'] + perfm_out['loss'],
            'ae_val_loss': ae_out['loss'],
            'perfm_val_loss': perfm_out['loss'],
            'dummy_loss': perfm_out['dummy_loss']
        }

    def losslog(self, name, value):
        self.log(name,
                 value,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.wd)


def make_stack(output_features, max_layers, conv_in_size, middle_features,
               middle_activation, dilation, kernel_size, stride):
    """
    Returns a convolutional stack which accepts `conv_in_size` and outputs a
    `output_features x 1 x 1` tensor
    """
    def add_module(input_features, conv_in_size, dilation, kernel_size,
                   stride):
        """
        Add a module to the stack if the output size is >= 0

        Returns (True, size_after_the_module) if the module is added,
        (False, same_size_as_input) if the module is not added.
        """

        # computing size after the first block
        next_conv_in_size = conv_output_size(conv_in_size, dilation,
                                             kernel_size, stride)

        if next_conv_in_size[0] > 0 and \
                conv_in_size[0] > 1 and \
                conv_in_size[0] != next_conv_in_size[0]:
            # if after conv, size is not negative
            # and if the input has something to be reduced
            # and if the conv changes the size (it can happens that
            # dilation creates such a situation)
            if next_conv_in_size[1] < 1:
                # if we cannot apply kernels on the frames, let's
                # apply them framewise except for the last layer
                k = (kernel_size[0], 1)
                s = (stride[0], 1)
                d = (dilation[0], 1)
                next_conv_in_size = (next_conv_in_size[0], conv_in_size[1])
            else:
                k, s, d = kernel_size, stride, dilation

            if next_conv_in_size[0] == 1 and next_conv_in_size[1] == 1:
                # if this is the last layer
                conv_out_features = output_features
            else:
                conv_out_features = middle_features

            module = [
                nn.Conv2d(input_features,
                          conv_out_features,
                          kernel_size=k,
                          stride=s,
                          padding=0,
                          dilation=d,
                          groups=1,
                          bias=False),
                nn.InstanceNorm2d(
                    conv_out_features, affine=True, track_running_stats=True)
                if conv_out_features > 1 else nn.Identity(),
                middle_activation()
            ]
            return next_conv_in_size, module
        else:
            return conv_in_size, None

    # start adding blocks until we can
    stack: List[Any] = []
    input_features = 1
    # input_features = output_features
    for _ in range(max_layers):
        conv_in_size, new_module = add_module(input_features, conv_in_size,
                                              dilation, kernel_size, stride)
        if new_module is None:
            break
        input_features = middle_features
        stack += new_module

    # add the last block to get size 1 along feature dimension and
    # (optionally) along the frame dimension
    if len(stack) == 0:
        raise RuntimeError(
            "Network hyper-parameters would create a one-layer convnet")

    if conv_in_size[0] > 1 or conv_in_size[1] > 1:
        stack += [
            nn.Conv2d(input_features,
                      output_features,
                      kernel_size=conv_in_size,
                      stride=1,
                      dilation=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.InstanceNorm2d(
                output_features, affine=True, track_running_stats=True)
            if output_features > 1 else nn.Identity()
        ]
        stack.append(middle_activation())

        stack.append(
            nn.Conv2d(output_features,
                      output_features,
                      groups=1,
                      kernel_size=1))

        stack.append(middle_activation())
    else:
        # change the last activation so that the outputs are in (0, 1)
        stack.append(
            nn.Conv2d(output_features,
                      output_features,
                      groups=1,
                      kernel_size=1))
        stack.append(middle_activation())
    return nn.Sequential(*stack)


class AbsLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)
