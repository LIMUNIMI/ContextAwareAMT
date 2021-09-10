import math
from typing import List, Any
from copy import copy

# import plotly.express as px
import torch  # type: ignore
from torch import nn  # type: ignore
from pytorch_lightning import LightningModule  # type: ignore
import pandas as pd
import numpy as np

from . import data_management


def get_conv(inchannels, outchannels, kernel, grouped, transposed, **kwargs):
    groups = 1
    if transposed:
        conv = nn.ConvTranspose2d
        if grouped:
            groups = outchannels
    else:
        conv = nn.Conv2d
        if grouped:
            groups = inchannels
    return conv(inchannels,
                outchannels,
                kernel,
                groups=groups,
                bias=False,
                **kwargs)


class ResidualBlock(nn.Module):
    def __init__(self,
                 inchannels,
                 outchannels,
                 activation,
                 reduce=False,
                 kernel=3,
                 transposed=False):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.activation = activation
        self.reduce = reduce
        self.transposed = transposed
        self.padding = 'valid' if reduce else 'same'
        self.stack = nn.Sequential(
            get_conv(inchannels,
                     outchannels,
                     kernel,
                     True,
                     transposed,
                     padding=self.padding),
            nn.BatchNorm2d(outchannels),
            activation,
            get_conv(outchannels, outchannels, 1, False, transposed),
            nn.BatchNorm2d(outchannels),
            activation,
        )
        self.kernel = (kernel, kernel)
        if not reduce:
            if inchannels == outchannels:
                self.proj = None
            else:
                self.proj = get_conv(inchannels, outchannels, 1, True,
                                     transposed)
        else:
            self.proj = get_conv(inchannels, outchannels, kernel, True,
                                 transposed)

    def forward(self, x):
        if not self.reduce:
            _x = self.stack(x)
            if not self.proj:
                out = _x + x
            else:
                out = _x + self.proj(x)
        else:
            out = self.stack(x) + self.proj(x)  # type: ignore
        self.out = out
        return out

    def outsize(self, insize):
        if not self.reduce:
            return insize
        elif self.transposed:
            return tuple(
                [insize[i] + self.kernel[i] - 1 for i in range(len(insize))])
        else:
            return tuple(
                [insize[i] - self.kernel[i] + 1 for i in range(len(insize))])


class ResidualStack(nn.Module):
    def __init__(self,
                 nblocks,
                 inchannels,
                 outchannels,
                 activation,
                 transposed,
                 kernel=3):
        super().__init__()
        self.nblocks = nblocks
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.activation = activation
        self.transposed = transposed
        self.kernel = kernel
        stack = []
        for _ in range(nblocks - 1):
            stack.append(
                ResidualBlock(inchannels,
                              outchannels,
                              activation,
                              reduce=False,
                              transposed=transposed,
                              kernel=kernel))
            inchannels = outchannels

        stack.append(
            ResidualBlock(inchannels,
                          outchannels,
                          activation,
                          reduce=True,
                          transposed=transposed,
                          kernel=kernel))
        self.stack = nn.Sequential(*stack)

    def outsize(self, insize):
        outsize = insize
        for l in self.stack:
            outsize = l.outsize(outsize)
        return outsize

    def forward(self, x):
        self.out = self.stack(x)
        return self.out

    def get_outputs(self):
        out = []
        for block in self.stack[::-1]:
            out.append(block.out)
        return out


class Encoder(nn.Module):
    def __init__(self, insize, dropout, k1, k2, activation, kernel):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.insize = insize
        self.activation = activation
        self.kernel = kernel

        # make the stack
        stack = []
        outchannels = 1
        while insize[0] > kernel and insize[1] > kernel:
            inchannels = outchannels
            nblocks = max(1, int(2**k1 / outchannels))
            outchannels *= k2
            blocks = ResidualStack(nblocks,
                                   inchannels,
                                   outchannels,
                                   activation,
                                   transposed=False,
                                   kernel=kernel)
            stack.append(blocks)
            insize = blocks.outsize(insize)

        # add one convolution to reduce the size to 1x1
        stack.append(
            nn.Sequential(nn.Conv2d(outchannels, outchannels, insize),
                          nn.BatchNorm2d(outchannels), nn.Tanh()))

        self.stack = nn.Sequential(*stack)
        self.outchannels = outchannels

    def forward(self, x):

        x = self.dropout(x)

        # unsqueeze the channels dimension
        x = x.unsqueeze(1).expand(x.shape[0], 1, x.shape[1], x.shape[2])

        # apply the stack
        self.out = self.stack(x)

        # output has shape (batches, self.outchannels, 1, 1)
        return self.out

    def get_outputs(self):
        out = []
        # out.append(self.out)
        for layer in self.stack[::-1]:
            if type(layer) == ResidualStack:
                out += layer.get_outputs()
        return out


class TripletEncoder(LightningModule):
    def __init__(self, loss_fn, **kwargs):
        """
        triplet-encoder module

        the loss must accept 3 values:
            * prediction
            * same target prediction
            * different target prediction
        """

        super().__init__()
        self.encoder = Encoder(**kwargs)
        # self.decoder = Decoder(self.encoder)
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):

        latent_x = self.encoder(batch['x'])
        latent_same = self.encoder(batch['ae_same'])
        latent_diff = self.encoder(batch['ae_diff'])
        # out = self.decoder(latent)
        loss = self.loss_fn(latent_x, latent_same, latent_diff)

        return {'loss': loss, 'latent': latent_x}

    def validation_step(self, batch, batch_idx):

        latent_x = self.encoder(batch['x'])
        latent_same = self.encoder(batch['ae_same'])
        latent_diff = self.encoder(batch['ae_diff'])
        # out = self.decoder(latent)
        loss = self.loss_fn(latent_x, latent_same, latent_diff)

        return {'loss': loss, 'latent': latent_x}


class Performer(LightningModule):
    def __init__(self, hparams, loss_fn, avg_pred):
        """
        A stack of linear layers that transform `features` into only one
        feature.

        Accept the output of the decoder, having shape: (batches,
        features, 1, 1).

        Returns a tensor with shape (batches, 1)

        * `hyperparams` must contains the following values:

            * middle_features: int [x in k*(2^x)]
            * num_layers: int
            * input_features: int [x in k*(2^x)]
            * middle_activation: callable
            * k: int
        """
        super().__init__()

        self.avg_pred = avg_pred
        self.loss_fn = loss_fn
        middle_features, num_layers, input_features,\
            middle_activation, k = hparams

        middle_features = k * (2**middle_features)
        # input_features = k * (2**input_features)

        stack = []
        for i in range(num_layers - 2):
            stack.append(nn.Linear(middle_features, middle_features))
            stack.append(
                nn.BatchNorm1d(middle_features,
                               affine=True,
                               track_running_stats=True))
            stack.append(middle_activation)
        self.stack = nn.Sequential(
            nn.Linear(input_features, middle_features),
            nn.BatchNorm1d(middle_features,
                           affine=True,
                           track_running_stats=True), middle_activation,
            *stack, nn.Linear(middle_features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.stack(x[:, :, 0, 0])

    def training_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out, batch['y'].unsqueeze(-1))

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out, batch['y'].unsqueeze(-1))
        dummy_loss = self.loss_fn(out, self.avg_pred.expand_as(out))

        return {'loss': loss, 'dummy_loss': dummy_loss}


class EncoderPerformer(LightningModule):
    """
    An iterative transfer-learning LightningModule for
    autoencoder-performer architecture
    """
    def __init__(self,
                 tripletencoder,
                 performer,
                 contexts,
                 mode,
                 lr=1,
                 wd=0,
                 ema_period=20,
                 ema_alpha=0.5,
                 njobs=0):
        super().__init__()
        self.tripletencoder = tripletencoder
        self.performers = nn.ModuleDict(
            {str(c): copy(performer)
             for c in range(len(contexts))})
        self.lr = lr
        self.wd = wd
        self.active = 0  # 0 -> both, 1 -> autoencoder only, 2 -> performers only
        self.mode = mode
        self.contexts = contexts
        self.loss_pool = {"ae": [], "perfm": []}
        self.ema_loss_pool = {"ae": [], "perfm": []}
        self.ema_period = ema_period
        self.ema_alpha = ema_alpha
        self.njobs = njobs

    def training_step(self, batch, batch_idx):

        ae_out = self.tripletencoder.training_step(batch, batch_idx)
        if self.active < 2:
            self.tripletencoder.freeze()
        perfm_out = self.performers[batch['c'][0]].training_step(
            {
                'x': ae_out['latent'],
                'y': batch['y']
            }, batch_idx)
        if self.active < 2:
            self.tripletencoder.unfreeze()

        loss = ae_out['loss'] + perfm_out['loss']

        lr_scheduler = self.lr_schedulers()
        if lr_scheduler is not None:
            lr_scheduler.step()

        self.losslog('train_loss', loss)
        self.losslog('ae_train_loss', ae_out['loss'])
        self.losslog('perfm_train_loss', perfm_out['loss'])
        return {
            'loss': loss,
            'ae_train_loss': ae_out['loss'].detach(),
            'perfm_train_loss': perfm_out['loss'].detach(),
        }

    def validation_step(self, batch, batch_idx, log=True):

        ae_out = self.tripletencoder.validation_step(batch, batch_idx)
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

        loss = ae_out['loss'] + perfm_out['loss']
        self.loss_pool["ae"].append(ae_out["loss"].cpu().numpy().tolist())
        self.loss_pool["perfm"].append(perfm_out["loss"].cpu().numpy().tolist())
        if log:
            self.losslog('val_loss', loss)
            self.losslog('ae_val_loss', ae_out['loss'])
            self.losslog('perfm_val_loss', perfm_out['loss'])
            self.losslog('dummy_loss', perfm_out['dummy_loss'])
        return {
            'loss': loss,
            'ae_val_loss': ae_out['loss'],
            'perfm_val_loss': perfm_out['loss'],
            'dummy_loss': perfm_out['dummy_loss']
        }

    def on_validation_epoch_end(self, log=True):
        # compute loss average and log ema
        if log:
            self.ema_loss_pool["ae"].append(np.mean(self.loss_pool["ae"]))
            self.ema_loss_pool["perfm"].append(np.mean(self.loss_pool["perfm"]))
            ae_ema = ema(self.ema_loss_pool["ae"], self.ema_period, self.ema_alpha)
            perfm_ema = ema(self.ema_loss_pool["perfm"], self.ema_period, self.ema_alpha)
            self.losslog('ae_val_loss_avg', ae_ema)
            self.losslog('perfm_val_loss_avg', perfm_ema)

    def losslog(self, name, value):
        self.log(name,
                 value,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay=0)
        # optim = torch.optim.SGD(self.parameters(),
        #                         momentum=0.9,
        #                         lr=.1,
        #                         nesterov=False,
        #                         weight_decay=0.0)
        # return {
        #     "optimizer":
        #     optim,
        #     "lr_scheduler":
        #     torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
        # }

    def train_dataloader(self):
        dataloader = data_management.get_loader(['train'], False,
                                                self.contexts, self.mode, njobs=self.njobs)
        return dataloader

    def val_dataloader(self):
        dataloader = data_management.get_loader(['validation'], False,
                                                self.contexts, self.mode, njobs=self.njobs)
        self.val_batches = len(dataloader)
        return dataloader


def ema(values: list, min_periods: int, alpha: float):
    ema = pd.DataFrame(values).ewm(
        alpha=alpha, min_periods=min_periods).mean().values[-1, 0]
    return ema
