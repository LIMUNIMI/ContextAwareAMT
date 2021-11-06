from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pandas as pd
import rotograd
# import plotly.express as px
import torch
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from torch import nn

from . import data_management, utils


def ema(values: list, min_periods: int, span: float):
    ema = pd.DataFrame(values).ewm(span=span,
                                   min_periods=min_periods).mean().values[-1,
                                                                          0]
    return ema


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

        if type(kernel) is not tuple:
            self.kernel = (kernel, kernel)
        else:
            self.kernel = kernel
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
            out = self.stack(x) + self.proj(x)
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
        for layer in self.stack:
            outsize = layer.outsize(outsize)
        return outsize

    def forward(self, x):
        self.out = self.stack(x)
        return self.out

    def get_outputs(self):
        out = []
        for block in self.stack[::-1]:
            out.append(block.out)
        return out


def make_stack(insize, k1, k2, activation, kernel, condition):
    """
    The initial number of blocks is 2**k1 and channels are 1.
    The i-th stack has number of channels equal to k2**i
    In general the number of blocks is (2**k1)/(k2**i) and the number of
    channels is k2**i
    """
    stack = []
    outchannels = 1
    while condition(insize, kernel):
        inchannels = outchannels
        nblocks = max(1, round(2**k1 / outchannels))
        outchannels = round(outchannels * k2)
        blocks = ResidualStack(nblocks,
                               inchannels,
                               outchannels,
                               activation,
                               transposed=False,
                               kernel=kernel)
        stack.append(blocks)
        insize = blocks.outsize(insize)
    return stack, outchannels, insize


class Encoder(LightningModule):
    def __init__(self, insize, dropout, k1, k2, activation, kernel):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.insize = insize
        self.activation = activation
        self.kernel = kernel

        # make the stack
        stack, outchannels, insize = make_stack(
            insize, k1, k2, activation, kernel,
            lambda x, y: x[0] > y and x[1] > y)

        # add one convolution to reduce the size to 1x1
        stack.append(
            nn.Sequential(nn.Conv2d(outchannels, outchannels, insize),
                          nn.BatchNorm2d(outchannels), activation))

        self.stack = nn.Sequential(
            *stack,
            nn.Conv2d(outchannels,
                      outchannels,
                      kernel_size=1,
                      groups=outchannels), activation)
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


class Specializer(LightningModule):
    def __init__(self, middle_features, k1, k2, activation, kernel, nout,
                 loss_fn):
        super().__init__()

        stack, outchannels, insize = make_stack((middle_features, 1), k1, k2,
                                                activation, (kernel, 1),
                                                lambda x, y: x[0] > y[0])

        stack.append(
            nn.Sequential(
                nn.Conv2d(outchannels, nout, insize), nn.BatchNorm2d(nout),
                activation,
                nn.Conv2d(nout, nout, kernel_size=1, groups=nout)
                if nout == 1 else nn.Identity(),
                nn.Sigmoid() if nout == 1 else nn.Softmax(dim=1))
            # nn.Sigmoid() if nout == 1 else nn.Identity())
        )

        self.stack = nn.Sequential(*stack)
        self.loss_fn = loss_fn
        self.nout = nout

    def forward(self, x):
        x = self.stack(torch.transpose(x, 1, 2))[:, :, 0, 0]
        if self.nout == 1:
            return x[:, 0]
        else:
            return x

    def training_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out.float(), batch['y'].float())

        # # autoweight the loss in respect to the maximum ever seen
        # if loss > self.maxloss:
        #     self.maxloss = loss.detach()
        # loss = loss / self.maxloss
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        out = self.forward(batch['x'])
        loss = self.loss_fn(out.float(), batch['y'].float())

        if self.nout > 1:
            out = torch.argmax(out, 1)

        return {'out': out, 'loss': loss}


class EncoderPerformer(LightningModule):
    """
    An iterative transfer-learning LightningModule for
    context-aware transcription
    """
    def __init__(self,
                 encoder,
                 performer,
                 cont_classifier,
                 contexts,
                 mode,
                 context_specific,
                 multiple_performers,
                 lr=1,
                 wd=0,
                 ema_period=None,
                 njobs=0,
                 perfm_testloss=nn.L1Loss(reduction='none')):
        super().__init__()
        self.context_specific = context_specific
        self.multiple_performers = multiple_performers
        self.encoder = encoder
        if self.context_specific:
            self.context_classifier = cont_classifier
        self.performers = nn.ModuleDict({
            str(c): deepcopy(performer) if multiple_performers else performer
            for c in range(len(contexts))
        })
        self.contexts = contexts
        self.lr = lr
        self.wd = wd
        self.mode = mode
        self._reset_loss_pool()
        self.reset_ema()
        self.ema_period = ema_period
        self.njobs = njobs
        self.perfm_testloss = perfm_testloss
        self.test_latent_x = []
        self.test_latent_y = []
        self.use_rotograd = False

    @property
    def use_ema(self):
        return self.ema_period is not None

    def reset_ema(self):
        self.ema_loss_pool = {"cont": [], "perfm": []}

    def _reset_loss_pool(self):
        self.loss_pool = {"cont": [], "perfm": []}

    @property
    def use_rotograd(self):
        return self.multiple_optimizers and not self.automatic_optimization

    @use_rotograd.setter
    def use_rotograd(self, v):
        self.automatic_optimization = not v
        self.multiple_optimizers = v

    def forward(self, x, context):
        enc_out = self.encoder.forward(x)
        perfm_out = self.performers[context].forward(enc_out)
        return perfm_out

    def training_step(self, batch, batch_idx):

        out = dict()
        context_s = batch['c'][0]
        context_i = int(context_s)
        if self.use_rotograd:
            opts = self.optimizers()
            if type(opts) in [list, tuple]:
                for opt in opts:
                    opt.zero_grad()
            else:
                opts.zero_grad()

        if self.use_rotograd:
            context = rotograd.cached()
        else:
            context = nullcontext()

        # Speeds-up computations by caching Rotograd's parameters
        with context:
            enc_out = self.encoder.forward(batch['x'])
            perfm_out = self.performers[context_s].training_step(
                {
                    'x': enc_out,
                    'y': batch['y']
                }, batch_idx)
            loss = perfm_out['loss']
            out['perfm_train_loss'] = perfm_out['loss'].detach()
            self.losslog('perfm_train_loss', perfm_out['loss'])

            if self.context_specific:
                new_y = torch.zeros(enc_out.shape[0],
                                    len(self.contexts),
                                    dtype=enc_out.dtype,
                                    device=enc_out.device)
                new_y[:, context_i] = 1
                cont_out = self.context_classifier.training_step(
                    {
                        'x': enc_out,
                        'y': new_y
                    }, batch_idx)
                loss = loss + cont_out['loss']
                out['cont_train_loss'] = cont_out['loss'].detach()
                self.losslog('cont_train_loss', cont_out['loss'])

            lr_scheduler = self.lr_schedulers()
            if lr_scheduler is not None:
                lr_scheduler.step()

            loss = loss.float()
            out['loss'] = loss
            self.losslog('train_loss', loss)

            if self.use_rotograd:
                self.manual_backward(loss)

        if self.use_rotograd:
            if type(opts) in [list, tuple]:
                for opt in opts:
                    opt.step()
            else:
                opts.step()

        return out

    def validation_step(self, batch, batch_idx):

        context_s = batch['c'][0]
        context_i = int(context_s)
        enc_out = self.encoder.forward(batch['x'])
        perfm_out = self.performers[context_s].validation_step(
            {
                'x': enc_out,
                'y': batch['y']
            }, batch_idx)
        loss = perfm_out['loss']
        out = {'perfm_val_loss': perfm_out['loss'].detach()}
        self.losslog('perfm_val_loss', perfm_out['loss'])
        if self.context_specific:
            new_y = torch.zeros(enc_out.shape[0],
                                len(self.contexts),
                                dtype=enc_out.dtype,
                                device=enc_out.device)
            new_y[:, context_i] = 1
            cont_out = self.context_classifier.validation_step(
                {
                    'x': enc_out,
                    'y': new_y
                }, batch_idx)
            loss = loss + cont_out['loss']
            out['cont_val_loss'] = cont_out['loss'].detach()
            self.loss_pool["cont"].append(cont_out["loss"].cpu().numpy())

        out['loss'] = loss
        self.loss_pool["perfm"].append(perfm_out["loss"].cpu().numpy())
        self.losslog('val_loss', loss)
        if self.context_specific:
            self.losslog('cont_val_loss', cont_out['loss'])
        return out

    def on_validation_epoch_end(self):
        # compute loss average and log ema
        if self.use_ema:
            # loss_pool contains the losses from this epoch
            self.ema_loss_pool["cont"].append(np.mean(self.loss_pool["cont"]))
            self.ema_loss_pool["perfm"].append(np.mean(
                self.loss_pool["perfm"]))
            cont_ema = ema(self.ema_loss_pool["cont"], self.ema_period,
                           self.ema_period)
            perfm_ema = ema(self.ema_loss_pool["perfm"], self.ema_period,
                            self.ema_period)
            self.losslog('cont_val_loss_early_stop', cont_ema)
            self.losslog('perfm_val_loss_early_stop', perfm_ema)
            self._reset_loss_pool()
        else:
            self.losslog('cont_val_loss_early_stop',
                         self.ema_loss_pool["cont"][-1])
            self.losslog('perfm_val_loss_early_stop',
                         self.ema_loss_pool["perfm"][-1])
        for key, val in self.performer_weight_moments().items():
            self.losslog("weight_variance_" + key, val)

    def test_step(self, batch, batch_idx):

        context_s = batch['c'][0]
        context_i = int(context_s)
        enc_out = self.encoder.forward(batch['x'])
        perfm_out = self.performers[context_s].validation_step(
            {
                'x': enc_out,
                'y': batch['y']
            }, batch_idx)['out']
        if self.context_specific:
            new_y = torch.zeros(enc_out.shape[0],
                                len(self.contexts),
                                dtype=enc_out.dtype,
                                device=enc_out.device)
            new_y[:, context_i] = 1
            cont_out = self.context_classifier.validation_step(
                {
                    'x': enc_out,
                    'y': new_y
                }, batch_idx)['out'].cpu().numpy()
        else:
            cont_out = None

        # record latents variables for clustering and accuracy computation
        self.test_latent_x.append(enc_out.cpu().numpy())
        batch_y = torch.tensor(context_i,
                               device=enc_out.device,
                               dtype=torch.long).expand(enc_out.shape[0])
        self.test_latent_y.append(batch_y.cpu().numpy())
        # * add test_epoch_end in which latent variables are clusterized
        return self.perfm_testloss(batch['y'],
                                   perfm_out).cpu().numpy(), cont_out

    def test_epoch_end(self, outputs, log=True):
        perfm_outputs = np.concatenate([o[0] for o in outputs])
        perfm_out_avg = np.mean(perfm_outputs)
        perfm_out_std = np.std(perfm_outputs)

        cont_labels = np.concatenate(self.test_latent_y)
        if self.context_specific:
            cont_outputs = np.concatenate([o[1] for o in outputs])
            cont_recalls = recall_score(cont_labels,
                                        cont_outputs,
                                        average=None)
            cont_bal_acc = np.mean(cont_recalls)
            if log:
                self.losslog('cont_bal_acc', cont_bal_acc)
                for i, rec in enumerate(cont_recalls):
                    self.losslog(f'rec_test_{i}', rec)

        cluster_computer = KMeans(n_clusters=len(self.contexts))
        labels = cluster_computer.fit_predict(
            np.concatenate(self.test_latent_x)[:, :, 0, 0])
        ami = adjusted_mutual_info_score(cont_labels, labels)

        if log:
            self.losslog('perfm_test_avg', perfm_out_avg)
            self.losslog('perfm_test_std', perfm_out_std)
            self.losslog('test_ami', ami)

    def performer_weight_moments(self):
        """
        Computes the average variance of the weights of the performers
        """
        # get all the parameters of the performers
        params = [list(perf.parameters()) for perf in self.performers.values()]
        s = []
        for i in range(len(params[0])):
            # for each performer parameter
            # compute point-wise variances
            v = torch.var(
                # double() is needed because when everything is the same
                # (variance=0) there is some issue with precision and variance
                # would appear to be > 0
                torch.stack([p[i].detach().double() for p in params]),
                dim=(0, ),
                unbiased=True)
            # append to the list the average variance
            s.append(torch.mean(v))
        return utils.torch_moments(torch.stack(s))

    def losslog(self, name, value):
        self.log(name,
                 value,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=0)

        if not self.multiple_optimizers:
            return optimizer

        self.rotograd_model = rotograd.RotoGrad(
            self.encoder, [self.performers, self.context_classifier],
            self.encoder.outchannels,
            alpha=1.,
            burn_in_period=10)

        optim_rotograd = torch.optim.Adadelta(self.rotograd_model.parameters(),
                                              lr=self.lr / 2)

        return optim_rotograd, optimizer

    def train_dataloader(self):
        dataloader = data_management.get_loader(['train'],
                                                False,
                                                self.contexts,
                                                self.multiple_performers,
                                                self.mode,
                                                njobs=self.njobs)
        return dataloader

    def val_dataloader(self):
        dataloader = data_management.get_loader(['validation'],
                                                False,
                                                self.contexts,
                                                self.multiple_performers,
                                                self.mode,
                                                njobs=self.njobs)
        return dataloader

    def test_dataloader(self):
        dataloader = data_management.get_loader(['test'],
                                                False,
                                                self.contexts,
                                                self.multiple_performers,
                                                self.mode,
                                                njobs=1)
        # for some reason there are leakings with njobs > 1
        return dataloader
