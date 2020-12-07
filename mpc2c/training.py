import torch
import torch.nn.functional as F

from .mytorchutils import train_epochs
from . import data_management, feature_extraction
from . import settings as s


def train_pedaling(nmf_params, hyperparams):
    trainloader = data_management.get_loader(['train'], nmf_params, 'pedaling')
    validloader = data_management.get_loader(['validation'], nmf_params,
                                             'pedaling')
    model = feature_extraction.MIDIParameterEstimation(
        s.BINS, 3, hyperparams['kernel'], hyperparams['stride'],
        hyperparams['dilation']).to(s.DEVICE).to(s.DTYPE)
    return train(trainloader, validloader, model, hyperparams['lr'])


def train_velocity(nmf_params, hyperparams):
    trainloader = data_management.get_loader(['train'], nmf_params, 'velocity')
    validloader = data_management.get_loader(['validation'], nmf_params,
                                             'velocity')
    model = feature_extraction.MIDIVelocityEstimation(
        s.BINS, 1, hyperparams['kernel'], hyperparams['stride'],
        hyperparams['dilation']).to(s.DEVICE).to(s.DTYPE)

    return train(trainloader,
                 validloader,
                 model,
                 lr=hyperparams['lr'],
                 weight_decay=hyperparams['wd'])


def train(trainloader, validloader, model, *args, **kwargs):
    optim = torch.optim.Adadelta(model.parameters(), *args, **kwargs)

    def trainloss_fn(x, y, lens):
        x, y, lens = x[0], y[0], lens[0]
        y /= 127

        if not lens:
            return F.l1_loss(x, y)

        loss = torch.zeros(len(lens))
        for batch, L in enumerate(lens):
            loss[batch] = F.l1_loss(x[batch, :L], y[batch, :L])
        return loss

    validloss_fn = trainloss_fn
    return train_epochs(model, optim, trainloss_fn, validloss_fn, trainloader,
                        validloader)
