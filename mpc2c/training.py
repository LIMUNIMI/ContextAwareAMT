from time import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from . import data_management, feature_extraction, plotting
from . import settings as s


def train_pedaling(nmf_params):
    trainloader = data_management.get_loader(['train'], nmf_params, 'pedaling')
    validloader = data_management.get_loader(['validation'], nmf_params,
                                             'pedaling')
    model = feature_extraction.MIDIParameterEstimation(s.BINS, 3).to(
        s.DEVICE).to(s.DTYPE)
    train(trainloader, validloader, model)


def train_velocity(nmf_params):
    trainloader = data_management.get_loader(['train'], nmf_params, 'velocity')
    validloader = data_management.get_loader(['validation'], nmf_params,
                                             'velocity')
    model = feature_extraction.MIDIVelocityEstimation(s.BINS).to(s.DEVICE).to(
        s.DTYPE)

    train(trainloader, validloader, model)


def train(trainloader, validloader, model):
    optim = torch.optim.Adadelta(model.parameters(), lr=s.LR_VELOCITY)

    def trainloss_fn(x, y, lens=None):
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
