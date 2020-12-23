import torch
import torch.nn.functional as F

from . import data_management, feature_extraction
from . import settings as s
from .mytorchutils import train_epochs


def train_pedaling(nmf_params,
                   hyperparams,
                   lr,
                   wd,
                   context=None,
                   state_dict=None):
    trainloader = data_management.get_loader(
        ['train', context] if context is not None else ['train'], nmf_params,
        'pedaling')
    validloader = data_management.get_loader(
        ['validation', context] if context is not None else ['validation'],
        nmf_params, 'pedaling')
    if s.REDUMP:
        return
    model = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS, ),
        output_features=3,
        note_level=False,
        hyperparams=((hyperparams['kernel_0'], ), (hyperparams['stride_0'], ),
                     (hyperparams['dilation_0'], ))).to(s.DEVICE).to(s.DTYPE)

    # TODO: if state_dict is not None, load it and fix initial weights
    return train(trainloader, validloader, model, lr=lr, wd=wd)


def train_velocity(nmf_params,
                   hyperparams,
                   lr,
                   wd,
                   context=None,
                   state_dict=None):
    trainloader = data_management.get_loader(
        ['train', context] if context is not None else ['train'], nmf_params,
        'velocity')
    validloader = data_management.get_loader(
        ['validation', context] if context is not None else ['validation'],
        nmf_params, 'velocity')
    if s.REDUMP:
        return
    model = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS, s.MINI_SPEC_SIZE),
        output_features=1,
        note_level=True,
        hyperparams=((hyperparams['kernel_0'], hyperparams['kernel_1']),
                     (hyperparams['stride_0'], hyperparams['stride_1']),
                     (hyperparams['dilation_0'],
                      hyperparams['dilation_1']))).to(s.DEVICE).to(s.DTYPE)

    # TODO: if state_dict is not None, load it and fix initial weights
    return train(trainloader, validloader, model, lr=lr, wd=wd)


def train(trainloader, validloader, model, *args, **kwargs):
    print(model)
    print("Total number of parameters: ",
          sum([p.numel() for p in model.parameters() if p.requires_grad]))
    optim = torch.optim.Adadelta(model.parameters(), *args, **kwargs)

    def trainloss_fn(x, y, lens):
        x, y, lens = x[0], y[0], lens[0]
        y /= 127

        if not lens:
            # if `lens` is False, then we are do not have features nor frames
            x = x[..., 0, 0]
            return F.l1_loss(x, y)

        loss = torch.zeros(len(lens))
        for batch, L in enumerate(lens):
            loss[batch] = F.l1_loss(x[batch, :L], y[batch, :L])
        return loss

    validloss_fn = trainloss_fn
    return train_epochs(model,
                        optim,
                        trainloss_fn,
                        validloss_fn,
                        trainloader,
                        validloader,
                        plot_losses=s.PLOT_LOSSES)
