# from memory_profiler import profile
from pprint import pprint

import torch
import torch.nn.functional as F

from . import data_management, feature_extraction
from . import settings as s
from .mytorchutils import count_params, train_epochs, compute_average


def model_test(model_build_func, test_sample):
    """
    A function to build a constraint around the model size; the constraint
    tries to build the model and use it with a random function
    """
    def constraint(hpar):
        print("----------")
        print("checking this set of hpar: ")
        pprint(hpar)
        allowed = True

        if allowed:
            try:
                model = model_build_func(hpar)
                print("model created")
                model(test_sample.to(s.DEVICE).to(s.DTYPE))
                print("model tested")
            # except Exception as e:
            #     import traceback
            #     traceback.print_exc(e)
            except Exception as e:
                print(e)
                allowed = False

        print(f"hyper-parameters allowed: {allowed}")
        return allowed

    return constraint


def build_velocity_model(hpar):
    m = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS - 1, s.MINI_SPEC_SIZE),
        output_features=1,
        note_level=True,
        max_layers=s.MAX_LAYERS,
        hyperparams=((hpar['kernel_0'], hpar['kernel_1']), (1, 1), (1, 1),
                     hpar['lstm_hidden_size'], hpar['lstm_layers'],
                     hpar['middle_features'], hpar['middle_activation'],
                     hpar['k'])).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def build_pedaling_model(hpar):
    m = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS - 1, 1),
        output_features=3,
        note_level=False,
        max_layers=s.MAX_LAYERS,
        hyperparams=((hpar['kernel_0'], 1), (1, 1), (1, 1),
                     hpar['lstm_hidden_size'], hpar['lstm_layers'],
                     hpar['middle_features'], hpar['middle_activation'],
                     hpar['k'])).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def train_pedaling(nmf_params, hpar, lr, wd, context=None, state_dict=None):
    trainloader = data_management.get_loader(
        ['train', context] if context is not None else ['train'], nmf_params,
        'pedaling')
    validloader = data_management.get_loader(
        ['validation', context] if context is not None else ['validation'],
        nmf_params, 'pedaling')
    if s.REDUMP:
        return
    model = build_pedaling_model(hpar)
    if state_dict is not None:
        model.load_state_dict(state_dict, end=s.TRANSFER_PORTION)
        model.freeze(s.FREEZE_PORTION)

    dummy_avg = compute_average(trainloader, -1)

    return train(trainloader,
                 validloader,
                 model,
                 lr,
                 wd,
                 dummy_loss=lambda x: dummy_avg)


def train_velocity(nmf_params, hpar, lr, wd, context=None, state_dict=None):
    trainloader = data_management.get_loader(
        ['train', context] if context is not None else ['train'], nmf_params,
        'velocity')
    validloader = data_management.get_loader(
        ['validation', context] if context is not None else ['validation'],
        nmf_params, 'velocity')
    if s.REDUMP:
        return

    model = build_velocity_model(hpar)

    if state_dict is not None:
        model.load_state_dict(state_dict, end=s.TRANSFER_PORTION)
        model.freeze(s.FREEZE_PORTION)

    dummy_avg = compute_average(trainloader, n_jobs=10)

    return train(trainloader,
                 validloader,
                 model,
                 lr,
                 wd,
                 dummy_loss=lambda x: dummy_avg)


def train(trainloader, validloader, model, lr, wd, dummy_loss):
    print(model)
    print("Total number of parameters: ", count_params(model))
    optim = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=wd)

    def make_loss_func(loss_func):
        def _loss_fn(x, y, lens):
            x, y, lens = x[0], y[0], lens[0]

            if lens == torch.tensor(False):
                # if `lens` is False, then it's like note_level
                x = x[..., 0, 0]
                return loss_func(x, y)

            loss = torch.zeros(len(lens))
            for batch, L in enumerate(lens):
                loss[batch] = loss_func(x[batch, :L], y[batch, :L])
            return loss

        return _loss_fn

    trainloss_fn = make_loss_func(F.l1_loss)
    validloss_fn = make_loss_func(F.l1_loss)
    train_loss = train_epochs(model,
                              optim,
                              trainloss_fn,
                              validloss_fn,
                              trainloader,
                              validloader,
                              dummy_loss=dummy_loss,
                              trainloss_on_valid=True,
                              plot_losses=s.PLOT_LOSSES)
    complexity_loss = count_params(model) * s.COMPLEXITY_PENALIZER

    return train_loss + complexity_loss
