# from memory_profiler import profile

from pprint import pprint

import torch
import torch.nn.functional as F

from . import data_management, feature_extraction
from . import settings as s
from .mytorchutils import (compute_average, count_params, make_loss_func,
                           train_epochs)


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
                model = model_build_func(hpar, s.TRAIN_DROPOUT)
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


def build_velocity_model(hpar, dropout):
    m = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS, s.MINI_SPEC_SIZE),
        output_features=1,
        note_level=True,
        max_layers=s.MAX_LAYERS,
        dropout=dropout,
        hyperparams=((hpar['kernel_0'], hpar['kernel_1']), (1, 1), (1, 1),
                     hpar['lstm_hidden_size'], hpar['lstm_layers'],
                     hpar['middle_features'], hpar['middle_activation'], 1,
                     hpar['sigmoid_last'])).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def build_pedaling_model(hpar, dropout):
    m = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS, 1),
        output_features=3,
        note_level=False,
        max_layers=s.MAX_LAYERS,
        dropout=dropout,
        hyperparams=((hpar['kernel_0'], 1), (1, 1), (1, 1),
                     hpar['lstm_hidden_size'], hpar['lstm_layers'],
                     hpar['middle_features'], hpar['middle_activation'], 3,
                     hpar['sigmoid_last'])).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def train(hpar,
          mode,
          transfer_step=None,
          context=None,
          state_dict=None,
          copy_checkpoint=True):
    # loaders
    trainloader, validloader = data_management.multiple_splits_one_context(
        ['train', 'validation'], context, mode, False)

    # building model
    if state_dict is not None:
        dropout = s.TRANSFER_DROPOUT
        lr_k = s.TRANSFER_LR_K
        early_range = s.TRANSFER_EARLY_RANGE
        early_stop = s.TRANSFER_EARLY_STOP
        wd = s.TRANSFER_WD
        transfer_layers = transfer_step
        freeze_layers = transfer_step
    else:
        wd = s.WD
        lr_k = s.LR_K
        dropout = s.TRAIN_DROPOUT
        early_range = s.EARLY_RANGE
        early_stop = s.EARLY_STOP

    if mode == 'velocity':
        model = build_velocity_model(hpar, dropout)
        axes = []
    elif mode == 'pedaling':
        model = build_pedaling_model(hpar, dropout)
        axes = [-1]

    if state_dict is not None:
        model.load_state_dict(state_dict, end=transfer_layers)
        model.freeze(freeze_layers)

    print(model)
    print("Total number of parameters: ", count_params(model))

    # learning rate
    lr = lr_k / len(trainloader)
    print(f"Using learning rate {lr:.2e}")

    # dummy model (baseline)
    dummy_avg = compute_average(trainloader.dataset,
                                *axes,
                                n_jobs=-1,
                                backend='threading')
    # optimizer
    optim = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=wd)

    # loss functions
    trainloss_fn = make_loss_func(F.l1_loss)
    validloss_fn = make_loss_func(F.l1_loss)

    # training
    train_loss = train_epochs(model,
                              optim,
                              trainloss_fn,
                              validloss_fn,
                              trainloader,
                              validloader,
                              dummy_loss=lambda x: dummy_avg,
                              trainloss_on_valid=True,
                              early_stop=early_stop,
                              early_range=early_range,
                              plot_losses=s.PLOT_LOSSES,
                              copy_checkpoint=copy_checkpoint)
    complexity_loss = count_params(model) * s.COMPLEXITY_PENALIZER

    return train_loss + complexity_loss
