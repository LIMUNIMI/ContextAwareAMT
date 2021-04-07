# from memory_profiler import profile

from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from . import data_management, feature_extraction
from . import settings as s
from .asmd_resynth import get_contexts
from .mytorchutils import (VL_MM_Trainable, best_checkpoint_saver,
                           checkpoint_saver, compute_average, count_params,
                           early_stopping, make_loss_func)


def model_test(model_build_func, test_sample):
    """
    A function to build a constraint around the model size; the constraint
    tries to build the model and to use it with a give ntest sample (can be
    created randomly, but shapes must be similar to a real case)
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
                     hpar['middle_features'], hpar['middle_activation'],
                     1)).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def build_pedaling_model(hpar, dropout):
    m = feature_extraction.MIDIParameterEstimation(
        input_size=(s.BINS, 1),
        output_features=3,
        note_level=False,
        max_layers=s.MAX_LAYERS,
        dropout=dropout,
        hyperparams=((hpar['kernel_0'],
                      1), (1, 1), (1, 1), hpar['lstm_hidden_size'],
                     hpar['lstm_layers'], hpar['middle_features'],
                     hpar['middle_activation'], 3)).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def train(
        hpar,
        mode,
        # TODO: remove transfer_step param
        transfer_step=None,
        # TODO: remove context param
        context=None,
        # TODO: remove state_dict param
        state_dict=None,
        copy_checkpoint='',
        return_model=False):
    """
    1. Builds a model given `hpar` and `mode`
    2. Transfer knowledge from `state_dict` if provided
    3. Freeze first `transfer_step` layers
    4. Train the model on `context`
    5. Saves the trained model weights to `copy_checkpoint`
    6. Returns the best validation loss function + the complexity penalty set
       in `settings`
    7. Also returns the model checkpoint if `return_model` is True
    """
    # loaders
    # TODO: get loaders for all the contexts
    trainloader, validloader = data_management.multiple_splits_one_context(
        ['train', 'validation'], context, mode, False)

    # building model
    if state_dict is not None:
        dropout = s.TRANSFER_DROPOUT
        wd = s.TRANSFER_WD
        transfer_layers = None
        freeze_layers = transfer_step
        lr_k = s.TRANSFER_LR_K
    else:
        wd = s.WD
        dropout = s.TRAIN_DROPOUT
        lr_k = s.LR_K

    # TODO: get 3 models (encoder, decoder, performer)
    # TODO: copy the performer to all the contexts
    if mode == 'velocity':
        model = build_velocity_model(hpar, dropout)
        axes = []
    elif mode == 'pedaling':
        model = build_pedaling_model(hpar, dropout)
        axes = [-1]

    # TODO: remove
    if state_dict is not None:
        model.load_state_dict(state_dict, end=transfer_layers)
        model.freeze(freeze_layers)

    # TODO: remove this
    n_params_free = count_params(model, requires_grad=True)
    # n_params_all = count_params(model, requires_grad=False)
    print(model)
    print("Total number of parameters: ", n_params_free)

    # learning rate
    # lr = s.TRANSFER_LR_K * (s.LR_K / len(trainloader)) * (n_params_all /
    #                                                       n_params_free)
    lr = lr_k / len(trainloader)

    print(f"Using learning rate {lr:.2e}")

    # dummy model (baseline)
    dummy_avg = compute_average(trainloader.dataset,
                                *axes,
                                n_jobs=-1,
                                backend='threading')

    # loss functions
    trainloss_fn = make_loss_func(F.l1_loss)
    validloss_fn = make_loss_func(F.l1_loss)

    model = VL_MM_Trainable(model,
                            inferenceloss_fn=validloss_fn,
                            dummyloss_fn=lambda x, y, z: dummy_avg,
                            optimizer=torch.optim.Adadelta,
                            lr=lr,
                            wd=wd,
                            trainingloss_fn=trainloss_fn)

    # training
    # TODO: train epoch by epoch on each different context (and different model
    early_stopper = early_stopping(s.EARLY_STOP, s.EARLY_RANGE)
    trainer = Trainer(callbacks=[
        best_checkpoint_saver(copy_checkpoint), early_stopper,
        checkpoint_saver(f"checkpoint_{mode}")
    ],
                      precision=s.PRECISION,
                      max_epochs=s.EPOCHS,
                      logger=MLFlowLogger(
                          experiment_name=f'{mode}_{context}_{transfer_step}'))
    trainer.fit(model, trainloader, validloader)
    complexity_loss = count_params(model) * s.COMPLEXITY_PENALIZER
    loss = early_stopper.best_score + complexity_loss

    if return_model:
        return loss, model
    else:
        del model
        return loss


def skopt_objective(hpar: dict, mode: str):
    """
    Runs a training on `orig` context and then retrain the model on each
    specific context.

    Returns the average loss function of the training on the
    specific contexts, including the complexity penalty.
    """

    contexts = get_contexts(Path(s.CARLA_PROJ))
    # train on orig
    print("\n============================\n")
    print("----------------------------")
    print("training on orig")
    print("----------------------------\n")
    _, orig_model = train(hpar,
                          mode,
                          context='orig',
                          state_dict=None,
                          copy_checkpoint='',
                          transfer_step=None,
                          return_model=True)

    # train on the other contexts
    state_dict = orig_model.state_dict()
    del orig_model
    losses = []
    for context in contexts.keys():
        if context == 'orig':
            # skip the `orig` context
            continue

        print("\n----------------------------")
        print(f"testing tl on {context}")
        print("----------------------------\n")
        losses.append(
            train(hpar,
                  mode,
                  context=context,
                  state_dict=deepcopy(state_dict),
                  transfer_step=0,
                  copy_checkpoint='',
                  return_model=False))
        # if losses[-1] > 1:
        #     # if loss was an error, e.g. a nan
        #     return 9999.0

    print("\n============================\n")
    return np.mean(losses)
