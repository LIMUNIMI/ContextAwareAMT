# from memory_profiler import profile

import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from . import data_management, feature_extraction
from . import settings as s
from .mytorchutils import best_checkpoint_saver, compute_average, count_params
from .asmd_resynth import get_contexts


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


def build_autoencoder(hpar, dropout):
    m = feature_extraction.AutoEncoder(loss_fn=F.l1_loss,
                                       input_size=(s.BINS, s.MINI_SPEC_SIZE),
                                       max_layers=s.MAX_LAYERS,
                                       dropout=dropout,
                                       hyperparams=hpar).to(s.DEVICE).to(
                                           s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def get_pedaling_hpar(hpar):
    return (1, (hpar['kernel_0'], 1), (1, 1), (1, 1), hpar['lstm_hidden_size'],
            hpar['lstm_layers'], hpar['encoder_features'],
            hpar['middle_activation'], hpar['latent_features'])


def get_velocity_hpar(hpar):
    return (1, (hpar['kernel_0'], 1), (1, 1), (1, 1), hpar['lstm_hidden_size'],
            hpar['lstm_layers'], hpar['encoder_features'],
            hpar['middle_activation'], hpar['latent_features'])


def build_performer_model(hpar, avg_pred):
    m = feature_extraction.Performer(
        (hpar['performer_layers'], 1, hpar['latent_features'],
         hpar['performer_features'], hpar['middle_activation']), F.l1_loss,
        avg_pred)

    return m


def train(hpar, mode, copy_checkpoint=''):
    """
    1. Builds a model given `hpar` and `mode`
    2. Train the model
    3. Saves the trained model weights to `copy_checkpoint`
    4. Returns the best validation loss function + the complexity penalty set
       in `settings`
    """
    # the logger
    logger = MLFlowLogger(experiment_name=f'{mode}',
                          tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'))

    # loaders
    trainloader = data_management.get_loader(
            ['train'], False, get_contexts(s.CARLA_PROJ))
    validloader = data_management.get_loader(
            ['validation'], False, get_contexts(s.CARLA_PROJ))

    # building model
    if mode == 'velocity':
        ae_hpar = get_velocity_hpar(hpar)
        axes = []
    elif mode == 'pedaling':
        ae_hpar = get_pedaling_hpar(hpar)
        axes = [-1]

    # dummy model (baseline)
    dummy_avg = compute_average(trainloader.dataset,
                                *axes,
                                n_jobs=-1,
                                backend='threading').to(s.DEVICE)

    autoencoder = build_autoencoder(ae_hpar, s.TRAIN_DROPOUT)
    performer = build_performer_model(hpar, dummy_avg)

    print(performer)
    print(autoencoder)

    # learning rate
    # lr = s.TRANSFER_LR_K * (s.LR_K / len(trainloader)) * (n_params_all /
    #                                                       n_params_free)
    # lr = lr_k / len(trainloader)
    lr = 1

    models = [
        feature_extraction.EncoderDecoderPerformer(autoencoder,
                                                   deepcopy(performer), lr,
                                                   s.WD)
        for i in range(len(loaders))
    ]

    # logging initial stuffs
    logger.log_metrics({
        "initial_lr": lr,
        "train_batches": len(trainloader),
        "valid_batches": len(validloader)
    })

    # training
    # TODO: monitored loss should be something else...
    # this loss is also the same used for hyper-parameters tuning
    early_stopper = EarlyStopping(monitor='perfm_loss',
                                  min_delta=s.EARLY_RANGE,
                                  patience=s.EARLY_STOP)

    checkpoint_saver = ModelCheckpoint(f"checkpoint_{mode}",
                                       filename='{epoch}-{ae_loss:.2f}',
                                       monitor='perfm_loss',
                                       save_top_k=1,
                                       mode='min',
                                       save_weights_only=True)
    callbacks = [early_stopper, checkpoint_saver]
    if copy_checkpoint:
        callbacks.append(best_checkpoint_saver(copy_checkpoint))

    trainer = Trainer(callbacks=callbacks,
                      precision=s.PRECISION,
                      max_epochs=s.EPOCHS,
                      logger=logger,
                      gpus=s.GPUS)
    my_train(trainer, models, loaders)

    complexity_loss = count_params(models[0]) * s.COMPLEXITY_PENALIZER
    loss = early_stopper.best_score + complexity_loss
    logger.log_metrics({
        "best_validation_loss": float(early_stopper.best_score),
        "total_loss": float(loss)
    })

    return loss


def my_train(trainer, models, loaders):

    L = len(models)
    for epoch in range(s.EPOCHS):

        # TODO: check that it runs this function at each loop and that logging
        # works correctly
        __import__('ipdb').set_trace()
        trainer.fit(models[epoch % L], *loaders[epoch % L], max_epochs=1)
        if trainer.should_stop:
            # early stop...
            break
