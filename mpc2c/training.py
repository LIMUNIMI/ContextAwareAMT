# from memory_profiler import profile

import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch.nn.functional as F  # type: ignore
from torch import nn  # type: ignore
import torch  # type: ignore
import torchinfo
from pytorch_lightning import Trainer  # type: ignore
from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # type: ignore
from pytorch_lightning.loggers import MLFlowLogger  # type: ignore

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
                model = model_build_func(hpar)
                print("model created")
                model.eval()
                model.to(s.DEVICE).validation_step(
                    {
                        'x': test_sample.to(s.DEVICE).to(s.DTYPE),
                        'y': torch.tensor(0.5).to(s.DEVICE).to(s.DTYPE),
                        'ae_diff': test_sample.to(s.DEVICE).to(s.DTYPE),
                        'ae_same': test_sample.to(s.DEVICE).to(s.DTYPE),
                        'c': '0'
                    },
                    0,
                    log=False)
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


def reconstruction_loss(pred, same_pred, diff_pred):
    # return F.l1_loss(pred, same_pred) / (F.l1_loss(same_pred, diff_pred) + F.l1_loss(pred, diff_pred))
    return F.triplet_margin_with_distance_loss(
        pred,
        same_pred,
        diff_pred,
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=1,
        swap=True,
        reduction='sum')


def build_encoder(hpar, dropout, generic=False):
    if generic:
        loss_fn = lambda pred, _, diff_pred: F.l1_loss(pred, diff_pred)
    else:
        loss_fn = reconstruction_loss

    k, activation, kernel = get_hpar(hpar)

    m = feature_extraction.TripletEncoder(
        loss_fn=loss_fn,
        insize=(
            s.BINS,
            s.MINI_SPEC_SIZE),  # mini_spec_size should change for pedaling...
        dropout=dropout,
        k=k,
        activation=activation,
        kernel=kernel).to(s.DEVICE).to(s.DTYPE)
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def get_hpar(hpar):
    return (hpar['ae_k'], hpar['activation'], hpar['kernel'])


def build_performer_model(hpar, infeatures, avg_pred):
    m = feature_extraction.Performer(
        (hpar['performer_features'], hpar['performer_layers'], infeatures,
         hpar['activation'], 1), F.l1_loss, avg_pred)

    return m


def build_model(hpar,
                mode,
                dropout=s.TRAIN_DROPOUT,
                dummy_avg=torch.tensor(0.5).cuda(),
                generic=True):

    contexts = list(get_contexts(s.CARLA_PROJ).keys())
    autoencoder = build_encoder(hpar, dropout, generic)
    performer = build_performer_model(hpar, autoencoder.encoder.outchannels,
                                      dummy_avg)
    model = feature_extraction.EncoderPerformer(autoencoder, performer,
                                                contexts, mode)
    return model


def my_train(mode, copy_checkpoint, logger, model, ae_train=True, perfm_train=True):
    """
    Creates callbacks, train and freeze the part that raised an early-stop.
    Return the best loss of that one and 0 for the other.
    """
    # setup callbacks
    checkpoint_saver = ModelCheckpoint(f"checkpoint_{mode}",
                                       filename='{epoch}-{ae_loss:.2f}',
                                       monitor='val_loss',
                                       save_top_k=1,
                                       mode='min',
                                       save_weights_only=True)
    callbacks = [checkpoint_saver]
    if ae_train:
        ae_stopper = EarlyStopping(monitor='ae_val_loss_avg',
                                   min_delta=s.EARLY_RANGE,
                                   patience=s.EARLY_STOP)
        callbacks.append(ae_stopper)
    if perfm_train:
        perfm_stopper = EarlyStopping(monitor='perfm_val_loss_avg',
                                      min_delta=s.EARLY_RANGE,
                                      patience=s.EARLY_STOP)
        callbacks.append(perfm_stopper)
    if copy_checkpoint:
        callbacks.append(best_checkpoint_saver(copy_checkpoint))

    trainer = Trainer(
        callbacks=callbacks,
        precision=s.PRECISION,
        max_epochs=s.EPOCHS,
        logger=logger,
        # weights_summary="full",
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        log_gpu_memory=True,
        track_grad_norm=2,
        overfit_batches=10,
        gpus=s.GPUS)

    # training!
    trainer.fit(model)

    # if early stopping interrupted, then we return the loss, otherwise we return -1
    if ae_stopper.stopped_epoch > 0:  # type: ignore
        model.autoencoder.freeze()
        ae_loss = ae_stopper.best_score  # type: ignore
    else:
        ae_loss = -1
    if perfm_stopper.stopped_epoch > 0:  # type: ignore
        model.performer.freeze()
        perfm_loss = perfm_stopper.best_score  # type: ignore
    else:
        perfm_loss = -1
    return ae_loss, perfm_loss


def train(hpar, mode, copy_checkpoint='', generic=False):
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

    # dummy model (baseline)
    # TODO: check the following function!
    # dummy_avg = compute_average(trainloader.dataset,
    #                             *axes,
    #                             n_jobs=-1,
    #                             backend='threading').to(s.DEVICE)

    model = build_model(hpar, mode, generic=generic)
    torchinfo.summary(model)

    # training
    # this loss is also the same used for hyper-parameters tuning
    ae_loss, perfm_loss = my_train(mode, copy_checkpoint, logger, model)
    # if training was stopped by `early-stopping`, freeze that part and finish
    # training the other one
    if (ae_loss == -1) != (perfm_loss == -1):
        _ae_loss, _perfm_loss = my_train(mode, copy_checkpoint, logger, model,
                                         ae_loss == -1, perfm_loss == -1)
        ae_loss = max(ae_loss, _ae_loss)
        perfm_loss = max(perfm_loss, _perfm_loss)

    complexity_loss = count_params(model) * s.COMPLEXITY_PENALIZER
    loss = (ae_loss + perfm_loss) * complexity_loss
    logger.log_metrics({
        "best_ae_loss": float(ae_loss),
        "best_perfm_loss": float(perfm_loss),
        "total_loss": float(loss)
    })

    # this is the loss used by hyper-parameters optimization
    return loss
