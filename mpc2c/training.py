# from memory_profiler import profile

import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch.nn.functional as F  # type: ignore
import torch  # type: ignore
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


def reconstruction_loss(pred, same, diff):
    return max(
        torch.tensor(0.0),
        torch.tensor(1.0) + F.l1_loss(pred, same) - F.l1_loss(pred, diff)).to(pred.device)


def build_autoencoder(hpar, dropout, generic=False):
    if generic:
        loss_fn = lambda pred, _, diff: F.l1_loss(pred, diff)
    else:
        loss_fn = reconstruction_loss
    m = feature_extraction.AutoEncoder(loss_fn=loss_fn,
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
    return (1, (hpar['kernel_0'],
                hpar['kernel_1']), (1, 1), (1, 1), hpar['lstm_hidden_size'],
            hpar['lstm_layers'], hpar['encoder_features'],
            hpar['middle_activation'], hpar['latent_features'])


def build_performer_model(hpar, avg_pred):
    m = feature_extraction.Performer(
        (hpar['performer_features'], hpar['performer_layers'],
         hpar['latent_features'], hpar['middle_activation'], 1), F.l1_loss,
        avg_pred)

    return m


def my_train(mode,
             copy_checkpoint,
             logger,
             model,
             trainloader,
             validloader,
             ae_train=True,
             perfm_train=True):
    """
    Creates callbacks, train and freeze the part that raised an early-stop.
    Return the best loss of that one and 0 for the other.
    """
    checkpoint_saver = ModelCheckpoint(f"checkpoint_{mode}",
                                       filename='{epoch}-{ae_loss:.2f}',
                                       monitor='val_loss',
                                       save_top_k=1,
                                       mode='min',
                                       save_weights_only=True)
    callbacks = [checkpoint_saver]
    # if ae_train:
    #     ae_stopper = EarlyStopping(monitor='ae_val_loss',
    #                                min_delta=s.EARLY_RANGE,
    #                                patience=s.EARLY_STOP)
    #     callbacks.append(ae_stopper)
    # if perfm_train:
    #     perfm_stopper = EarlyStopping(monitor='perfm_val_loss',
    #                                   min_delta=s.EARLY_RANGE,
    #                                   patience=s.EARLY_STOP)
    #     callbacks.append(perfm_stopper)
    if copy_checkpoint:
        callbacks.append(best_checkpoint_saver(copy_checkpoint))

    trainer = Trainer(callbacks=callbacks,
                      precision=s.PRECISION,
                      max_epochs=s.EPOCHS,
                      logger=logger,
                      gpus=s.GPUS)
    trainer.fit(model, trainloader, validloader)
    if ae_stopper.stopped_epoch > 0:  # type: ignore
        model.autoencoder.freeze()
        ae_loss = ae_stopper.best_score  # type: ignore
    else:
        ae_loss = 0
    if perfm_stopper.stopped_epoch > 0:  # type: ignore
        model.performer.freeze()
        perfm_loss = perfm_stopper.best_score  # type: ignore
    else:
        perfm_loss = 0
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

    # loaders
    contexts = list(get_contexts(s.CARLA_PROJ).keys())
    trainloader = data_management.get_loader(['train'], False, contexts, mode)
    validloader = data_management.get_loader(['validation'], False, contexts,
                                             mode)
    print(f"Num batches: {len(trainloader)}, {len(validloader)}")

    # building model
    if mode == 'velocity':
        ae_hpar = get_velocity_hpar(hpar)
        # axes = []
    elif mode == 'pedaling':
        ae_hpar = get_pedaling_hpar(hpar)
        # axes = [-1]
    else:
        raise RuntimeError(f"mode {mode} not known!")

    # dummy model (baseline)
    # TODO: check the following function!
    # dummy_avg = compute_average(trainloader.dataset,
    #                             *axes,
    #                             n_jobs=-1,
    #                             backend='threading').to(s.DEVICE)
    dummy_avg = torch.tensor(0.5).cuda()

    autoencoder = build_autoencoder(ae_hpar, s.TRAIN_DROPOUT, generic)
    performer = build_performer_model(hpar, dummy_avg)

    print(performer)
    print(autoencoder)

    # learning rate
    # lr = s.TRANSFER_LR_K * (s.LR_K / len(trainloader)) * (n_params_all /
    #                                                       n_params_free)
    # lr = lr_k / len(trainloader)
    lr = s.LR

    model = feature_extraction.EncoderDecoderPerformer(autoencoder, performer,
                                                       len(contexts), lr, s.WD)

    # logging initial stuffs
    logger.log_metrics({
        "initial_lr": lr,
        "train_batches": len(trainloader),
        "valid_batches": len(validloader)
    })

    # training
    # this loss is also the same used for hyper-parameters tuning
    ae_loss, perfm_loss = my_train(mode, copy_checkpoint, logger, model,
                                   trainloader, validloader)
    # if training was stopped by `early-stopping`, freeze that part and finish
    # to train the other one
    if ae_loss != 0 or perfm_loss != 0:
        _ae_loss, _perfm_loss = my_train(mode,
                                         copy_checkpoint,
                                         logger,
                                         model,
                                         trainloader,
                                         validloader,
                                         ae_train=ae_loss == 0,
                                         perfm_train=perfm_loss == 0)
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
