# from memory_profiler import profile

import os
from pprint import pprint

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn

from . import feature_extraction
from . import settings as s
from .asmd_resynth import get_contexts

RANDGEN = torch.Generator()


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
            model = None
            try:
                model = model_build_func(hpar)
                print("model created")
                model.eval()
                with torch.no_grad():
                    model.to(s.DEVICE).validation_step(
                        {
                            "x": test_sample.to(s.DEVICE).to(s.DTYPE),
                            "y": torch.tensor(0.5).to(s.DEVICE).to(s.DTYPE),
                            "cont_diff": test_sample.to(s.DEVICE).to(s.DTYPE),
                            "cont_same": test_sample.to(s.DEVICE).to(s.DTYPE),
                            "c": "0",
                        },
                        0,
                        log=False,
                    )
                print("model tested")
            # except Exception as e:
            #     import traceback
            #     traceback.print_exc(e)
            except Exception as e:
                print(e)
                allowed = False
            finally:
                del model

        print(f"hyper-parameters allowed: {allowed}")
        return allowed

    return constraint


def cosine_distance(x, y, reduction="none"):
    out = 1 - F.cosine_similarity(x, y)
    if reduction == "none":
        return out
    elif reduction == "sum":
        return out.sum(axis=-1)


def specific_loss(pred, same_pred, diff_pred):
    return F.triplet_margin_with_distance_loss(
        pred,
        same_pred,
        diff_pred,
        distance_function=cosine_distance,
        margin=1,
        swap=True,
        reduction="sum",
    )


def generic_loss(pred, same_pred, diff_pred):
    if torch.randint(6, (1,), generator=RANDGEN) > 0:
        return cosine_distance(pred, diff_pred, reduction="sum")
    else:
        return cosine_distance(pred, same_pred, reduction="sum")


def build_encoder(hpar, dropout):

    k1, k2, activation, kernel = get_hpar(hpar)

    m = (
        feature_extraction.Encoder(
            insize=(
                s.BINS,
                s.MINI_SPEC_SIZE,
            ),  # mini_spec_size should change for pedaling...
            dropout=dropout,
            k1=k1,
            k2=k2,
            activation=activation,
            kernel=2 * kernel + 1,
        )
        .to(s.DEVICE)
        .to(s.DTYPE)
    )
    # feature_extraction.init_weights(m, s.INIT_PARAMS)
    return m


def get_hpar(hpar):
    return (hpar["enc_k1"], hpar["enc_k2"], hpar["activation"], hpar["kernel"])


def build_specializer_model(hpar, infeatures, loss, nout):
    m = feature_extraction.Performer(
        (
            hpar["performer_features"],
            hpar["performer_layers"],
            infeatures,
            hpar["activation"],
            1,
        ),
        loss,
        nout,
    )

    return m


def build_context_classifier(hpar, infeatures, loss, nout):
    m = feature_extraction.ContextClassifier(
        infeatures, hpar['activation'], hpar['kernel'], nout, loss)
    return m


def build_model(hpar, mode, dropout=s.TRAIN_DROPOUT, context_specific=True):

    contexts = list(get_contexts(s.CARLA_PROJ).keys())
    encoder = build_encoder(hpar, dropout)
    performer = build_specializer_model(
        hpar, encoder.outchannels, nn.L1Loss(reduction="sum"), 1
    )
    cont_classifier = build_context_classifier(
        hpar, encoder.outchannels, nn.CrossEntropyLoss(reduction="sum"), len(contexts)
    )
    model = feature_extraction.EncoderPerformer(
        encoder,
        performer,
        cont_classifier,
        contexts,
        mode,
        context_specific,
        ema_period=s.EMA_PERIOD,
    )
    return model


def my_train(
    mode,
    copy_checkpoint,
    logger,
    model,
    context_specific,
    cont_train=True,
    perfm_train=True,
):
    """
    Creates callbacks, train and freeze the part that raised an early-stop.
    Return the best loss of that one and 0 for the other.
    """
    # stopped_epoch = 9999  # let's enter the while the first time
    # while stopped_epoch > s.EARLY_STOP:
    # setup callbacks
    checkpoint_saver = ModelCheckpoint(
        f"checkpoint_{mode}_context={context_specific}",
        filename="{epoch}-{cont_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )
    callbacks = [checkpoint_saver]
    cont_stopper = perfm_stopper = None
    if cont_train:
        cont_stopper = EarlyStopping(
            monitor="cont_val_loss_avg",
            min_delta=s.EARLY_RANGE,
            check_finite=False,
            patience=s.EARLY_STOP,
        )
        callbacks.append(cont_stopper)
    if perfm_train:
        perfm_stopper = EarlyStopping(
            monitor="perfm_val_loss_avg",
            min_delta=s.EARLY_RANGE,
            check_finite=False,
            patience=s.EARLY_STOP,
        )
        callbacks.append(perfm_stopper)
    # if copy_checkpoint:
    #     callbacks.append(best_checkpoint_saver(copy_checkpoint))

    if s.SWA:
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=int(0.8 * s.EPOCHS),
                annealing_epochs=int(0.2 * s.EPOCHS),
            )
        )

    # training!
    trainer = Trainer(
        callbacks=callbacks,
        precision=s.PRECISION,
        max_epochs=s.EPOCHS,
        logger=logger,
        auto_lr_find=True,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        # gradient_clip_val=0.5,
        # weights_summary="full",
        # log_every_n_steps=1,
        # log_gpu_memory=True,
        # track_grad_norm=2,
        overfit_batches=60,
        # fast_dev_run=True,
        gpus=s.GPUS,
    )

    # model.njobs = 1  # there's some leak when using njobs > 0
    # if os.path.exists("lr_find_temp_model.ckpt"):
    #     os.remove("lr_find_temp_model.ckpt")
    # d = trainer.tune(model, lr_find_kwargs=dict(min_lr=1e-7, max_lr=1))
    # if d["lr_find"] is None or d["lr_find"].suggestion() is None:
    model.lr = 1
    model.learning_rate = 1
    model.njobs = s.NJOBS
    # need to reload dataloaders for using multiple jobs
    # trainer.train_dataloader = model.train_dataloader()
    # trainer.val_dataloader = model.val_dataloader()
    # trainer.test_dataloader = model.test_dataloader()
    print("Fitting the model!")
    trainer.fit(model)

    return cont_stopper, perfm_stopper


def train(hpar, mode, context_specific, copy_checkpoint="", test=True):
    """
    1. Builds a model given `hpar` and `mode`
    2. Train the model
    3. Saves the trained model weights to `copy_checkpoint`
    4. Test the trained model if `test is True`
    4. Returns the best validation loss (or test loss if `test` is True)
    """
    # the logger
    logger = MLFlowLogger(
        experiment_name=f"{mode}", tracking_uri=os.environ.get("MLFLOW_TRACKING_URI")
    )

    logger.log_hyperparams(hpar)
    logger.log_hyperparams({"mode": mode, "context_specific": context_specific})

    model = build_model(hpar, mode, context_specific=context_specific)
    # torchinfo.summary(model)

    # training
    # this loss is also the same used for hyper-parameters tuning
    cont_stopper, perfm_stopper = my_train(
        mode,
        copy_checkpoint,
        logger,
        model,
        context_specific,
        cont_train=context_specific,
        perfm_train=True,
    )

    return 1.0

    if cont_stopper and perfm_stopper:
        # cases:
        # A: encoder was stopped
        # B: performers were stopped
        # C: none was stopped

        print(f"First-training losses: {cont_stopper.best_score:.2e}, {perfm_stopper.best_score:.2e}")
        if cont_stopper.stopped_epoch == 0 and perfm_stopper.stopped_epoch > 0:  # type: ignore
            # case A
            for p in model.performers.values():
                p.freeze()
            for c in model.context_classifiers.values():
                c.freeze()
            print("Continuing training encoder...")
            cont_stopper, _ = my_train(
                mode, copy_checkpoint, logger, model, context_specific, True, False
            )
        if cont_stopper.stopped_epoch > 0:  # type: ignore
            # case A and B
            model.encoder.freeze()
            print("Continuing training performers...")
            _, perfm_stopper = my_train(
                mode, copy_checkpoint, logger, model, context_specific, False, True
            )

        cont_loss, perfm_loss = cont_stopper.best_score, perfm_stopper.best_score  # type: ignore

        loss = cont_loss + perfm_loss
        print(f"Final losses: {cont_loss:.2e}, {perfm_loss:.2e}")
        logger.log_metrics(
            {
                "best_cont_val_loss": float(cont_loss),  # type: ignore
            }
        )
    elif perfm_stopper:
        print(f"First-training loss: {perfm_stopper.best_score:.2e}")
        perfm_loss = loss = perfm_stopper.best_score
    else:
        # not reachable here only to shutup the linter
        loss = 0.0

    logger.log_metrics(
        {
            "best_perfm_val_loss": float(perfm_loss),  # type: ignore
        }
    )

    if test:
        trainer = Trainer(precision=s.PRECISION, logger=logger, gpus=s.GPUS)
        loss = trainer.test(model)[0]["perfm_test_avg"]

    logger.log_metrics(
        {
            "final_weight_variance_" + k: float(v)
            for k, v in model.performer_weight_moments().items()
        }
    )

    # this is the loss used by hyper-parameters optimization
    return float(loss)
