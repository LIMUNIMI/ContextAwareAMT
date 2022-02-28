import shutil
from time import time
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from . import context


def count_params(model):
    """
    Compute the number of parameters
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


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


def train_epochs(model,
                 optim,
                 trainloss_fn,
                 validloss_fn,
                 trainloader,
                 validloader,
                 epochs: int = 500,
                 early_stop: int = 15,
                 plot_losses: bool = True,
                 device: str = 'cuda',
                 dtype=torch.float32,
                 dummy_predictor: Optional[Callable] = None,
                 trainloss_on_valid: bool = False,
                 copy_checkpoint: bool = True):
    """
    A typical training algorithm with early stopping and loss plotting

    Arguments
    ---------
    `model` : torch.nn.Model

    `optim` : torch.optim.Optimizer

    `trainloss_fn` : callable
        a loss function which accepts 3 arguments:

        * targets
        * predictions
        * length of the inputs along the batch dimension

    `validloss_fn` : callable
        a loss function which accepts 3 arguments:

        * targets
        * predictions
        * length of the inputs along the batch dimension

    `trainloader` : torch.utils.data.DataLoader
        must yield:

        * `inputs` : list
            a list of tensors that will be used as inputs to the model (nn
            that need multiple inputs can still use this!): each input will be
            an argument to the model forward pass
        * `targets` : list
            a list of tensors that will be passed to the loss functions as it
            is (nn that have intermediate targets can still use that!)
        * `lens` : list
            a list of tensors where each element contains the length of the
            corresponding target sample

    `validloader` : torch.utils.data.DataLoader
        same as `trainloader` but used for validation

    `epochs` : int
        the maximum number of epochs (defaults to 500)

    `early_stop` : int
        how many consecutive epochs with valid loss greater of the lowest valid
        loss are allowed before of stopping the training (defaults to 15)

    `plot_losses` : bool
        if True, use `visdom` to plot losses while training (defaults to True)

    `device` : str
        the device that will be used, e,g. `cuda`, `cpu`, `cuda:0` `cuda:1`
        (defaults to 'cuda')

    `dtype` : torch.dtype
        the dtype to use for the data (defaults to torch.float32)

    `dummy_predictor` : None or Callable
        if Callable, it should accept the list of targets and return one value
        that is used as dummy prediction for all of them; it will compute the
        validation loss against the dummy prediction. You can use this option
        to compute the compare with a baseline predicting the average of the
        targets.

    `trainloss_on_valid` : bool
        if True, this computes the train loss on the validation set too

    `copy_checkpoint` : bool
        if True, the best checkpoint is saved to the working directory

    Note
    ----
    This algorithms correctly handles cases in which each sample has a
    different length.

    Moreover, it allows to use two different losses for training and
    validation. The method will use the training loss also during validation
    for comparison purposes; however, sometimes you are interested in metric
    different from the loss.

    You should thus use the loss that suits best the backpropagation
    algorithm as training loss, and your target metric as validation loss; this
    approach allows you to monitor the metric of interest during validation, to
    use early stopping against your metric of interest and to monitor the
    validation comparing it with the training.

    This function prints losses and optionally plots them. It also computes a
    `dummy` loss as the valid loss function when the result is the mean of the
    target. You should take care that your valid loss is less than this dummy
    loss.
    """
    best_epoch = 0
    best_loss = 9999
    for epoch in range(epochs):
        epoch_ttt = time()
        print(f"-- Epoch {epoch} --")
        trainloss, validloss, trainloss_valid, dummyloss = [], [], [], []
        print("-> Training")
        model.train()
        for inputs, targets, lens in tqdm(trainloader):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device).to(dtype)
            for i in range(len(targets)):
                targets[i] = targets[i].to(device).to(dtype)

            optim.zero_grad()
            out = model(*inputs, *lens)
            loss = trainloss_fn(out, targets, lens)
            loss.backward()
            optim.step()
            loss = loss.detach().cpu().numpy()
            if np.isnan(loss):
                raise RuntimeError("Nan in training loss!")
            trainloss.append(loss)

        trainloss = np.mean(trainloss)
        print(f"training loss : {trainloss:.4e}")

        print("-> Validating")
        values_same = []
        with torch.no_grad():
            model.eval()
            for inputs, targets, lens in tqdm(validloader):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device).to(dtype)
                for i in range(len(targets)):
                    targets[i] = targets[i].to(device).to(dtype)

                out = model.predict(*inputs, *lens)
                values_same.append(
                    any([torch.min(x) == torch.max(x) for x in out]))

                loss = validloss_fn(out, targets, lens).detach().cpu().numpy()
                validloss.append(loss)
                if np.isnan(loss):
                    raise RuntimeError("Nan in training loss!")
                if dummy_predictor:
                    dummy_out = [
                        dummy_predictor(targets).expand_as(out[i]).to(device)
                        for i in range(len(targets))
                    ]
                    loss = validloss_fn(dummy_out, targets,
                                        lens).detach().cpu().numpy()
                    dummyloss.append(loss)
                if trainloss_on_valid:
                    out = model(*inputs, *lens)
                    trainloss_valid.append(
                        trainloss_fn(out, targets,
                                     lens).detach().cpu().numpy())

        if any(values_same):
            print("Warning: all the predicted values are the same in at least \
one output in at least one validation batch!")
        if all(values_same):
            print("Warning: all the predicted values are the same in at least \
one output in all the validation batches!")

        validloss = np.mean(validloss)
        print(f"validation loss : {validloss:.4e}")
        if trainloss_on_valid:
            trainloss_valid = np.mean(trainloss_valid)
            print(f"validation-training loss : {trainloss_valid:.4e}")
        if dummy_predictor:
            dl = np.mean(dummyloss)
            print(f"dummy loss : {dl:.4e}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            state_dict = model.state_dict()
            fname = f"checkpoints/checkpoint{best_loss:.4f}.pt"
            torch.save({'dtype': dtype, 'state_dict': state_dict}, fname)
        elif epoch - best_epoch > early_stop:
            print("-- Early stop! --")
            break
        else:
            print(f"{epoch - best_epoch} from early stop!!")

        if plot_losses:
            plot_losses_func(trainloss,
                             validloss,
                             trainloss_valid,
                             epoch=epoch)

        print("Time for this epoch: ", time() - epoch_ttt)

    if copy_checkpoint:
        shutil.copy(fname, './')
    return best_loss


def plot_losses_func(*losses, epoch=0):
    context.vis.line(torch.tensor([losses]),
                     X=torch.tensor([epoch]),
                     update='append',
                     win="losses",
                     opts=dict(legend=[f'loss{i}' for i in range(len(losses))],
                               title="losses!"))
