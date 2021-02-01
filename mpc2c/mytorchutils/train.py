from time import time
from typing import Callable, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from . import context
from .data import DatasetDump


def compute_average(dataloader, *axes: int, **joblib_kwargs):
    """
    A functional interface to AveragePredictor using dataloaders
    """
    predictor = AveragePredictor(*axes)
    predictor.add_dataloader(dataloader, **joblib_kwargs)
    return predictor.predict()


class AveragePredictor(object):
    """
    A simple predictor which computes an average and use that one for any
    sample
     Doesn't support multiple targets for now.

    Example:

    .. code:
        predictor = AveragePredictor()
        for sample in samples:
            predictor.add_to_average(sample)
        predictor.predict()

        # if you add other samples, you also need to manually update the
        # tracking average:
        for sample in new_samples:
            predictor.add_to_average(sample)
        predictor.update_tracking_avg()
        predictor.predict()
    """

    def __init__(self, *axes: int):
        """
        `axes` : int
            the axes along which the data will be summed
        """
        self.axes = axes
        self.__sum_values__: torch.Tensor = None
        self.__counter__: int = 0

    def update_tracking_avg(self):
        self.__avg__ = self.__sum_values__ / self.__counter__
        for ax in self.axes:
            self.__avg__ = self.__avg__.unsqueeze(ax)

    def add_to_average(self, sample: torch.Tensor, update_tracking_avg=False):
        if self.__sum_values__ is None:
            self.__sum_values__ = sample.sum(dim=self.axes)
        else:
            self.__sum_values__ += sample.sum(dim=self.axes)
        # computing the number of elements that were summed
        if self.axes:
            size = 1
            for ax in self.axes:
                size *= sample.shape[ax]
        else:
            size = sample.numel()

        self.__counter__ += size
        if update_tracking_avg:
            self.update_tracking_avg()

    def add_dataloader(self, dataset: DatasetDump,
                       **joblib_kwargs):
        """
        Add the targets retrieved by the DatasetDump object.

        `joblib_kwargs` are keyword arguments for joblib.Parallel

        N.B. DatasetDump allows to iterate over targets only, making the
        loading of data much lighter.
        """

        def proc(self, targets):
            self.add_to_average(targets[None])
            # for i, target in enumerate(targets):
            #     if lens[i] == torch.tensor(False):
            #         # None here so that the batch dimension is kept and the
            #         # predicted value still has it
            #         self.add_to_average(target[None])
            #     else:
            #         for batch, L in enumerate(lens[i]):
            #             # None here so that the batch dimension is kept and the
            #             # predicted value still has it
            #             self.add_to_average(target[None, batch, :L])
            return self.__sum_values__, self.__counter__

        out = Parallel(**joblib_kwargs)(
            delayed(proc)(type(self)(*self.axes), targets)
            for targets in dataset.itertargets())

        # `out` is:
        #   List[Tuple[float, float]]
        # `*out` is:
        #   Tuple[float, float], Tuple[float, float], Tuple[float, float], ...
        # `zip(*out)` is:
        #   Tuple[float, float, float, ...], Tuple[float, float, float, ...]
        out = list(zip(*out))
        self.__sum_values__ = sum(out[0])
        self.__counter__ = sum(out[1])
        self.update_tracking_avg()

    def predict(self, *x):
        if not hasattr(self, '__avg__'):
            self.update_tracking_avg()
        return self.__avg__


def count_params(model):
    """
    Compute the number of parameters
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


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
                 dummy_loss: Optional[Callable] = None,
                 trainloss_on_valid: bool = False):
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

    `dummy_loss` : None or Callable
        if Callable, it should accept the list of targets and return one value
        that is used as dummy prediction; this computes the dummy loss against
        the average of the target

    `trainloss_on_valid` : bool
        if True, this computes the train loss on the validation set too

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
                if dummy_loss:
                    dummy_out = [
                        dummy_loss(targets).expand_as(out[i]).to(device)
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
        if dummy_loss:
            dl = np.mean(dummyloss)
            print(f"dummy loss : {dl:.4e}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            state_dict = model.state_dict()
            name = f"checkpoints/checkpoint{best_loss:.4f}.pt"
            torch.save({'dtype': dtype, 'state_dict': state_dict}, name)
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
    return best_loss


def plot_losses_func(*losses, epoch=0):
    context.vis.line(torch.tensor([losses]),
                     X=torch.tensor([epoch]),
                     update='append',
                     win="losses",
                     opts=dict(legend=[f'loss{i}' for i in range(len(losses))],
                               title="losses!"))
