import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from .data import DumpableDataset


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

    def add_dataloader(self, dataset: DumpableDataset, **joblib_kwargs):
        """
        Add the targets retrieved by the DumpableDataset object.

        `joblib_kwargs` are keyword arguments for joblib.Parallel

        N.B. DumpableDataset allows to iterate over targets only, making the
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
            #             # None here so that the batch dimension is kept and
            #             the # predicted value still has it
            #             self.add_to_average(target[None, batch, :L])
            return self.__sum_values__, self.__counter__

        out = Parallel(**joblib_kwargs)(
            delayed(proc)(type(self)(*self.axes), targets)
            for targets in tqdm(dataset.itertargets()))

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


def test(model,
         testloader,
         testloss_fn,
         dummy_loss=None,
         device='cuda',
         dtype=torch.float32,
         return_predictions=False):
    """
    docs to do, but similar to train function
    """
    values_same = []
    dummyloss = []
    testloss = []
    predictions = []
    with torch.no_grad():
        model.eval()
        for inputs, targets, lens in tqdm(testloader):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device).to(dtype)
            for i in range(len(targets)):
                targets[i] = targets[i].to(device).to(dtype)

            out = model.predict(*inputs, *lens)
            predictions.append(out)
            values_same.append(any([torch.min(x) == torch.max(x)
                                    for x in out]))

            loss = testloss_fn(out, targets, lens).detach().cpu().numpy()
            testloss.append(loss.detach().cpu().numpy())
            if np.isnan(loss):
                raise RuntimeError("Nan in training loss!")
            if dummy_loss:
                dummy_out = [
                    dummy_loss(targets).expand_as(out[i]).to(device)
                    for i in range(len(targets))
                ]
                loss = testloss_fn(dummy_out, targets,
                                   lens).detach().cpu().numpy()
                dummyloss.append(loss)

    if any(values_same):
        print("Warning: all the predicted values are the same in at least \
one output in at least one testation batch!")
        if all(values_same):
            print("Warning: all the predicted values are the same in at least \
one output in all the testation batches!")

        testloss = np.mean(testloss)
        print(f"testing loss : {testloss:.4e}")
        if dummy_loss:
            dl = np.mean(dummyloss)
            print(f"dummy loss : {dl:.4e}")
    if return_predictions:
        return testloss, predictions
    else:
        return testloss
