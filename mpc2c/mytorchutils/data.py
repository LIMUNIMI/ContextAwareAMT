import pathlib
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


class Iterator:
    """
    An iterator with an arbitrary number of counters.
    The first counter (index 0) is the one which will
    cause the `StopIteration` if it is larger than the `len` of this object --
    it means that the first counter must be in pair with the `length` argument
    """

    def __init__(self, n_counters, length, func):
        self.counters = [0] * n_counters
        self.length = length
        self.func = func

    def __len__(self):
        return self.length

    def __next__(self):
        if self.counters[0] == len(self):
            raise StopIteration
        ret = self.func(self.counters)
        return ret

    def __iter__(self):
        return self


class DumpableDataset(TorchDataset):
    """
    This class is useful to save preprocessed datasets.

    It takes a dataset (an `asmd.Dataset` object) and a path where reprocessed
    dataset samples will be saved.  You should then call the method 'dump'
    which preprocess the dataset and dumps it to files.  Finally, you should
    iterate over this object (it has special methods to get iterators from it).

    The dataset can also be used without dumping if you provide `process_fn`

    This allows for fast dumping in loop of complex objects (such as
    collections of samples where the cardinality of each collections is
    variable and each sample has a huge number of features).

    TODO: add ability to dump samples on the first run.

    Arguments
    ----------
    `num_samples` : list[int] or None
        a list containing the number of samples in each song (e.g. number
        of notes, frames or whatelse depending on what is your concept of
        sample); if `None` (default), one sample per song is considered

    `process_fn` : Callable or None
        a function that will be used to process songs if the dataset has not
        been dumped. It should only accept an int and an `asmd.Dataset` object.
    """

    def __init__(self,
                 dataset,
                 root: str,
                 dumped: bool = False,
                 num_samples=None,
                 folder_size=1500,
                 process_fn: Optional[Callable] = None):
        super().__init__()
        self.dataset = dataset
        self.dumped = dumped
        self.root = pathlib.Path(root)
        self.process_fn = process_fn
        if self.process_fn is not None:
            # setup stuffs for iterate samples over songs
            self.__song_counter__ = 0
            self.__counter__ = 0
            self.__xx__: list = []
            self.__yy__: list = []
        if not num_samples:
            self.num_samples = [1] * len(self.dataset)
        else:
            self.num_samples = num_samples

        self.folder_size = folder_size

    def dump(self, process_fn, *args, **kwargs):
        """
        Preprocess data and dumps them to file.

        Arguments
        ---------

        `process_fn` : Callable
            a function which is used to preprocess each song in the dataset;
            this should respect the asmd.Dataset.parallel structure: its first
            argument should be an int (the index of the song), and the second
            argument should be the dataset itself

        `*args` : any
            additional arguments for `process_fn`

        `**kwargs` : any
            additional key-word arguments for `process_fn` and joblib
        """
        if not self.dumped:
            print(f"Dumping data to {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)

            def pickle_fn(i, dataset, get_data_fn, *args, **kwargs):
                xx, yy = get_data_fn(i, dataset, *args, **kwargs)
                index = sum(self.num_samples[:i])
                for j in range(len(xx)):
                    x = xx[j]
                    y = yy[j]
                    dest_path = self.get_folder(index)
                    dest_path.mkdir(parents=True, exist_ok=True)
                    path_x = dest_path / f"x{index}.npz"
                    path_y = dest_path / f"y{index}.npz"
                    # this while prevents filesystem errors
                    # while not path_x.exists() or not path_y.exists():
                    np.savez(path_x, x)
                    np.savez(path_y, y)
                    index += 1

            self.dataset.parallel(pickle_fn, process_fn, *args, **kwargs)

            self.dumped = True

    def get_folder(self, index: int):
        """
        Returns the `Path` of a folder where a certain index has been saved.

        This is useful for managing large indices..
        """
        return self.root / str(index // self.folder_size)

    def get_target(self, i):
        """
        Randomly access a given sample. If not dumped, this has to look for the
        song containing the sample and to preprocess it.
        """
        if self.dumped:
            y = np.load(self.get_folder(i) / f"y{i}.npz")['arr_0']
        elif self.process_fn is not None:
            self.__counter__ = self._load_song_containing_sample(i)
            y = self.__yy__[self.__counter__]
        else:
            raise RuntimeError("Dataset not dumped and no `process_fn` known")
        y = torch.from_numpy(y)

        return y

    def get_input(self, i):
        """
        Randomly access a given sample. If not dumped, this has to look for the
        song containing the sample and to preprocess it.
        """
        if self.dumped:
            x = np.load(self.get_folder(i) / f"x{i}.npz")['arr_0']
        elif self.process_fn is not None:
            self.__counter__ = self._load_song_containing_sample(i)
            x = self.__xx__[self.__counter__]
        else:
            raise RuntimeError("Dataset not dumped and no `process_fn` known")
        x = torch.from_numpy(x)

        return x

    def __getitem__(self, i):
        """
        Randomly access a given sample. If not dumped, this has to look for the
        song containing the sample and to preprocess it.
        """
        x = self.get_input(i)
        y = self.get_target(i)
        return x, y

    def _load_song_containing_sample(self, i):
        # compute the song containing sample `i`
        sum = 0
        for k, num in enumerate(self.num_samples):
            sum += num
            if sum > i:
                sum -= num

        if self.__song_counter__ != k - 1:
            self.__song_counter__ = k - 1
            self.__xx__, self.__yy__ = self.process_fn(self.__song_counter__,
                                                       self.dataset)

        return i - sum

    def _get_single_next(self, func):
        """
        Returns a function that can be used in `Iterator` as `__next__`

        `func` is a Callable called with as argument the counter
        """
        def _next(counters):
            ret = func(counters[0])
            counters[0] += 1
            return ret

        return _next

    def _get_double_next(self, func):
        """
        Returns a function that can be used in `Iterator` as `__next__`

        `func` is a Callable called with as argument the second counter
        """
        self.__xx__, self.__yy__ = self.process_fn(0, self.dataset)

        def _next(counters):
            if counters[1] == self.num_samples[counters[0]]:
                counters[0] += 1
                self.__xx__, self.__yy__ = self.process_fn(
                    counters[0], self.dataset)
                counters[1] = 0
            ret = func(counters[1])
            counters[1] += 1
            return ret

        return _next

    def __len__(self):
        return sum(self.num_samples)

    def iter_dumped(self):
        """
        Returns an iterator for dumped datasets
        """
        def func(x):
            return self.get_input(x), self.get_target(x)
        return Iterator(1, len(self), self._get_single_next(func))

    def iter_not_dumped(self):
        """
        Returns an iterator for non-dumped datasets (much faster than random access)
        """
        def func(x):
            return torch.from_numpy(self.__xx__[x]), torch.from_numpy(
                self.__yy__[x])

        return Iterator(2, len(self.dataset.paths),
                        self._get_double_next(func))

    def itertargets(self):
        """
        Iterate only the targets (dumped or not)
        """
        if self.dumped:
            return Iterator(1, len(self),
                            self._get_single_next(self.get_target))
        elif self.process_fn is not None:

            return Iterator(
                2, len(self.dataset.paths),
                self._get_double_next(
                    lambda x: torch.from_numpy(self.__yy__[x])))
        else:
            raise RuntimeError("Dataset not dumped and no `process_fn` known")

    def iterinputs(self):
        """
        Iterate only the inputs (dumped or not)
        """
        if self.dumped:
            return Iterator(1, len(self),
                            self._get_single_next(self.get_input))
        elif self.process_fn is not None:

            return Iterator(
                2, len(self.dataset.paths),
                self._get_double_next(
                    lambda x: torch.from_numpy(self.__xx__[x])))
        else:
            raise RuntimeError("Dataset not dumped and no `process_fn` known")


def pad_collate(batch):
    xx, yy = zip(*batch)
    xx = list(xx)
    yy = list(yy)
    lens = [x.shape[-1] for x in xx]

    m = max(lens)
    with torch.no_grad():
        for i in range(len(xx)):
            pad = m - xx[i].shape[-1]
            xx[i] = F.pad(xx[i], (0, 0, 0, pad), 'constant', 0)
            yy[i] = F.pad(yy[i], (0, 0, 0, pad), 'constant', 0)
    x_pad = torch.stack(xx)
    y_pad = torch.stack(yy)

    return [
        x_pad,
    ], [
        y_pad,
    ], [
        lens,
    ]


def dummy_collate(batch):
    # batch is a list of tuples (x, y)
    # let's transform it in two lists, one for x and one for y
    xx, yy = zip(*batch)
    xx = list(xx)
    yy = list(yy)

    # if there are multiple x or y in each xx or yy,
    # we need to have separated list for each of them
    if type(xx[0]) is tuple:
        xx = zip(*xx)
    if type(yy[0]) is tuple:
        yy = zip(*yy)
    return [torch.stack(xx[i]) for i in range(len(xx))
            ], [torch.stack(yy[i]) for i in range(len(yy))], [
                torch.tensor(False),
    ]
