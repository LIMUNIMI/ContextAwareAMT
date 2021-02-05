import pathlib
from typing import Callable, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


class DatasetDump(TorchDataset):
    """
    This class is useful to save preprocessed datasets.

    It takes a dataset (an `asmd.Dataset` object) and a path where reprocessed
    dataset samples will be saved.  You should then call the method 'dump'
    which preprocess the dataset and dumps it to files.  Finally, you should
    iterate over this object (it's an Iterable).

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
            self.__song_iter__ = iter(range(len(dataset.paths)))
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
        y = np.load(self.get_folder(i) / f"y{i}.npz")['arr_0']
        y = torch.from_numpy(y)
        return y

    def get_input(self, i):
        x = np.load(self.get_folder(i) / f"x{i}.npz")['arr_0']
        x = torch.from_numpy(x)
        return x

    def __getitem__(self, i):
        if self.dumped:
            x = self.get_input(i)
            y = self.get_target(i)
        elif self.process_fn is not None:
            if self.__counter__ == len(self.__xx__):
                # reset counter and load new song
                self.__xx__, self.__yy__ = self.process_fn(
                    next(self.__song_counter__), self.dataset)
                self.__counter__ = 0
            # increase counter and return new sample from the already loaded
            # song
            x = self.__xx__[self.__counter__]
            y = self.__yy__[self.__counter__]
            self.__counter__ += 1
        else:
            raise RuntimeError("Dataset not dumped and no `process_fn` known")
        return x, y

    def __len__(self):
        return sum(self.num_samples)

    def itertargets(self):
        for i in range(len(self)):
            yield self.get_target(i)

    def iterinputs(self):
        for i in range(len(self)):
            yield self.get_input(i)


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
    xx, yy = zip(*batch)
    return [
        torch.stack(xx),
    ], [
        torch.stack(yy),
    ], [
        torch.tensor(False),
    ]
