import pathlib

import numpy as np

import torch
import torch.nn.functional as F
from asmd import asmd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from . import nmf
from . import settings as s
from . import utils


def process_pedaling(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    diff_spec = nmf_tools.V - nmf_tools.W @ nmf_tools.H
    winlen = s.FRAME_SIZE / s.SR
    hop = s.HOP_SIZE / s.SR
    pedaling = dataset.get_pedaling(i,
                                    frame_based=True,
                                    winlen=winlen,
                                    hop=hop)[0]
    # padding so that pedaling and diff_spec have the same length
    pedaling, diff_spec = utils.pad(pedaling[:, 1:].T, diff_spec)
    return diff_spec[None], pedaling[None]


def process_velocities(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    velocities = dataset.get_score(i, score_type=['precise_alignment'])[:, 3]
    minispecs = nmf_tools.get_minispecs()
    return minispecs, velocities


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
    """

    def __init__(self,
                 dataset: asmd.Dataset,
                 root: str,
                 dumped: bool = False,
                 num_samples=None,
                 folder_size=1500):
        super().__init__()
        self.dataset = dataset
        self.dumped = dumped
        self.root = pathlib.Path(root)
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
                    # this while prevents filesystm errors
                    while not path_x.exists() or not path_y.exists():
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

    def __getitem__(self, i):
        assert self.dumped, "Dataset not dumped!"
        x = np.load(self.get_folder(i) / f"x{i}.npz")['arr_0']
        y = np.load(self.get_folder(i) / f"y{i}.npz")['arr_0']
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return sum(self.num_samples)


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

    return x_pad, y_pad, lens


def dummy_collate(batch):
    xx, yy = zip(*batch)
    return torch.stack(xx), torch.stack(yy), torch.tensor(False)


def get_loader(groups, nmf_params, mode):
    dataset = asmd.Dataset().filter(datasets=s.DATASETS, groups=groups)

    if mode == 'velocity':
        num_samples = [
            len(gt['precise_alignment']['pitches'])
            for i in range(len(dataset.paths)) for gt in dataset.get_gts(i)
        ]
        velocity_dataset = DatasetDump(dataset,
                                       pathlib.Path(s.VELOCITY_DATA_PATH) /
                                       "_".join(groups),
                                       not s.REDUMP,
                                       num_samples=num_samples)
        # max_nbytes=None disable shared memory for large arrays
        velocity_dataset.dump(process_velocities,
                              nmf_params,
                              n_jobs=s.NJOBS,
                              max_nbytes=None)
        return DataLoader(velocity_dataset,
                          batch_size=s.VEL_BATCH_SIZE,
                          num_workers=s.NJOBS,
                          pin_memory=True,
                          collate_fn=dummy_collate)
    elif mode == 'pedaling':
        pedaling_dataset = DatasetDump(dataset,
                                       pathlib.Path(s.PEDALING_DATA_PATH) /
                                       "_".join(groups),
                                       not s.REDUMP,
                                       num_samples=None)
        # max_nbytes=None disable shared memory for large arrays
        pedaling_dataset.dump(process_pedaling,
                              nmf_params,
                              n_jobs=s.NJOBS,
                              max_nbytes=None)
        return DataLoader(pedaling_dataset,
                          batch_size=s.PED_BATCH_SIZE,
                          num_workers=s.NJOBS,
                          pin_memory=True,
                          collate_fn=pad_collate)
