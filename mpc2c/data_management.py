import pathlib
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

import numpy as np
from asmd import asmd

from . import nmf
from . import settings as s
from . import utils


def process_pedaling(i, dataset, nmf_tools):
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['non_aligned'])
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


def process_velocities(i, dataset, nmf_tools):
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['non_aligned'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    velocities = dataset.get_score(i, score_type=['precise_alignment'])[:, 3]
    minispecs = nmf_tools.get_minispecs()
    return minispecs, velocities


class DatasetDump(TorchDataset):
    def __init__(self, dataset: Iterable, root: str, dumped: bool = False):
        super().__init__()
        self.dataset = dataset
        self.dumped = dumped
        self.root = pathlib.Path(root)

    def dump(self, process_fn, nmf_tools, num_samples):
        """
        Generate data and dumps them to file.
        """
        if not self.dumped:
            print(f"Dumping data to {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)

            def pickle_fn(i, dataset, get_data_fn, nmf_tools):
                xx, yy = get_data_fn(i, dataset, nmf_tools)
                index = sum(num_samples[:i])
                for j in range(len(xx)):
                    x = xx[j]
                    y = yy[j]
                    index += j
                    np.savez(self.root / f"x{index}.npz", x)
                    np.savez(self.root / f"y{index}.npz", y)

            # max_nbytes=None disable shared memory for large arrays
            self.dataset.parallel(pickle_fn,
                                  process_fn,
                                  nmf_tools,
                                  n_jobs=s.NJOBS,
                                  max_nbytes=None)

            self.dumped = True

    def __getitem__(self, i):
        assert self.dumped, "Dataset not dumped!"
        x = np.load(self.root / f"x{i}.npz")
        y = np.load(self.root / f"y{i}.npz")
        return x, y


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


def reshape_collate(batch):
    xx, yy = zip(*batch)
    return xx[0], yy[0], None


def get_loader(groups, nmf_params, mode):
    nmf_tools = nmf.NMFTools(*nmf_params)
    dataset = asmd.Dataset().filter(datasets=s.DATASETS, groups=groups)
    if mode == 'velocity':
        num_samples = [
            len(gt['pitches']) for i in range(len(dataset.paths))
            for gt in dataset.get_gts(i)
        ]
        velocity_dataset = DatasetDump(dataset, s.VELOCITY_DATA_PATH,
                                       not s.REDUMP)
        velocity_dataset.dump(process_velocities, nmf_tools, num_samples)
        return DataLoader(
            velocity_dataset,
            batch_size=1,
            # batch_size=s.BATCH_SIZE,
            num_workers=s.NJOBS,
            pin_memory=True,
            collate_fn=reshape_collate)
    elif mode == 'pedaling':
        num_samples = [1 for i in range(len(dataset.paths))]
        pedaling_dataset = DatasetDump(dataset, s.PEDALING_DATA_PATH,
                                       not s.REDUMP)
        pedaling_dataset.dump(process_pedaling, nmf_tools, num_samples)
        return DataLoader(pedaling_dataset,
                          batch_size=s.BATCH_SIZE,
                          num_workers=s.NJOBS,
                          pin_memory=True,
                          collate_fn=pad_collate)
