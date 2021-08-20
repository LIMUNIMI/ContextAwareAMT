import pickle
from typing import List
from itertools import cycle
from random import choice
from time import time

import numpy as np
import essentia as es  # type: ignore
from torch.utils.data import DataLoader, Sampler  # type: ignore

from . import nmf
from . import settings as s
from . import utils
from .asmd.asmd import asmd, dataset_utils
from .mytorchutils import DatasetDump

# import heartrate
# heartrate.trace(port=8080)

class AEDataset(DatasetDump):
    def __init__(self, contexts: List[str], *args, **kwargs):
        """
        if `generic`, then data are generated for generic independence,
        otherwise for specific indipendence
        """
        super().__init__(*args, **kwargs)
        self.contexts = contexts
        # keep track of which samples among those dumped were used
        self.not_used = np.ones(sum(self.lengths), dtype=np.bool8)
        # keep track of the context of each sample
        fname = self.root / 'sample_contexts.pkl'
        if fname.exists():
            self.sample_contexts = pickle.load(open(fname, 'rb')).astype(np.int64)
        else:
            self.sample_contexts = np.zeros(self.not_used.shape[0], dtype=np.int64)
            for idx in range(self.sample_contexts.shape[0]):
                # get context of idx
                song, _ = self._get_song_indices(idx, filtered=True)
                context = self.songs[song]['groups'][-1]
                self.sample_contexts[idx] = self.contexts.index(context)
            pickle.dump(self.sample_contexts, open(fname, 'wb'))

    def __getitem__(self, idx):
        """
        Returns 4 data: input, target label, target reconstruction (same and different context)

        the input is of context `c`

        raise `StopIteration` when finished
        """

        # computing:
        c = self.sample_contexts[idx]
        # 1. dataset with the same context
        same = self.set_operation(dataset_utils.filter, groups=[self.contexts[c]])
        # 2.  dataset with different contexts (contains data not in this split too)
        different = same.set_operation(dataset_utils.complement)  # type: ignore
        # 3.  dataset with different contexts (only in this split)
        different = self.set_operation(dataset_utils.intersect,
                                       different.dataset)  # type: ignore
        # the part above takes 1 second each (due to the creation of `inverted`
        # object, totalling 3 seconds)

        # find the first sample not used in `same` and in `different`

        # both `same`, `different` and this dataset were filtered from the same
        # dumped dataset, so the `original` index is the same

        x = same.get_input(idx, filtered=False)
        y = same.get_target(idx, filtered=False)

        # take a random sample from `different` with the same label
        bin = self.get_bin(y)
        diff_target = choice(different.inverted[bin])
        # take a random sample from `same` with the same label
        same_target = choice(same.inverted[bin])
        ae_same = same.get_input(*same_target)
        ae_diff = different.get_input(*diff_target)

        return {
            "c": str(c),
            "x": x,
            "y": y,
            "ae_same": ae_same,
            "ae_diff": ae_diff
        }


class AEBatchSampler(Sampler):
    def __init__(self, batch_size: int, ae_dataset: AEDataset):
        """
        Makes batches so that each one has a different context
        """
        super().__init__(ae_dataset)
        self.batch_size = batch_size
        self.ae_dataset = ae_dataset
        self.contexts = cycle(ae_dataset.contexts)

    def __len__(self):
        return len(self.ae_dataset) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        c = self.ae_dataset.contexts.index(next(self.contexts))
        # find the first `self.batch_size` samples not used and having context `c`
        batch = np.argwhere(
            np.logical_and(self.ae_dataset.not_used,
                           (self.ae_dataset.sample_contexts == c)))[:self.batch_size, 0]
        if batch.shape[0] == 0:
            raise StopIteration
        self.ae_dataset.not_used[batch] = False
        return batch


def ae_collate(batch):
    """
    `batch` is what is returned by `AEBatchSampler.__iter__`: a list of tuples
    where each tuple is what returned by `AEDataset.next_item_context`
    """

    # do we need it? if samples have the same size, not
    pass


def transform_func(dbarr: es.array):
    """
    Takes a 2d array in float32 and computes the first 13 MFCC, on each column
    resulting in a new 2darray with 13 columns
    """
    out = []
    for col in range(dbarr.shape[1]):
        amp = utils.db2amp(dbarr[:, col])
        out.append(s.MFCC(amp / (amp.sum() + s.EPS) * amp.size))

    return es.array(out).T


def process_pedaling(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset_utils.get_score_mat(dataset,
                                        i,
                                        score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    diff_spec = transform_func(nmf_tools.initV) - transform_func(
        nmf_tools.renormalize(nmf_tools.W @ nmf_tools.H, initV_sum=True))
    winlen = s.FRAME_SIZE / s.SR
    hop = s.HOP_SIZE / s.SR
    pedaling = dataset.get_pedaling(
        i, frame_based=True, winlen=winlen, hop=hop)[0] / 127
    # padding so that pedaling and diff_spec have the same length
    pedaling, diff_spec = utils.pad(pedaling[:, 1:].T, diff_spec)
    # TODO check the shape here, it should be (frames, features, 1)
    __import__('ipdb').set_trace()
    return diff_spec[None], pedaling[None]


def process_velocities(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset_utils.get_score_mat(dataset,
                                        i,
                                        score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    velocities = dataset_utils.get_score_mat(
        dataset, i, score_type=['precise_alignment'])[:,
                                                      3] / 127  # type: ignore
    minispecs = nmf_tools.get_minispecs(transform=transform_func)
    # now the shape should be (notes, features, frames) for minispecs
    # now the shape should be (notes, ) for velocities
    return minispecs, velocities


def get_loader(groups, redump, contexts, mode=None, nmf_params=None):
    """
    `nmf_params` and `mode` are needed only if `redump` is True
    """
    if redump:
        dumped = False
    else:
        dumped = True

    if mode == 'velocity':
        process_fn = process_velocities
        data_path = s.VELOCITY_DATA_PATH
        batch_size = s.VEL_BATCH_SIZE
    elif mode == 'pedaling':
        process_fn = process_pedaling
        data_path = s.PEDALING_DATA_PATH
        batch_size = s.PED_BATCH_SIZE
    else:
        raise RuntimeError(
            f"mode {mode} not known: available are `velocity` and `pedaling`")

    dataset = AEDataset(
        contexts,
        asmd.Dataset(definitions=[s.RESYNTH_DATA_PATH],
                     metadataset_path=s.METADATASET_PATH), data_path, dumped)

    if not dumped:
        dataset.dump(process_fn, nmf_params, n_jobs=s.NJOBS, max_nbytes=None)

    # select the groups, subsample dataset, and shuffle it
    dataset = dataset.set_operation(dataset_utils.filter, groups=groups)
    dataset = dataset.set_operation(  # type: ignore
        lambda *x, **y: dataset_utils.choice(*x, **y)[0],  # type: ignore
        p=[s.DATASET_LEN, 1 - s.DATASET_LEN],
        random_state=1992)

    dataset[0]
    return DataLoader(dataset,
                      batch_sampler=AEBatchSampler(batch_size, dataset),
                      num_workers=s.NJOBS,
                      pin_memory=True)
    # collate_fn=ae_collate)
