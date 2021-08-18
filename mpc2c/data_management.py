from typing import List
from itertools import cycle
from random import choice

import numpy as np
import essentia as es  # type: ignore
from torch.utils.data import DataLoader, Sampler  # type: ignore

from . import nmf
from . import settings as s
from . import utils
from .asmd_resynth import get_contexts
from .asmd.asmd import asmd, dataset_utils
from .mytorchutils import DatasetDump


class AEDataset(DatasetDump):
    def __init__(self, contexts: List[str], *args, **kwargs):
        """
        if `generic`, then data are generated for generic independence,
        otherwise for specific indipendence
        """
        super().__init__(*args, **kwargs)
        self.contexts = set(contexts)
        # keep track of which samples among those dumped were used
        self.used = np.zeros(sum(self.lengths), dtype=np.bool8)

    def next_item_context(self, c):
        """
        Returns 4 data: input, target label, target reconstruction (same and different context)

        the input is of context `c`

        raise `StopIteration` when finished
        """
        # computing:
        # 1. dataset with the same context
        same = self.get_context(c)
        # 2.  dataset with different contexts
        different = same.complement()  # type: ignore
        # find the first sample not used in `same` and in `different`
        # both `same`, `different` and this dataset were filtered from the same
        # dumped dataset, so the `original` index is the same

        # looking for the first sample in `same` not used yet (original index)
        input = -1
        for s in same.included:  # type: ignore
            if not self.used[s]:
                self.used[s] = True
                input = s
                break

        if input == -1:
            raise StopIteration

        x = same.get_input(input)
        y = same.get_target(input)

        # take a random sample from `different` with the same label
        diff_target = choice(different.inverted[y])
        # take a random sample from `same` with the same label
        same_target = choice(same.inverted[y])

        return {
            "x": x,
            "y": y,
            "ae_same": same.get_input(*same_target),
            "ae_diff": different.get_input(*diff_target)
        }

    def get_context(self, c: str):
        return self.apply_func(dataset_utils.filter, self, groups=[c])

    def complement(self):
        return self._apply_func(dataset_utils.complement, self)


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
        c = next(self.contexts)
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(self.ae_dataset.next_item_context(c))
            except StopIteration:
                break
        if len(batch) == 0:
            raise StopIteration
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
        list(get_contexts(s.CARLA_PROJ).keys()),
        asmd.Dataset(definitions=[s.RESYNTH_DATA_PATH],
                     metadataset_path=s.METADATASET_PATH), data_path, dumped)

    if not dumped:
        dataset.dump(process_fn, nmf_params, n_jobs=s.NJOBS, max_nbytes=None)

    # select the groups, subsample dataset, and shuffle it
    dataset = dataset.apply_func(dataset_utils.filter, groups=groups)
    dataset = dataset.apply_func( # type: ignore
        lambda *x, **y: dataset_utils.choice(*x, **y)[0],  # type: ignore
        p=[s.DATASET_LEN, 1 - s.DATASET_LEN],
        random_state=1992)

    return DataLoader(dataset,
                      batch_sampler=AEBatchSampler(batch_size, dataset),
                      num_workers=s.NJOBS,
                      pin_memory=True)
    # collate_fn=ae_collate)
