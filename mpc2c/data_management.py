import random

import essentia as es
import numpy as np
from torch.utils.data import DataLoader

from . import nmf
from . import settings as s
from . import utils
from .asmd_resynth import get_contexts
from .asmd.asmd import asmd, dataset_utils
from .mytorchutils import DatasetDump, no_batch_collate


class AEDataset(DatasetDump):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_target(self, i):
        """
        Get two targets: the first one is the value (velocity or pedaling
        level), while the second one is another note/frame with the same value
        but from a different context
        """
        value = self._get_sample(self.included[i], getter_fn=self._get_target)
        # value should have shape returned by `process_velocities` or
        # `process_pedaling`, something like (note/frames,)
        ae_target = []
        for val in value:
            # search the inverted index for another element with the same value
            # as `element`

            # compute the value's bin
            bin = np.searchsorted(self.bins, val)

            # pick a random element in the list
            _i, _j, = random.choice(self.inverted[bin])

            # get that val element using chosen
            ae_target.append(
                self._get_sample(_i, getter_fn=self._get_input)[_j])
        return value, np.asarray(ae_target)


def transform_func(arr: es.array):
    """
    Takes a 2d array in float32 and computes the first 13 MFCC, on each column
    resulting in a new 2darray with 13 columns
    """
    out = []
    for col in range(arr.shape[1]):
        out.append(s.MFCC(arr[:, col]))

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
        dataset, i, score_type=['precise_alignment'])[:, 3] / 127
    minispecs = nmf_tools.get_minispecs(transform=transform_func)
    # now the shape should be (notes, features, frames) for minispecs
    # now the shape should be (notes, ) for velocities
    return minispecs, velocities


def get_loader(groups, redump, mode=None, nmf_params=None):
    """
    `nmf_params` and `mode` are needed only if `redump` is True
    """
    # create a dataset always at the song_level
    # dump the whole dataset
    if redump:
        dumped = False
    else:
        dumped = True

    dataset = AEDataset(
        asmd.Dataset(definitions=[s.RESYNTH_DATA_PATH],
                     metadataset_path=s.METADATASET_PATH), s.DATA_PATH, dumped)

    if not dumped:
        if mode == 'velocity':
            process_fn = process_velocities
        elif mode == 'pedaling':
            process_fn = process_pedaling
        else:
            raise RuntimeError(
                f"mode {mode} not known: available are `velocity` and `pedaling`"
            )

        dataset.dump(process_fn, nmf_params, n_jobs=s.NJOBS, max_nbytes=None)

    # select the groups and subsample dataset
    dataset.apply_func(dataset_utils.filter, groups=groups)
    dataset.apply_func(lambda *x, **y: dataset_utils.choice(*x, **y)[0],
                       p=[s.DATASET_LEN, 1 - s.DATASET_LEN],
                       random_state=1992)

    return DataLoader(dataset,
                      batch_size=1,
                      num_workers=s.NJOBS,
                      pin_memory=True,
                      collate_fn=no_batch_collate)


def multiple_splits_one_context(splits, *args, contexts=None, **kwargs):
    ret = []
    for context in contexts or get_contexts():
        for split in splits:
            ret.append(
                get_loader([split, context] if context is not None else [split],
                           *args, **kwargs))
    if len(ret) == 1:
        ret = ret[0]
    return ret
