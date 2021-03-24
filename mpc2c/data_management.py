import pathlib

import essentia as es
from torch.utils.data import DataLoader

from . import nmf
from . import settings as s
from . import utils
from .asmd.asmd import asmd, dataset_utils
from .mytorchutils import (DatasetDump, dummy_collate, no_batch_collate,
                           pad_collate)


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
    score = dataset_utils.get_score_mat(dataset, i, score_type=['precise_alignment'])
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
    return diff_spec[None], pedaling[None]


def process_velocities(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset_utils.get_score_mat(dataset, i, score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    velocities = dataset_utils.get_score_mat(dataset, i, score_type=['precise_alignment'
                                                  ])[:, 3] / 127
    minispecs = nmf_tools.get_minispecs(transform=transform_func)
    return minispecs, velocities


def get_loader(groups, mode, redump, nmf_params=None, song_level=False):
    """
    nmf_params is needed only if `redump` is True
    `song_level` allows to make each bach correspond to one song (e.g. for
    testing at the song-level)
    """
    dataset = dataset_utils.filter(asmd.Dataset(
        paths=[s.RESYNTH_DATA_PATH], metadataset_path=s.METADATASET_PATH),
                                   groups=groups)
    dataset, _ = dataset_utils.choice(dataset,
                                      p=[s.DATASET_LEN, 1 - s.DATASET_LEN],
                                      random_state=1992)
    # dataset.paths = dataset.paths[:int(s.DATASET_LEN * len(dataset.paths))]
    if mode == 'velocity':
        num_samples = [
            len(gt['precise_alignment']['pitches'])
            for i in range(len(dataset.paths)) for gt in dataset.get_gts(i)
        ]
        velocity_dataset = DatasetDump(dataset,
                                       pathlib.Path(s.VELOCITY_DATA_PATH) /
                                       "_".join(groups),
                                       not redump,
                                       song_level=song_level,
                                       num_samples=num_samples)
        # max_nbytes=None disable shared memory for large arrays
        if redump:
            velocity_dataset.dump(process_velocities,
                                  nmf_params,
                                  n_jobs=s.NJOBS,
                                  max_nbytes=None)
        return DataLoader(
            velocity_dataset,
            batch_size=s.VEL_BATCH_SIZE if not song_level else 1,
            num_workers=s.NJOBS,
            pin_memory=True,
            collate_fn=dummy_collate if not song_level else no_batch_collate)
    elif mode == 'pedaling':
        pedaling_dataset = DatasetDump(dataset,
                                       pathlib.Path(s.PEDALING_DATA_PATH) /
                                       "_".join(groups),
                                       not redump,
                                       song_level=song_level,
                                       num_samples=None)
        # max_nbytes=None disable shared memory for large arrays
        if redump:
            pedaling_dataset.dump(process_pedaling,
                                  nmf_params,
                                  n_jobs=s.NJOBS,
                                  max_nbytes=None)
        return DataLoader(
            pedaling_dataset,
            batch_size=s.PED_BATCH_SIZE if not song_level else 1,
            num_workers=s.NJOBS,
            pin_memory=True,
            collate_fn=pad_collate if not song_level else no_batch_collate)


def multiple_splits_one_context(splits, context, *args, **kwargs):
    ret = []
    for split in splits:
        ret.append(
            get_loader([split, context] if context is not None else [split],
                       *args, **kwargs))
    if len(ret) == 1:
        ret = ret[0]
    return ret
