import pathlib

import essentia as es
from asmd import asmd
from torch.utils.data import DataLoader

from . import nmf
from . import settings as s
from . import utils
from .mytorchutils import DatasetDump, dummy_collate, pad_collate


def transform_func(arr: es.array):
    """
    Takes a 2d array in float32 and computes the first 13 MFCC, on each column,
    and discards the first one, resulting in a new 2darray with 12 columns
    """
    out = []
    for col in range(arr.shape[1]):
        out.append(s.MFCC(arr[:, col])[1:])

    return es.array(out).T


def process_pedaling(i, dataset, nmf_params):
    nmf_tools = nmf.NMFTools(*nmf_params)
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    diff_spec = transform_func(nmf_tools.initV) - transform_func(
        nmf_tools.renormalize(nmf_tools.W @ nmf_tools.H))
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
    score = dataset.get_score(i, score_type=['precise_alignment'])
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    velocities = dataset.get_score(i, score_type=['precise_alignment'
                                                  ])[:, 3] / 127
    minispecs = nmf_tools.get_minispecs(transform=transform_func)
    return minispecs, velocities


def get_loader(groups, nmf_params, mode, redump):
    dataset = asmd.Dataset(
        paths=[s.RESYNTH_DATA_PATH],
        metadataset_path=s.METADATASET_PATH).filter(groups=groups)
    dataset.paths = dataset.paths[:int(s.DATASET_LEN * len(dataset.paths))]
    if mode == 'velocity':
        num_samples = [
            len(gt['precise_alignment']['pitches'])
            for i in range(len(dataset.paths)) for gt in dataset.get_gts(i)
        ]
        velocity_dataset = DatasetDump(dataset,
                                       pathlib.Path(s.VELOCITY_DATA_PATH) /
                                       "_".join(groups),
                                       not redump,
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
                                       not redump,
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
