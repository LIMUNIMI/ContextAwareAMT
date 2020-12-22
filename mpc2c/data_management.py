import pathlib

from asmd import asmd
from torch.utils.data import DataLoader

from . import nmf
from . import settings as s
from . import utils
from .mytorchutils import DatasetDump, dummy_collate, pad_collate


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


def get_loader(groups, nmf_params, mode):
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
