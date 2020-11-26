import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from asmd import asmd

from . import nmf
from . import settings as s
from . import utils


class PedalingDataset(TorchDataset, asmd.Dataset):
    def __init__(self, nmf_params, **asmd_filtering_kwargs):
        super().__init__()
        self.filter(**asmd_filtering_kwargs)
        self.nmf_tools = nmf.NMFTools(*nmf_params)

    def __getitem__(self, i):
        audio, sr = self.get_mix(i, sr=s.SR)
        score = self.get_score(i, score_type=['non_aligned'])
        self.nmf_tools.perform_nmf(audio, score)
        self.nmf_tools.to2d()
        diff_spec = self.nmf_tools.V - self.nmf_tools.W @ self.nmf_tools.H
        winlen = s.FRAME_SIZE / s.SR
        hop = s.HOP_SIZE / s.SR
        pedaling = self.get_pedaling(i,
                                     frame_based=True,
                                     winlen=winlen,
                                     hop=hop)[0]
        # padding so that pedaling and diff_spec have the same length
        pedaling, diff_spec = utils.pad(pedaling[:, 1:].T, diff_spec)
        return torch.from_numpy(diff_spec).unsqueeze(0), torch.from_numpy(
            pedaling)


class VelocityDataset(TorchDataset, asmd.Dataset):
    def __init__(self, nmf_params, **asmd_filtering_kwargs):
        super().__init__()
        self.filter(**asmd_filtering_kwargs)
        self.nmf_tools = nmf.NMFTools(*nmf_params)

        # self.num_notes = [
        #     self.get_score(i, score_type=['precise_alignment']).shape[0]
        #     for i in range(len(self.paths))
        # ]
        # self.generator = self.generate_data()

    # def __len__(self):
    #     return sum(self.num_notes)

    def __getitem__(self, i):
        audio, sr = self.get_mix(i, sr=s.SR)
        score = self.get_score(i, score_type=['non_aligned'])
        self.nmf_tools.perform_nmf(audio, score)
        self.nmf_tools.to2d()
        velocities = self.get_score(i, score_type=['precise_alignment'])[:, 3]
        minispecs = self.nmf_tools.get_minispecs()
        return torch.from_numpy(minispecs), torch.from_numpy(velocities)#, torch.tensor(False)
        # return next(self.generator)

    def generate_data(self):
        for i in range(len(self.paths)):
            audio, sr = self.get_mix(i, sr=s.SR)
            score = self.get_score(i, score_type=['non_aligned'])
            self.nmf_tools.perform_nmf(audio, score)
            self.nmf_tools.to2d()
            velocities = self.get_score(i, score_type=['precise_alignment'])[:,
                                                                             3]
            # for j, minispec in enumerate(self.nmf_tools.generate_minispecs()):
            #     yield torch.from_numpy(minispec).unsqueeze(0), torch.tensor(
            #         velocities[j])


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
    if mode == 'velocity':
        return DataLoader(
            VelocityDataset(nmf_params, datasets=s.DATASETS, groups=groups),
            batch_size=1,
            # batch_size=s.BATCH_SIZE,
            num_workers=s.NJOBS,
            pin_memory=True,
            collate_fn=reshape_collate)
    elif mode == 'pedaling':
        return DataLoader(PedalingDataset(nmf_params,
                                          datasets=s.DATASETS,
                                          groups=groups),
                          batch_size=s.BATCH_SIZE,
                          num_workers=s.NJOBS,
                          pin_memory=True,
                          collate_fn=pad_collate)
