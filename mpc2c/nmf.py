import pickle
import random
from copy import copy
from types import SimpleNamespace

import numpy as np

from asmd import asmd

from . import settings as s
from .utils import (find_start_stop, make_pianoroll, spectrogram,
                    stretch_pianoroll)


def NMF(V,
        W,
        H,
        B=10,
        num_iter=10,
        params=None,
        cost_func='Music',
        fixH=False,
        fixW=False):
    """Given a non-negative matrix V, find non-negative templates W and
    activations H that approximate V.

    References
    ----------
    [1] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"

    [2] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
    "Nonnegative Matrix and Tensor Factorizations: Applications to
    Exploratory Multi-Way Data Analysis and Blind Source Separation"
    John Wiley and Sons, 2009.

    [3] D. Jeong, T. Kwon, and J. Nam, “Note-Intensity Estimation of Piano
    Recordings Using Coarsely Aligned MIDI Score,” Journal of the Audio
    Engineering Society, vol. 68, no. 1/2, pp. 34--47, 2020.

    Parameters
    ----------
    V: array-like
        K x M non-negative matrix to be factorized

    cost_func : str
        Cost function used for the optimization, currently supported are:
          'EucDdist' for Euclidean Distance
          'KLDiv' for Kullback Leibler Divergence
          'ISDiv' for Itakura Saito Divergence
          'Music' for score-informed music applications [3]

    num_iter : int
        Number of iterations the algorithm will run.

    W  : np.ndarray
        The initial W modified in place

    H : np.ndarray
        The initial H modified in place

    fixW : bool
        If True, W is not updated

    params : dict
        parameters for `Music` updates with these names:
            a1, a2, a3, b1, b2, Mh, Mw

        `Mh` and `Mw` *MUST* be provided, the others can miss and in that case
        the following are used [3]:
            a1, a2, a3, b1, b2 = 0, 1e3, 1, 1e2, 0

    B : int
        the number of basis for template
    """
    # normalize activations
    W /= W.sum(axis=0)

    # normalize H
    H /= H.sum()

    # normalize to unit sum
    V /= V.sum()

    # get important params
    K, M = V.shape
    L = num_iter
    if cost_func == 'Music':
        # default ones
        if "Mh" not in params or "Mw" not in params:
            raise RuntimeError("Mh and Mw *MUST* be provided")
        params = {
            "a1": params.get("a1") or 10,
            "a2": params.get("a2") or 10,
            "a3": params.get("a3") or 0,
            "b1": params.get("b1") or 0,
            "b2": params.get("b2") or 0,
            "Mh": params["Mh"],
            "Mw": params["Mw"]
        }
        p = SimpleNamespace(**params)

    # create helper matrix of all ones
    onesMatrix = np.ones((K, M))

    # main iterations
    for iter in range(L):

        # compute approximation
        Lambda = s.EPS + W @ H

        # switch between pre-defined update rules
        if cost_func == 'EucDist':
            # euclidean update rules
            if not fixW:
                W *= (V @ H.T / (Lambda @ H.T + s.EPS))

            if not fixH:
                H *= (W.T @ V / (W.T @ Lambda + s.EPS))

        elif cost_func == 'KLDiv':
            # Kullback Leibler divergence update rules
            if not fixW:
                W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + s.EPS)

            if not fixH:
                H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + s.EPS)

        elif cost_func == 'ISDiv':
            # Itakura Saito divergence update rules
            if not fixW:
                W *= ((Lambda**-2 * V) @ H.T) / ((Lambda**-1) @ H.T + s.EPS)

            if not fixH:
                H *= (W.T @ (Lambda**-2 * V)) / (W.T @ (Lambda**-1) + s.EPS)

        elif cost_func == 'Music':
            # update rules for music score-informed applications

            if not fixW:
                W_indicator = np.zeros_like(W)
                W_indicator[:, ::B] += W[:, ::B]
                numW = (V / Lambda) @ H.T
                numW[1:] += 2 * p.b1 * W_indicator[1:]
                numW[:-1] += 2 * p.b1 * W_indicator[:-1] + p.b2 * p.Mw[:-1]

                W *= numW / (onesMatrix @ H.T + s.EPS + p.b2 +
                             4 * p.b1 * W_indicator)

            if not fixH:
                numH = W.T @ (V / Lambda) + p.a1 * p.Mh
                numH[:, B:] += p.a2 * H[:, B:]
                numH[:, :-B] += H[:, :-B]
                H *= numH / (W.T @ onesMatrix + s.EPS + p.a1 + p.a3 +
                             4 * p.a2 * H)

        else:
            raise ValueError('Unknown cost function')

        # normalize templates to unit sum
        W /= W.sum(axis=0)

        # normalize H
        H /= H.sum()


class NMFTools:
    def __init__(self,
                 initW,
                 minpitch,
                 maxpitch,
                 res=0.001,
                 realign=False,
                 sr=s.SR,
                 cost_func=s.NMF_COST_FUNC):
        self.W = initW
        self.minpitch = minpitch
        self.maxpitch = maxpitch
        self.res = res
        self.sr = sr
        self.realign = realign
        self.cost_func = cost_func

    def initialize(self, audio, score):
        self.audio = audio
        self.score = score

        # remove stoping and starting silence in audio
        start, stop = find_start_stop(audio, sample_rate=self.sr)
        audio = audio[start:stop]
        self.V = spectrogram(audio, s.FRAME_SIZE, s.HOP_SIZE, s.SR)

        # compute the needed resolution for pianoroll
        res = len(audio) / self.sr / self.V.shape[1]
        self.pr = make_pianoroll(score,
                                 res=res,
                                 basis=s.BASIS,
                                 velocities=False,
                                 attack=s.ATTACK,
                                 eps=s.EPS_ACTIVATIONS,
                                 eps_range=s.EPS_RANGE)
        # remove trailing zeros in H
        nonzero_cols = self.pr.any(axis=0).nonzero()[0]
        start = nonzero_cols[0]
        stop = nonzero_cols[-1]
        self.pr = self.pr[:, start:stop + 1]

        print(f"Restretching {self.pr.shape[1] - self.V.shape[1]} cols")
        # stretch pianoroll
        self.H = stretch_pianoroll(self.pr, self.V.shape[1])

        # check shapes
        assert self.V.shape == (self.W.shape[0], self.H.shape[1]),\
            "V, W, H shapes are not comparable"
        assert self.H.shape[0] == self.W.shape[1],\
            "W, H have different ranks"

        self.W = self.W[:,
                        self.minpitch * s.BASIS:(self.maxpitch + 1) * s.BASIS]
        self.H = self.H[self.minpitch * s.BASIS:(self.maxpitch + 1) *
                        s.BASIS, :]
        self.pr = copy(self.H)
        self.H[self.H == 0] = s.EPS

    def perform_nmf(self, audio, score):
        self.to2d()
        self.initialize(audio, score)

        # perform nfm
        NMF(self.V,
            self.W,
            self.H,
            B=s.BASIS,
            num_iter=5,
            cost_func=self.cost_func,
            fixH=True,
            fixW=False)

        NMF(self.V,
            self.W,
            self.H,
            B=s.BASIS,
            num_iter=5,
            cost_func=self.cost_func,
            fixH=False,
            fixW=True)

    def realign(self, audio, score):
        raise NotImplementedError("'realign' not implemented yet!")
        # from .alignment.align_with_amt import audio_to_score_alignment
        # score = copy(score)
        # # align score
        # new_ons, new_offs = audio_to_score_alignment(score, audio, self.sr, res=self.res)
        # score[:, 1] = new_ons
        # score[:, 2] = new_offs

    def to3d(self):
        if self.W.ndim != 3:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch, s.BASIS, -1)
                self.pr = self.pr.reshape(npitch, s.BASIS, -1)
            self.W = self.W.reshape((-1, npitch, s.BASIS), order='C')

    def to2d(self):
        if self.W.ndim != 2:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch * s.BASIS, -1)
                self.pr = self.pr.reshape(npitch * s.BASIS, -1)
            self.W = self.W.reshape((-1, npitch * s.BASIS), order='C')

    def get_minispecs(self):

        # use the updated H and W for computing mini-spectrograms
        # and predict velocities
        mini_specs = []
        self.to3d()
        for pitch, onset, offset in self.gen_notes_from_H():
            # select the sorrounding space in H
            # start = max(0, argmax - s.MINI_SPEC_SIZE // 2)
            start = onset
            end = min(start + s.MINI_SPEC_SIZE, self.H.shape[2], offset + 1)

            # compute the mini_spec
            mini_spec = self.W[:, int(pitch - self.minpitch), :] @\
                self.H[pitch, :, start:end]

            # normalizing with rms
            # mini_spec /= (mini_spec**2).mean()**0.5
            # normalizing to the sum
            mini_spec /= mini_spec.sum()

            if mini_spec.shape[1] < s.MINI_SPEC_SIZE:
                mini_spec = np.pad(mini_spec,
                                   pad_width=[
                                       (0,
                                        s.MINI_SPEC_SIZE - mini_spec.shape[1])
                                   ],
                                   mode='constant',
                                   constant_values=s.PADDING_VALUE)

            mini_specs.append(mini_spec)

        return mini_specs

    def gen_notes_from_H(self):
        self.to3d()
        summed_pr = self.pr.sum(axis=1)
        input_onsets = np.argwhere(self.pr[:, 0, :] > 0)
        for note in input_onsets:
            # compute offset
            pitch, onset = note
            offset = -1
            for i in range(note[1], summed_pr.shape[1]):
                if summed_pr[note[0], i] == 0:
                    offset = i - 1
                    break
            if offset == -1:
                offset = summed_pr.shape[1]

            # argmax = np.argmax(self.H[pitch, :, onset:offset + 1]) + onset
            yield pitch, onset, offset


def processing(i, dataset, nmf_params):
    __import__('ipdb').set_trace()
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['non_aligned'])
    velocities = dataset.get_score(i, score_type=['precise_alignment'])[:, 3]
    nmf_tools = NMFTools(*nmf_params)
    nmf_tools.perform_nmf(audio, score)
    nmf_tools.to2d()
    diff_spec = nmf_tools.V - nmf_tools.W @ nmf_tools.H
    winlen = s.FRAME_SIZE / s.SR
    hop = s.HOP_SIZE / s.SR
    pedaling = dataset.get_pedaling(i,
                                    frame_based=True,
                                    winlen=winlen,
                                    hop=hop)[0]
    return (nmf_tools.get_minispecs(), velocities.tolist(), (diff_spec,
                                                             pedaling))


def create_datasets(nmf_params, mini_spec_path: str,
                    diff_spec_path: str) -> None:
    """
    Creates datasets and dumps them to file.

    * ``mini_spec_path`` will contain a list of tuples; each tuple contains the
    mini spectrogram and the corresponding velocities.
    * ``diff_spec_path`` will contain a list a tuples; each tuple contains the
    difference between the original and reconstructed spectrogram and the
    pedaling aligned with the score

    """
    dataset = asmd.Dataset().filter(datasets=s.NMF_DATASETS, groups=['train'])
    random.seed(1750)
    dataset.paths = random.sample(dataset.paths, s.NUM_SONGS_FOR_TRAINING)

    data = dataset.parallel(processing, nmf_params, n_jobs=s.NJOBS)

    mini_specs, diff_specs = [], []
    for d in data:
        specs, vels, diff_spec = d
        diff_specs.append(diff_spec)
        # removing nones
        for i in range(len(specs)):
            spec = specs[i]
            vel = vels[i]
            if spec is not None and vel is not None:
                mini_specs.append((spec, vel))

    pickle.dump(mini_specs, open(mini_spec_path, 'wb'))
    pickle.dump(diff_specs, open(diff_spec_path, 'wb'))
    print(
        f"number of (notes, spectrgrams) in training set: {len(mini_specs)}, {len(diff_specs)}"
    )
