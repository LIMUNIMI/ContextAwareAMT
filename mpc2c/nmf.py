from types import SimpleNamespace

import numpy as np

from . import settings as s
from .utils import (find_start_stop, make_pianoroll, pad, spectrogram,
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
    W /= W.sum(axis=0) + s.EPS

    # normalize H
    H /= H.sum() + s.EPS

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


class NMFTools:
    def __init__(self,
                 initW,
                 minpitch,
                 maxpitch,
                 realign=False,
                 sr=s.SR,
                 cost_func=s.NMF_COST_FUNC):
        self.initW = initW[:, minpitch * s.BASIS:(maxpitch + 1) * s.BASIS]
        self.minpitch = minpitch
        self.maxpitch = maxpitch
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

        # normalize to unit sum
        self.V /= self.V.sum()

        # compute the needed resolution for pianoroll
        res = len(audio) / self.sr / self.V.shape[1]
        self.initH = make_pianoroll(score,
                                    res=res,
                                    basis=s.BASIS,
                                    velocities=False,
                                    attack=s.ATTACK,
                                    eps=s.EPS_ACTIVATIONS,
                                    eps_range=s.EPS_RANGE)

        if s.PREPROCESSING == "stretch":
            # remove trailing zeros in H
            nonzero_cols = self.initH.any(axis=0).nonzero()[0]
            start = nonzero_cols[0]
            stop = nonzero_cols[-1]
            self.initH = self.H[:, start:stop + 1]

            # stretch pianoroll
            self.initH = stretch_pianoroll(self.initH, self.V.shape[1])
        elif s.PREPROCESSING == "pad":
            self.V, self.initH = pad(self.V, self.initH)

        self.initH = self.initH[self.minpitch * s.BASIS:(self.maxpitch + 1) *
                                s.BASIS, :]

        # check shapes
        assert self.V.shape == (self.initW.shape[0], self.initH.shape[1]),\
            "V, W, H shapes are not comparable"
        assert self.initH.shape[0] == self.initW.shape[1],\
            "W, H have different ranks"

    def perform_nmf(self, audio, score):
        self.to2d()
        # set initH and V
        self.initialize(audio, score)
        # prepare matrices that will be modified by nmf
        self.H = self.initH.copy()
        self.W = self.initW.copy()

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

    def to3d(self):
        if self.initW.ndim != 3:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch, s.BASIS, -1)
                self.initH = self.initH.reshape(npitch, s.BASIS, -1)
            self.W = self.W.reshape((-1, npitch, s.BASIS), order='C')
            self.initW = self.initW.reshape((-1, npitch, s.BASIS), order='C')

    def to2d(self):
        if self.initW.ndim != 2:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch * s.BASIS, -1)
                self.initH = self.initH.reshape(npitch * s.BASIS, -1)
            self.W = self.W.reshape((-1, npitch * s.BASIS), order='C')
            self.initW = self.initW.reshape((-1, npitch * s.BASIS), order='C')

    def generate_minispecs(self):

        # use the updated H and W for computing mini-spectrograms
        # and predict velocities
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
            mini_spec /= (mini_spec.sum() + s.EPS)

            if mini_spec.shape[1] < s.MINI_SPEC_SIZE:
                mini_spec = np.pad(mini_spec,
                                   pad_width=[
                                       (0, 0),
                                       (0,
                                        s.MINI_SPEC_SIZE - mini_spec.shape[1])
                                   ],
                                   mode='constant',
                                   constant_values=s.PADDING_VALUE)

            yield mini_spec

    def get_minispecs(self):
        mini_specs = []
        for mini_spec in self.generate_minispecs():
            mini_specs.append(mini_spec)
        return np.array(mini_specs)

    def gen_notes_from_H(self):
        self.to3d()
        summed_pr = self.initH.sum(axis=1)
        input_onsets = np.argwhere(self.initH[:, 0, :] > 0)
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
