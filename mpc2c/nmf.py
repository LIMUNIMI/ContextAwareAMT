from collections.abc import Iterable

import numpy as np

from . import settings as s
from .utils import find_start_stop, make_pianoroll, pad, stretch_pianoroll, amp2db


def NMF(
        V,
        W,
        H,
        # B=10,
        # params=None,
        num_iter=10,
        cost_func='Music',
        fixH=False,
        fixW=False,
        invertV=False,
        invertW=False):
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
          # 'Music' for score-informed music applications [3]

    num_iter : int
        Number of iterations the algorithm will run.

    W  : np.ndarray
        The initial W modified in place

    H : np.ndarray
        The initial H modified in place

    fixW : bool
        If True, W is not updated

    # params : dict
    #     parameters for `Music` updates with these names:
    #         a1, a2, a3, b1, b2, Mh, Mw

    #     `Mh` and `Mw` *MUST* be provided, the others can miss and in that case
    #     the following are used [3]:
    #         a1, a2, a3, b1, b2 = 0, 1e3, 1, 1e2, 0

    # B : int
    #     the number of basis for template (only needed for the `Music` distance)
    """
    # normalize to unit sum
    V /= V.max()
    if invertV:
        V[...] = 1 - V

    # normalize W
    W /= W.max(axis=0) + s.EPS
    if invertW:
        W[...] = 1 - W

    # normalize H
    H /= H.max()

    # get important params
    K, M = V.shape
    # if cost_func == 'Music':
    #     # default ones
    #     if "Mh" not in params or "Mw" not in params:
    #         raise RuntimeError("Mh and Mw *MUST* be provided")
    #     params = {
    #         "a1": params.get("a1") or 10,
    #         "a2": params.get("a2") or 10,
    #         "a3": params.get("a3") or 0,
    #         "b1": params.get("b1") or 0,
    #         "b2": params.get("b2") or 0,
    #         "Mh": params["Mh"],
    #         "Mw": params["Mw"]
    #     }
    #     p = SimpleNamespace(**params)

    # create helper matrix of all ones
    if cost_func == 'KLDiv':
        onesMatrix = np.ones((K, M))

    # main iterations
    for _ in range(num_iter):

        # compute approximation
        Lambda = W @ H

        # switch between pre-defined update rules
        if cost_func == 'EucDist':
            # euclidean update rules
            if not fixW:
                W *= (V @ H.T / (Lambda @ H.T + s.EPS))

            if not fixH:
                H *= (W.T @ V / (W.T @ Lambda + s.EPS))

        elif cost_func == 'KLDiv':
            # Kullback Leibler divergence update rules
            Lambda += s.EPS
            if not fixW:
                W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + s.EPS)

            if not fixH:
                H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + s.EPS)

        elif cost_func == 'ISDiv':
            # Itakura Saito divergence update rules
            Lambda += s.EPS
            if not fixW:
                W *= ((Lambda**-2 * V) @ H.T) / ((Lambda**-1) @ H.T + s.EPS)

            if not fixH:
                H *= (W.T @ (Lambda**-2 * V)) / (W.T @ (Lambda**-1) + s.EPS)

        # elif cost_func == 'Music':
        #     # update rules for music score-informed applications

        #     if not fixW:
        #         W_indicator = np.zeros_like(W)
        #         W_indicator[:, ::B] += W[:, ::B]
        #         numW = (V / Lambda) @ H.T
        #         numW[1:] += 2 * p.b1 * W_indicator[1:]
        #         numW[:-1] += 2 * p.b1 * W_indicator[:-1] + p.b2 * p.Mw[:-1]

        #         W *= numW / (onesMatrix @ H.T + s.EPS + p.b2 +
        #                      4 * p.b1 * W_indicator)

        #     if not fixH:
        #         numH = W.T @ (V / Lambda) + p.a1 * p.Mh
        #         numH[:, B:] += p.a2 * H[:, B:]
        #         numH[:, :-B] += H[:, :-B]
        #         H *= numH / (W.T @ onesMatrix + s.EPS + p.a1 + p.a3 +
        #                      4 * p.a2 * H)

        else:
            raise ValueError('Unknown cost function')


class NMFTools:
    def __init__(self,
                 initW,
                 minpitch,
                 maxpitch,
                 basis_frames=s.BASIS_FRAMES,
                 spec=s.SPEC,
                 realign=False,
                 cost_func=s.NMF_COST_FUNC):
        self.basis = basis_frames['attack_b'] +\
                                   basis_frames['release_b'] + basis_frames['inner_b']
        self.initW = initW[:, (minpitch - 1) * self.basis +
                           1:maxpitch * self.basis + 1].astype(np.float32)
        self.minpitch = minpitch
        self.maxpitch = maxpitch
        self.sr = spec.sample_rate
        self.spec = spec
        self.realign = realign
        self.cost_func = cost_func

    def initialize(self, audio, score):
        self.audio = audio
        self.score = score

        # remove stoping and starting silence in audio
        # start, stop = find_start_stop(audio, sample_rate=self.sr)
        # audio = audio[start:stop]
        self.initV = self.spec.spectrogram(audio, 440 if s.RETUNING else 0)
        self.initV = amp2db(self.initV)

        # computing resolution of the pianoroll (seconds per column)
        self.res = len(audio) / self.sr / self.initV.shape[1]
        self.initH = make_pianoroll(score,
                                    s.BASIS_FRAMES,
                                    res=self.res,
                                    velocities=False,
                                    only_onsets=False,
                                    eps=s.EPS_ACTIVATIONS,
                                    eps_range=s.EPS_RANGE).astype(np.float32)
        if s.PREPROCESSING == "stretch":
            # remove trailing zeros in H
            # nonzero_cols = self.initH.any(axis=0).nonzero()[0]
            # start = nonzero_cols[0]
            # stop = nonzero_cols[-1]
            # self.initH = self.H[:, start:stop + 1]

            # stretch pianoroll
            self.initH = stretch_pianoroll(self.initH, self.initV.shape[1])
        elif s.PREPROCESSING == "pad":
            self.initV, self.initH = pad(self.initV, self.initH)

        self.initH = self.initH[self.minpitch *
                                self.basis:(self.maxpitch + 1) * self.basis, :]

        # check shapes
        assert self.initV.shape == (self.initW.shape[0], self.initH.shape[1]),\
            "V, W, H shapes are not comparable"
        assert self.initH.shape[0] == self.initW.shape[1],\
            "W, H have different ranks"

    def perform_nmf(self, audio, score):
        self.to2d()
        # set initH and initV
        self.initialize(audio, score)
        # prepare matrices that will be modified by nmf
        self.H = self.initH.copy()
        self.W = self.initW.copy()
        self.V = self.initV.copy()

        # perform nfm
        K = 5
        start = 0
        hop = self.V.shape[1] // K
        end = hop
        for _ in range(K):
            NMF(
                self.V[:, start:end],
                self.W,
                self.H[:, start:end],
                num_iter=1,
                cost_func=self.cost_func,
                fixH=True,
                fixW=False,
                # inverting allows to have most of frames near to 0 instead of 1
                invertV=True,
                invertW=True)
            start = end
            end = min(self.V.shape[1], start + hop)

        NMF(self.V,
            self.W,
            self.H,
            num_iter=5,
            cost_func=self.cost_func,
            fixH=False,
            fixW=False)

    def to3d(self):
        if self.initW.ndim != 3:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch, self.basis, -1)
                self.initH = self.initH.reshape(npitch, self.basis, -1)
            self.W = self.W.reshape((-1, npitch, self.basis), order='C')
            self.initW = self.initW.reshape((-1, npitch, self.basis),
                                            order='C')

    def to2d(self):
        if self.initW.ndim != 2:
            npitch = self.maxpitch - self.minpitch + 1
            if hasattr(self, 'H'):
                self.H = self.H.reshape(npitch * self.basis, -1)
                self.initH = self.initH.reshape(npitch * self.basis, -1)
            self.W = self.W.reshape((-1, npitch * self.basis), order='C')
            self.initW = self.initW.reshape((-1, npitch * self.basis),
                                            order='C')

    def minispecs(self, onsets_from_H=False):
        """
        Arguments
        ---------

        `onsets_from_H` : bool
            see `gen_notes_from_H`
        """

        # use the updated H and W for computing mini-spectrograms
        # and predict velocities
        self.to3d()
        for pitch, onset, offset, velocity in self.gen_notes_from_H(
                onsets_from_H):
            # select the sorrounding space in H
            start = onset
            end = min(
                start + s.SPEC_LEN, self.H.shape[2], offset +
                s.BASIS_FRAMES['release_b'] * s.BASIS_FRAMES['release_f'])

            # compute the mini_spec
            mini_spec = 1 - self.W[:, pitch, :] @ self.H[pitch, :, start:end]

            # padding
            if mini_spec.shape[1] < s.SPEC_LEN:
                mini_spec = np.pad(mini_spec,
                                   pad_width=[
                                       (0, 0),
                                       (0, s.SPEC_LEN - mini_spec.shape[1])
                                   ],
                                   mode='constant',
                                   constant_values=s.PADDING_VALUE)

            yield mini_spec, velocity

    def gen_notes_from_H(self, onsets_from_H=False):
        """
        This functions is a generator which yields 3 values:

        * pitch
        * onset column
        * offset column

        If `onsets_from_H` is True, onsets column are inferred from the initial
        activation matrix (before of the NMF), otherwise they are inferred from
        the initial score. Differences between the two methods can raise if two
        onsets are very near in time (less than the activation matrix
        resolution): in that case, some notes that were in the score may not be
        generated by this function. If instead you use the initial score, you
        will get the correct number of notes, but the elaboration of those
        notes will still not be precise because the NMF tries to regress the
        spectrogram of two notes with only one activation

        Yields
        ------

        `int` : pitch starting from 0 (0 is midi pitch self.minpitch)
        `int` : onset column index in the activation/spectrogram
        `int` : offset column index in the activation/spectrogram
        """
        assert not onsets_from_H, "inferring onsets from H is not implemented anymore, look in previous versions in the git repo"
        self.to3d()
        summed_pr = self.initH.sum(axis=1)
        if onsets_from_H:
            input_onsets = np.argwhere(self.initH[:, 0, :] > 0)
        else:
            input_onsets = self.score
        for note in input_onsets:
            # compute offset
            pitch, onset, offset = note[:3]
            velocity = -255
            if not onsets_from_H:
                onset = int(onset / self.res) + 1
                offset = min(int(offset / self.res), summed_pr.shape[1])
                pitch = int(pitch - self.minpitch)
                velocity = note[3]

            # argmax = np.argmax(self.H[pitch, :, onset:offset + 1]) + onset
            yield pitch, onset, offset, velocity

    def diffspecs(self, win_size, hop_size, pedaling):
        """
        yield two matrices: window from V and from the corresponding
        reconstruction
        """

        # checking shapes and making everything of the same length
        ps = pedaling.shape[0]
        hs = self.H.shape[1]
        if ps != hs:
            # pad works on dimension 1, so I need to add one dummy dimension
            # before of the real one
            pedaling, H = pad(pedaling[None], self.H.copy())
            pedaling = pedaling[0]
            ps = pedaling.shape[0]
            hs = H.shape[1]

        assert ps == hs, f"Pedaling shape is {ps}, activation shape is {hs}"
        assert ps == self.initV.shape[
            1], f"Pedaling shape is {ps}, spectrogram shape is {hs}"

        self.to2d()
        for start in range(0, self.V.shape[1] - win_size, hop_size):
            end = start + win_size
            if np.any(H[:, start:end]):
                V = 1 - self.initV[:, start:end]
                V_hat = self.W @ H[:, start:end]
                # renormalizing V_hat to match the masked V
                V_hat /= V_hat.sum() + s.EPS
                V_hat *= np.sum(V * np.any(self.H[:, start:end], axis=0))
                p = pedaling[start:end]

                assert p.size > 0, "Not enough pedaling data"

                yield V, V_hat, np.mean(p)

    def collect(self, collection_attr, *args, transform=None):
        """
        Arguments
        ---------

        `*args`:
            are passed to collection_attr
        `transform` : Optional[Callable]
            a callable that is applied to each mini spec

        Returns
        -------

        np.array:
            3d array with shape (notes or windows, H, W)
        """
        out = []
        func = getattr(self, collection_attr)
        for el in func(*args):
            if transform is not None:
                el = transform(*el)
            out.append(el)
        if isinstance(out[0], Iterable):
            # el were tuples
            # transposing in pure-python
            out = list(zip(*out))
            return np.array(out[0]), np.array(out[1])
        else:
            # not used anymore...
            # this can happen if you use `onsets_from_H=True`
            return np.array(out)
