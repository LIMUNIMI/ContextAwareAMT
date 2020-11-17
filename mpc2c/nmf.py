import pickle
import random
from copy import copy
from types import SimpleNamespace

import numpy as np
from tqdm import trange

import settings as s
from asmd import audioscoredataset

from .utils import find_start_stop, make_pianoroll, stretch_pianoroll


def NMF(V,
        W,
        H,
        B=10,
        num_iter=10,
        params=None,
        cost_func='Music',
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
    if not fixW:
        normVec = W.sum(axis=0)
        W *= 1.0 / (s.EPS + normVec)

    # get important params
    K, M = V.shape
    L = num_iter
    if cost_func == 'Music':
        # default ones
        if "Mh" not in params or "Mw" not in params:
            raise RuntimeError("Mh and Mw *MUST* be provided")
        params = {
            "a1": params.get("a1") or 0,
            "a2": params.get("a2") or 1e3,
            "a3": params.get("a3") or 1,
            "b1": params.get("b1") or 1e2,
            "b2": params.get("b2") or 0,
            "Mh": params["Mh"],
            "Mw": params["Mw"]
        }
        p = SimpleNamespace(**params)

    # create helper matrix of all ones
    onesMatrix = np.ones((K, M))

    # normalize to unit sum
    V /= (s.EPS + V.sum())

    # main iterations
    for iter in trange(L, desc='NMF:'):

        # compute approximation
        Lambda = s.EPS + W @ H

        # switch between pre-defined update rules
        if cost_func == 'EucDist':
            # euclidean update rules
            if not fixW:
                W *= (V @ H.T / (Lambda @ H.T + s.EPS))

            H *= (W.T @ V / (W.T @ Lambda + s.EPS))

        elif cost_func == 'KLDiv':
            # Kullback Leibler divergence update rules
            if not fixW:
                W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + s.EPS)

            H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + s.EPS)

        elif cost_func == 'ISDiv':
            # Itakura Saito divergence update rules
            if not fixW:
                W *= ((Lambda**-2 * V) @ H.T) / ((Lambda**-1) @ H.T + s.EPS)

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

            numH = W.T @ (V / Lambda) + p.a1 * p.Mh
            numH[:, B:] += p.a2 * H[:, B:]
            numH[:, :-B] += H[:, :-B]
            H *= numH / (W.T @ onesMatrix + s.EPS + p.a1 + p.a3 + 4 * p.a2 * H)

        else:
            raise ValueError('Unknown cost function')

        # normalize templates to unit sum
        if not fixW:
            normVec = W.sum(axis=0)
            W *= 1.0 / (s.EPS + normVec)


def spectrogram(audio, frames=s.FRAME_SIZE, hop=s.HOP_SIZE):

    import essentia.standard as esst
    import essentia as es
    spectrogram = []
    spec = esst.SpectrumCQ(numberBins=s.BINS,
                           sampleRate=s.SR,
                           windowType='hann')
    for frame in esst.FrameGenerator(audio, frameSize=frames, hopSize=hop):
        spectrogram.append(spec(frame))

    return es.array(spectrogram).T


class NMFTools:
    def __init__(self,
                 initW,
                 minpitch,
                 maxpitch,
                 res=0.001,
                 sr=s.SR,
                 realign=False,
                 cost_func='EucDist',
                 eps_activations=1e-4):
        self.W = initW
        self.minpitch = minpitch
        self.maxpitch = maxpitch,
        self.res = res
        self.sr = sr
        self.realign = realign
        self.cost_func = cost_func
        self.eps_activations = eps_activations

    def initialize(self, audio, score):
        self.audio = audio
        self.score = score

        # remove stoping and starting silence in audio
        start, stop = find_start_stop(audio, sample_rate=self.sr)
        audio = audio[start:stop]
        V = spectrogram(audio)

        # compute the needed resolution for pianoroll
        res = len(audio) / self.sr / V.shape[1]
        self.pr = make_pianoroll(score,
                                 res=res,
                                 basis=s.BASIS,
                                 velocities=False,
                                 attack=s.ATTACK,
                                 eps=self.eps_activations,
                                 eps_range=s.EPS_RANGE)

        # remove trailing zeros in H
        nonzero_cols = self.pr.any(axis=0).nonzero()[0]
        start = nonzero_cols[0]
        stop = nonzero_cols[-1]
        self.pr = self.pr[:, start:stop + 1]

        # stretch pianoroll
        self.H = stretch_pianoroll(self.pr, V.shape[1])
        self.pr = copy(self.H)

        # check shapes
        assert V.shape == (self.W.shape[0], self.H.shape[1]),\
            "V, W, H shapes are not comparable"
        assert self.H.shape[0] == self.W.shape[1],\
            "W, H have different ranks"

        self.W = self.W[:,
                        self.minpitch * s.BASIS:(self.maxpitch + 1) * s.BASIS]
        self.H = self.H[self.minpitch * s.BASIS:(self.maxpitch + 1) *
                        s.BASIS, :]
        self.H[self.H == 0] = self.eps_activations

    def perform_nmf(self, audio, score):
        self.initialize(audio, score)

        # perform nfm
        NMF(self.V,
            self.W,
            self.H,
            B=s.BASIS,
            num_iter=5,
            cost_func=self.cost_func)

        NMF(self.V,
            self.W,
            self.H,
            B=s.BASIS,
            num_iter=5,
            cost_func=self.cost_func,
            fixW=True)

    def realign(self, audio, score):
        raise NotImplementedError("'realign' not implemented yet!")
        # from .alignment.align_with_amt import audio_to_score_alignment
        # score = copy(score)
        # # align score
        # new_ons, new_offs = audio_to_score_alignment(score, audio, self.sr, res=self.res)
        # score[:, 1] = new_ons
        # score[:, 2] = new_offs

    def get_minispecs(self):

        # use the updated H and W for computing mini-spectrograms
        # and predict velocities
        mini_specs = []
        npitch = self.maxpitch - self.minpitch + 1
        H = self.H.reshape(npitch, s.BASIS, -1)
        W = self.W.reshape((-1, npitch, s.BASIS), order='C')
        for pitch, onset, offset, argmax in self.gen_notes_from_H(H):
            # select the sorrounding space in H
            start = max(0, argmax - s.MINI_SPEC_SIZE // 2)
            end = min(start + s.MINI_SPEC_SIZE, H.shape[2])

            if end - start < s.MINI_SPEC_SIZE:
                mini_specs.append(None)
                continue

            # compute the mini_spec
            mini_spec = W[:, int(pitch - self.minpitch), :] @\
                H[int(pitch - self.minpitch), :, start:end]

            # normalizing with rms
            # mini_spec /= (mini_spec**2).mean()**0.5
            # normalizing to the sum
            mini_spec /= mini_spec.sum()

            mini_specs.append(mini_spec)

        return mini_specs

    def gen_notes_from_H(self, H):
        summed_pr = self.pr.sum(axis=2)
        input_onsets = np.argwhere(self.pr[:, :, 0] > 0)
        for note in input_onsets:
            # compute offset
            pitch = note[0]
            offset = -1
            for i in range(note[1], summed_pr.shape[1]):
                if summed_pr[i] == 0 or i == summed_pr.shape[1] - 1:
                    offset = i - 1
                    break
            onset = note[1]

            argmax = np.argmax(H[int(note[0] - self.minpitch), :,
                                 onset:offset])[1] + onset
            yield pitch, onset, offset, argmax


def processing(i, dataset, nmf_tools):
    audio, sr = dataset.get_mix(i, sr=s.SR)
    score = dataset.get_score(i, score_type=['non_aligned'])
    velocities = dataset.get_score(i, score_type=['precise_alignment'])[:, 3]
    nmf_tools.perform_nmf(audio, score)
    return (nmf_tools.get_minispecs(), velocities.tolist())


def create_mini_specs(nmf_tools, mini_spec_path):
    """
    Perform alignment and NMF but not velocity estimation; instead, saves all
    the mini_specs of each note in the Maestro dataset for successive training
    """
    from .maestro_split_indices import maestro_splits
    train, validation, test = maestro_splits()
    dataset = audioscoredataset.Dataset().filter(datasets=["Maestro"])
    random.seed(1750)
    train = random.sample(train, s.NUM_SONGS_FOR_TRAINING)
    dataset.paths = np.array(dataset.paths)[train].tolist()

    data = dataset.parallel(processing, nmf_tools, n_jobs=s.NJOBS)

    mini_specs, velocities = [], []
    for d in data:
        specs, vels = d
        # removing nones
        for i in range(len(specs)):
            spec = specs[i]
            vel = vels[i]
            if spec is not None and vel is not None:
                mini_specs.append(spec)
                velocities.append(vel)

    pickle.dump((mini_specs, velocities), open(mini_spec_path, 'wb'))
    print(
        f"number of (inputs, targets) in training set: {len(mini_specs)}, {len(velocities)}"
    )
