import essentia.standard as esst  # type: ignore
import numpy as np
import pretty_midi as pm  # type: ignore
from scipy.optimize import linear_sum_assignment
import torch


def write_to_file(fname, string, success_msg, error_msg):
    with open(fname, "w") as file:
        try:
            file.writelines(string)
            print(success_msg)
        except IOError:
            print(error_msg)


def torch_moments(t: list):
    """
    Computes moments on a list of tensors t
    """
    mean = torch.mean(t)
    diffs = t - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    return {"mean": mean, "var": var, "skew": skews, "kurt": kurtoses}


def permute_tensors(t0, t1):
    """
    Permute the rows of tensor `t1` so that the L1/L2 distances between `t0`
    and `t1` is minimized.

    Returns the permutation of columns (the indices)

    see https://math.stackexchange.com/questions/3225410/find-a-permutation-of-the-rows-of-a-matrix-that-minimizes-the-sum-of-squared-err

    For instance, `t0` and `t1` could be tensors describing a linear layer, in
    which case, the permutation of rows must be applied to the columns of the
    next layer
    """

    _, cols = linear_sum_assignment((t0 @ t1.T).detach().cpu().numpy(),
                                    maximize=True)

    # note: cols are the columns of the cost matrix `t0 @ t1.T`, but the rows of t1
    return cols.tolist()


def amp2db(arr, clean=True):
    """
    Convert an array to -dBFS

    If `clean` is False, you need to take care that all values are > 0
    (usually, > 1e-31 or 1e-15), otherwise, `arr` is normalized to sum and clipped
    """
    if clean:
        arr /= 1e4 * arr.sum()
        arr[arr < 1e-15] = 1e-15
    return -20 * np.log10(arr) / np.log10(np.finfo('float64').max)


def db2amp(arr):
    """
    Convert an array to amplitude from -dBFS
    """
    return 10**(-arr * np.log10(np.finfo('float64').max) / 20)


def pad(arr1, arr2):
    """
    Pad 2 2d-arrays so that they have the same length over axis 1.
    Useful for spectrograms, pianorolls, etc.
    Returns the inputs enlarged in same order as the input
    """
    # chose the shortest one
    if arr1.shape[1] < arr2.shape[1]:
        shortest, longest = arr1, arr2
    else:
        shortest, longest = arr2, arr1
    pad = longest.shape[1] - shortest.shape[1]
    shortest = np.pad(shortest, ((0, 0), (0, pad)),
                      mode='constant',
                      constant_values=0)
    if arr1.shape[1] < arr2.shape[1]:
        return shortest, longest
    else:
        return longest, shortest


def midi_pitch_to_f0(midi_pitch):
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch - 69) / 12)


def find_start_stop(audio, sample_rate=44100, seconds=False, threshold=-60):
    """
    Returns a tuple containing the start and the end of sound in an audio
    array.

    ARGUMENTS:
    `audio` : essentia.array
        an essentia array or numpy array containing the audio
    `sample_rate` : int
        sample rate
    `seconds` : boolean
        if True, results will be expressed in seconds (float)
    `threshold` : int
        a threshold as in `essentia.standard.StartStopSilence`

    RETURNS:
    `start` : int or float
        the sample where sound starts or the corresponding second

    `end` : int or float
        the sample where sound ends or the corresponding second
    """
    # reset parameters based on sample_rate
    fs, hs = 1024, 128
    if 44100 % sample_rate == 0 or sample_rate % 44100 == 0:
        # 1025, 22050, 44100, 88200, but not 48000, 96000 etc.
        ratio = sample_rate / 44100
        fs = round(1024 * ratio)
        hs = round(128 * ratio)
    processer = esst.StartStopSilence(threshold=threshold)
    for frame in esst.FrameGenerator(audio,
                                     frameSize=fs,
                                     hopSize=hs,
                                     startFromZero=True):
        start, stop = processer(frame)

    if seconds:
        start = specframe2sec(start, sample_rate, hs, fs)
        stop = specframe2sec(stop, sample_rate, hs, fs)
    else:
        start = int(specframe2sample(start, hs, fs))
        stop = int(specframe2sample(stop, hs, fs))

    if start == 2 * hs:
        start = 0

    return start, stop


def specframe2sec(frame, sample_rate=44100, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return specframe2sample(frame, hop_size, win_len) / sample_rate


def specframe2sample(frame, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return frame * hop_size + win_len / 2


def mat2midipath(mat, path):
    """
    Writes a midi file from a mat like asmd:

    pitch, start (sec), end (sec), velocity

    If `mat` is empty, just do nothing.
    """
    if len(mat) > 0:
        # creating pretty_midi.PrettyMIDI object and inserting notes
        midi = pm.PrettyMIDI()
        midi.instruments = [pm.Instrument(0)]
        for row in mat:
            velocity = int(row[3])
            if velocity < 0:
                velocity = 80
            midi.instruments[0].notes.append(
                pm.Note(velocity, int(row[0]), float(row[1]), float(row[2])))

        # writing to file
        midi.write(path)


def midipath2mat(path):
    """
    Open a midi file  with one instrument track and construct a mat like asmd:

    pitch, start (sec), end (sec), velocity
    """

    out = []
    for instrument in pm.PrettyMIDI(midi_file=path).instruments:
        for note in instrument.notes:
            out.append([note.pitch, note.start, note.end, note.velocity])

    return np.array(out)


def make_pianoroll(mat,
                   basis_frames,
                   res=0.25,
                   velocities=True,
                   only_onsets=False,
                   eps=1e-15,
                   eps_range=0):
    """
    return a pianoroll starting from a mat score from asmd

    if velocities are available, it will be filled with velocity values; to
    turn this off use `velocities=False`

    if `only_onsets` is true, only the attack is used and the other part of the
    notes are discarded (useful for aligning with amt).

    `basis_frames` is a dictionary similar to the following:

        BASIS_FRAMES = {
            #: the number of basis for the attack
            'attack_b': 1,
            #: the number of basis for the release
            'release_b': 15,
            #: the number of basis for the inner
            'inner_b': 14,
            #: the number of frames for the attack basis
            'attack_f': 1,
            #: the number of frames for the release basis
            'release_f': 1,
            #: the number of frames for the inner basis
            'inner_f': 2,
        }

    `eps_range` defines how many columns each note is enlarged before onset and
    after release, while `eps` defines the value to use for enlargement

    Note that pitch 0 is not used and pitch 128 cannot be added if MIDI pitches
    in [1, 128] are used (as in asmd)
    """

    L = int(np.max(mat[:, 2]) / res) + 1

    attack_b, release_b, inner_b =\
        basis_frames['attack_b'], basis_frames['release_b'], basis_frames['inner_b']
    attack_f, release_f, inner_f =\
        basis_frames['attack_f'], basis_frames['release_f'], basis_frames['inner_f']
    nbasis = attack_b + release_b + inner_b

    pr = np.zeros((128, nbasis, L))

    eps_range = int(eps_range / res)

    def fill_base(pitch, start, end, first_base, fpb, nbasis):
        """
        Fill a `nbases` from `first_base` for pitch `pitch` from frame `start`
        to `end` excluded, using `fpb` frames per base.
        If `end` would be before the number of basis specified, the procedure
        is interrupted and the note is filled up to the last frame.
        Returns the first non-filled frame
        """
        _end = min(start + nbasis * fpb, end)
        for b in range(first_base, first_base + nbasis):
            if start + fpb >= _end:
                if start < _end:
                    pr[pitch, b, start:_end] = vel
                # if start >= end, then start == _end (start > _end is impossible)
                break
            else:
                pr[pitch, b, start:start + fpb] = vel
                start += fpb
        # if the for loopended without reaching `break`, then
        # start == start + fpb * nbasis == _end
        return _end

    for i in range(mat.shape[0]):
        note = mat[i]
        pitch = int(note[0])
        vel = int(note[3])
        start = int(np.round(note[1] / res))
        end = min(L - 1, int(np.round(note[2] / res)) + 1)
        if velocities:
            vel = max(1, vel)
        else:
            vel = 1

        # the eps_range before onset
        if eps_range > 0:
            start_eps = max(0, start - eps_range)
            pr[pitch, 0, start_eps:start] = eps

        # the attack basis
        start = fill_base(pitch, start, end, 0, attack_f, attack_b)
        if only_onsets:
            continue

        # the inner basis
        start = fill_base(pitch, start, end, attack_b, inner_f, inner_b)

        # other basis until the offset
        while start < end:
            start = fill_base(pitch, start, end, attack_b + inner_b - 1, 1, 1)

        # the release
        release_end = min(end + release_f * release_b, pr.shape[2])
        fill_base(pitch, end, release_end, attack_b + inner_b, release_f,
                  release_b)

        # the eps range after the offset and after release
        if eps_range > 0:
            end_eps = min(L, end + eps_range)
            pr[pitch, attack_b + inner_b - 1, end:end_eps] = eps
            end_eps = min(L, release_end + eps_range)
            pr[pitch, nbasis - 1, release_end:end_eps] = eps

    # collapse pitch and basis dimension
    pr = pr.reshape((128 * nbasis, -1), order='C')
    return pr


def stretch_pianoroll(pr, out_length):
    """
    Stretch a pianoroll along the second dimension.
    """
    ratio = pr.shape[1] / out_length
    return np.array(
        list(
            map(lambda i: pr[:, min(round(i * ratio), pr.shape[1] - 1)],
                range(out_length)))).T
