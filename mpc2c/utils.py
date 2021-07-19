import essentia.standard as esst
import numpy as np
import pretty_midi as pm


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
                   res=0.25,
                   velocities=True,
                   only_onsets=False,
                   only_offsets=False,
                   basis=1,
                   attack=1,
                   release=1,
                   basis_l=1,
                   eps=1e-15,
                   eps_range=0):
    """
    return a pianoroll starting from a mat score from asmd

    if velocities are available, it will be filled with velocity values; to
    turn this off use `velocities=False`

    if `only_onsets` is true, only the attack is used and the other part of the
    notes are discarded (useful for aligning with amt). Similarly
    `only_offsets`; however, `only_offsets` doesn't take into account the
    release (which comes after the onset).

    `basis` is the number of basis for the nmf, representing the internal note
    state. Additional basis are `attack` and `release`. `attack` is the attack
    duration, `release` is the release duration (after the note offset); all
    other basis will be long `basis_l` columns except the last basis before the
    offset that will last till the end if needed; after that, the release basis
    is added.

    `eps_range` defines how many columns each note is enlarged before onset and
    after release, while `eps` defines the value to use for enlargement

    Note that pitch 0 is not used and pitch 128 cannot be added if MIDI pitches
    in [1, 128] are used (as in asmd)
    """

    L = int(np.max(mat[:, 2]) / res) + 1

    pr = np.zeros((128, basis + 2, L))

    eps_range = int(eps_range / res)

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

        if only_offsets:
            pr[pitch, basis, end - 1] = vel
            continue

        # the attack basis
        pr[pitch, 0, start:start + attack] = vel
        if only_onsets:
            continue

        # the eps_range before onset
        if eps_range > 0:
            start_eps = max(0, start - eps_range)
            pr[pitch, 0, start_eps:start] = eps

        start += attack

        # all the other basis
        END = False
        offset = start
        for b in range(basis - 1):
            for k in range(basis_l):
                offset = start + b * basis_l + k
                if offset < end:
                    pr[pitch, b + 1, offset] = vel
                else:
                    END = True
                    break
            if END:
                break

        # the offset part
        if offset < end:
            pr[pitch, basis, offset:end] = vel

        # the release
        if release > 0:
            end_release = min(L, end + release)
            pr[pitch, basis + 1, end:end_release] = vel
        # the eps range after the offset and after release
        if eps_range > 0:
            end_eps = min(L, end + eps_range)
            pr[pitch, basis, end:end_eps] = eps
            end_eps = min(L, end_release + eps_range)
            pr[pitch, basis + 1, end_release:end_eps] = eps

    # collapse pitch and basis dimension
    pr = pr.reshape((128 * (basis + 2), -1), order='C')
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
