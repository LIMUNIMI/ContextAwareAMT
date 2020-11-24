import essentia as es
import essentia.standard as esst
import numpy as np
import pretty_midi as pm


class Spectrometer():
    """
    Creates an object to compute spectrograms with given parameters.
    Log-spectrogram is designed for usual music frequencies between 23 and 5000
    Hz. Piano f0 are between 27.5 abd 4186 Hz.

    see ``spectrogram`` function for more details.
    """

    def __call__(self, frame: np.array):
        return self.apply(frame)

    def apply(self, frame: np.array):
        return self.spec(frame)

    def __init__(self, frame_size, sr, binsPerSemitone=3, log=True):
        spectrometer = esst.Spectrum(size=frame_size)
        if log:
            logspec = esst.LogSpectrum(frameSize=frame_size // 2 + 1,
                                       sampleRate=sr,
                                       binsPerSemitone=binsPerSemitone)

            # LogSpectrum also return a tuning estimation...
            self.spec = lambda x: logspec(spectrometer(x))[0]
        else:
            self.spec = spectrometer


def spectrogram(audio, frame_size, hop, sr, log=True, binsPerSemitone=3):
    """
    Computes a spectrogram with given parameters.
    Log-spectrogram is designed for usual music frequencies between 23 and 5000
    Hz. Piano f0 are between 27.5 abd 4186 Hz.

    Example to test:
    >>> import visdom
    >>> import numpy as np
    >>> vis = visdom.Visdom()
    >>> time = np.arange(0, 10, 1/22050)
    >>> audio_5000 = np.sin(2 * np.pi * 5000 * time)
    >>> vis.heatmap(spectrogram(audio_5000, 2048, 512, 22050, True))
    >>> audio_23 = np.sin(2 * np.pi * 23 * time)
    >>> vis.heatmap(spectrogram(audio_23, 2048, 512, 22050, True))
    """

    chromas = []
    spectrometer = Spectrometer(frame_size, sr, binsPerSemitone, log)
    for frame in esst.FrameGenerator(audio,
                                     frameSize=frame_size,
                                     hopSize=hop,
                                     startFromZero=True):
        chromas.append(spectrometer.apply(frame))

    return es.array(chromas).T


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
                   basis_l=1,
                   eps=1e-15,
                   eps_range=0):
    """
    return a pianoroll starting from a mat score from asmd

    if velocities are available, it will be filled with velocity values; to
    turn this off use `velocities=False`

    if `only_onsets` is true, onle the attack is used and the other part of the
    notes are discarded (useful for aligning with amt). Similarly
    `only_offsets`

    `basis` is the number of basis for the nmf; `attack` is the attack
    duration, all other basis will be long `basis_l` column except the last one
    that will last till the end if needed

    `eps_range` defines how to much is note is enlarged before onset and after
    offset in seconds, while `eps` defines the value to use for enlargement
    """

    L = int(np.max(mat[:, 2]) / res) + 1

    pr = np.zeros((128, basis, L))

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
            pr[pitch, basis - 1, end - 1] = vel
            continue

        # the attack basis
        pr[pitch, 0, start:start + attack] = vel

        # the eps_range before onset
        if eps_range > 0:
            start_eps = max(0, start - eps_range)
            pr[pitch, 0, start_eps:start] = eps

        start += attack

        # all the other basis
        END = False
        for b in range(1, basis):
            for k in range(basis_l):
                t = start + (b - 1) * basis_l + k
                if t < end:
                    pr[pitch, b, t] = vel
                else:
                    END = True
                    break
            if END:
                break

        # the ending part
        if not only_onsets:
            if start + (basis - 1) * basis_l < end:
                pr[pitch, basis - 1, start + (basis - 1) * basis_l:end] = vel
                # the eps_range after the offset
                if eps_range > 0:
                    end_eps = min(L, end + eps_range)
                    pr[pitch, basis - 1, end:end_eps] = eps

    # collapse pitch and basis dimension
    pr = pr.reshape((128 * basis, -1), order='C')
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
