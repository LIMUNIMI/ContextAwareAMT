import pickle
import time
from typing import Tuple, Dict, List
from pathlib import Path

import essentia.standard as esst  # type: ignore
import numpy as np
import plotly.graph_objects as go
import pretty_midi as pm  # type: ignore

from . import settings as s
from .utils import amp2db
from .essentiaspec import Spectrometer, peaks_enhance


def make_template(scale_path: List[str],
                  spec: Spectrometer,
                  basis: Dict[str, int],
                  retuning: bool = False,
                  peaks_enhancing=False) -> Tuple[np.ndarray, int, int]:
    """
    Creates a template.

    In the template, column 0 contains pitch 1 and columns 127 contains pitch
    128

    Arguments
    ---------

    `scale_path` : List[str]
        paths to the MIDI and audio file containing the scale
    `spec` : Spectrometer
        The object used for computing the spectrograms
    `basis` : Dict[str, int]
        data about how many basis to use for each part of
        the note and how many frames to use for each base; keys: 'attack_b',
        'release_b', 'inner_b', 'attack_f', 'release_f', 'inner_f'

        If a note lasts more than `inner` part, the remaining frames are put
        into the last `inner` base.
    `retuning` : bool
        If True, spectrograms are retuned to 440 Hz
    `peaks_enhancing` : bool
        if True, `peaks_enhancing` function is applied after having normalized
        the max value in each column to 1

    Returns
    -------

    np.ndarray :
        The template with shape (bins, 128 * sum(basis_frames))
    """
    sr = spec.sample_rate
    hop_size = spec.hop_size
    attack_b = basis['attack_b']
    attack_f = basis['attack_f']
    inner_b = basis['inner_b']
    inner_f = basis['inner_f']
    release_b = basis['release_b']
    release_f = basis['release_f']

    print("Loading midi")
    notes = pm.PrettyMIDI(midi_file=scale_path[0]).instruments[0].notes
    print("Loading audio")
    ttt = time.time()
    audio = esst.EasyLoader(filename=scale_path[1], sampleRate=sr)()
    print(f"Needed time: {time.time() - ttt: .2f}s")

    # compute the whole spectrogram
    print("Computing spectrogram...")
    retuning = 440 if retuning else 0  # type: ignore
    ttt = time.time()
    audio = spec.spectrogram(audio, retuning=retuning)  # type: ignore
    print("Converting amp to -dbFS")
    audio = amp2db(audio)

    # go.Figure(data=go.Heatmap(z=audio)).show()
    print(f"Needed time: {time.time() - ttt: .2f}s")
    # end_audio = audio.shape[1]

    nbasis = attack_b + inner_b + release_b
    template = np.zeros((audio.shape[0], 128, nbasis))
    counter = np.zeros((128, nbasis))

    maxpitch = 0
    minpitch = 128
    pitch = 0

    def fill_base(first_base, start, end, fpb, nbasis):
        """
        fills a base into the template and the counter, given the starting
        base, the excerpt starting and ending frames, the number of frames per
        base (fpb), and the maximum number of basis

        returns the last frame that has not been processed
        """
        _end = min(start + nbasis * fpb, end)
        for i in range(_end - start):
            # adding one frame at a time
            if start + i > _end:
                break
            base = first_base + i // fpb
            template[:, pitch, base] += audio[:, start + i]
            counter[pitch, base] += 1
        return end

    print("Computing template")
    ttt = time.time()
    for i in range(len(notes)):
        note = notes[i]
        if maxpitch < note.pitch:
            maxpitch = note.pitch
        if minpitch > note.pitch:
            minpitch = note.pitch

        # start and end frame
        pitch = note.pitch - 1
        start = int(note.start / (hop_size / sr))
        note_end = int(note.end / (hop_size / sr))

        # attack
        if attack_b > 0:
            start = fill_base(0, start, note_end, attack_f, attack_b)

        # inner basis
        if inner_b > 0:
            start = fill_base(attack_b, start, note_end, inner_f, inner_b)

        # other basis until the offset
        while note_end > start:
            start = fill_base(attack_b + inner_b - 1, start, note_end + 1, 1,
                              1)

        # release basis
        if release_b > 0:
            end = min(note_end + release_f * release_b, audio.shape[1])
            fill_base(attack_b + inner_b, start, end, release_f,
                      release_b)

    # normalizing template
    idx = np.nonzero(counter)
    template[:, idx[0], idx[1]] /= counter[idx]

    # normalizing each note to sum 1
    # s = template.sum(axis=(0, 2))
    # idx = np.nonzero(s)
    # template[:, idx, :] /= s[None, idx, None]  # None is needed to help broadcasting

    # collapsing basis and pitch dimension
    template = template.reshape((-1, 128 * nbasis), order='C')

    if peaks_enhancing:
        # normalize to max
        template /= template.max(axis=0) + 1e-15
        # apply peaks_enhancing function
        template = peaks_enhance(template, 2, 0.25, axis=0)

    print(f"Needed time: {time.time() - ttt: .2f}s")

    return template, minpitch, maxpitch


def main():

    template = make_template(
        scale_path=[str(Path(s.SCALE_DIR) / i) for i in s.SCALE_PATH],
        spec=s.SPEC,
        basis=s.BASIS_FRAMES,
        retuning=s.RETUNING)

    # plot template
    fig = go.Figure(data=go.Heatmap(z=template[0]))  # type: ignore
    try:
        import mlflow  # type: ignore
        with mlflow.start_run():
            mlflow.log_figure(fig, str(int(time.time())) + '.html')
    except Exception:
        import visdom  # type: ignore
        visdom.Visdom().plotlyplot(fig)

    # saving template
    pickle.dump(template, open(s.TEMPLATE_PATH, 'wb'))


if __name__ == "__main__":
    main()
