import pickle
import time
from typing import Tuple
from pathlib import Path

import essentia.standard as esst
import numpy as np
import plotly.graph_objects as go
import pretty_midi as pm

from . import settings as s
from .essentiaspec import Spectrometer, peaks_enhance


def make_template(scale_path: Tuple[str, str],
                  spec: Spectrometer,
                  basis: int,
                  basis_frames: Tuple[int, int, int],
                  retuning: bool = False,
                  peaks_enhancing=False) -> Tuple[np.ndarray, int, int]:
    """
    Creates a template.

    In the template, column 0 contains pitch 1 and columns 127 contains pitch
    128

    Arguments
    ---------

    `scale_path` : Tuple[str, str]
        paths to the MIDI and audio file containing the scale
    `spec` : Spectrometer
        The object used for computing the spectrograms
    `basis` : int
        number of basis for internal note times. One more base is used for the
        attack. One more base is used to account for the release after the offset.
    `basis_frames` : Tuple[int, int, int]
        how many frames to use for each basis, namely for the attack, for
        the other basis and, for the release; you can use this argument to
        prevent attack and/or other basis by using 0 frames for each field.
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
    audio = spec.spectrogram(audio, retuning=retuning)
    print(f"Needed time: {time.time() - ttt: .2f}s")
    end_audio = audio.shape[1]

    template = np.zeros((audio.shape[0], 128, basis + 2))
    counter = np.zeros((128, basis + 2))

    maxpitch = 0
    minpitch = 128

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
        if basis_frames[0] > 0:
            end = min(start + basis_frames[0], note_end)
            template[:, pitch, 0] += audio[:, start:end].sum(axis=1)
            counter[pitch, 0] += (end - start)
            start = end

        # other basis except the offset
        if basis_frames[1] > 0:
            should_end = start + (basis - 1) * basis_frames[1]
            end = min(should_end, note_end)
            # padding is needed to account for the case in which a note has not
            # all the basis
            pad_width = ((0, 0), (0, should_end - note_end))
            if pad_width[1][1] > 0:
                note_basis = np.pad(audio[:, start:end],
                                    pad_width,
                                    constant_values=0)
                # number of basis filled completely
                full_basis = (end - start) // basis_frames[1]
                counter[pitch, 1:full_basis + 1] += basis_frames[1]
                half_basis = (end - start) % basis_frames[1]
                counter[pitch, full_basis + 2] += half_basis
            else:
                note_basis = audio[:, start:end]
                counter[pitch, 1:basis] += basis_frames[1]

            template[:, pitch, 1:basis] += note_basis.reshape(
                -1, basis - 1, basis_frames[1]).sum(axis=2)
            start = end

        # offset basis
        if note_end > start:
            template[:, pitch, basis] += audio[:, start:note_end].sum(axis=1)
            counter[pitch, basis] += note_end - start

        # release basis
        if basis_frames[2] > 0:
            end = min(note_end + basis_frames[2], end_audio)
            template[:, pitch, basis + 1] += audio[:, note_end:end].sum(axis=1)
            counter[pitch, basis + 1] += end - note_end

    # normalizing template
    idx = np.nonzero(counter)
    template[:, idx[0], idx[1]] /= counter[idx]

    # collapsing basis and pitch dimension
    template = template.reshape((-1, 128 * (basis + 1)), order='C')

    if peaks_enhancing:
        # normalize to max
        template /= template.max(axis=0) + 1e-15
        # apply peaks_enhancing function
        template = peaks_enhance(template, 2, 0.25, axis=0)

    print(f"Needed time: {time.time() - ttt: .2f}s")

    return template, minpitch, maxpitch


def main():

    template = make_template(
        scale_path=[Path(s.SCALE_DIR) / i for i in s.SCALE_PATH],
        spec=s.SPEC,
        basis=s.BASIS,
        basis_frames=(s.ATTACK, s.BASIS_L, s.RELEASE),
        retuning=s.RETUNING)

    # plot template
    fig = go.Figure(data=go.Heatmap(z=template[0]))
    try:
        import mlflow
        with mlflow.start_run():
            mlflow.log_figure(fig, str(int(time.time())) + '.html')
    except:
        fig.show()

    # saving template
    pickle.dump(template, open(s.TEMPLATE_PATH, 'wb'))


if __name__ == "__main__":
    main()
