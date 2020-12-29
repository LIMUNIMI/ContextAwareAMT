import pretty_midi as pm
import numpy as np
import pickle
import sys
import plotly.graph_objects as go
from tqdm import trange
from . import settings as s
from .utils import Spectrometer
import visdom


def main():
    """
    Creates a template.

    Note that pitch 0 is not used and pitch 128 cannot be added if MIDI pitches
    in [1, 128] are used (as in asmd).
    """
    import essentia.standard as esst
    spec = Spectrometer(s.FRAME_SIZE, s.SR)

    print("Loading midi")
    notes = pm.PrettyMIDI(midi_file=s.SCALE_PATH[0]).instruments[0].notes
    print("Loading audio")
    audio = esst.EasyLoader(filename=s.SCALE_PATH[1], sampleRate=s.SR)()

    # template = np.zeros((FRAME_SIZE // 2 + 1, 128, BASIS))
    template = np.zeros((s.BINS, 128, s.BASIS))
    counter = np.zeros((128, s.BASIS))

    maxpitch = 0
    minpitch = 128

    for i in trange(len(notes)):
        note = notes[i]
        if maxpitch < note.pitch:
            maxpitch = note.pitch
        if minpitch > note.pitch:
            minpitch = note.pitch

        # start and end frame
        start = int(np.round((note.start) * s.SR))
        end = int(np.round((note.end) * s.SR))
        ENDED = False

        spd = np.zeros((s.BINS, s.BASIS))
        frames = esst.FrameGenerator(audio[start:end],
                                     frameSize=s.FRAME_SIZE,
                                     hopSize=s.HOP_SIZE)
        # attack
        for a in range(s.ATTACK):
            try:
                frame = next(frames)
            except StopIteration:
                print("Error: notes timing not correct")
                print(f"note: {start}, {end}, {len(audio)}")
                sys.exit(99)
            spd[:, 0] += spec.apply(frame)
        counter[note.pitch, 0] += s.ATTACK

        # other basis except the last one
        for b in range(1, s.BASIS-1):
            if not ENDED:
                for a in range(s.BASIS_L):
                    try:
                        frame = next(frames)
                    except StopIteration:
                        # note is shorter than the number of basis
                        ENDED = True
                        break
                    spd[:, b] += spec(frame)
                    counter[note.pitch, b] += 1

        # last basis
        if not ENDED:
            for frame in frames:
                spd[:, s.BASIS-1] += spec(frame)
                counter[note.pitch, s.BASIS-1] += 1
        template[:, note.pitch, :] += spd

    idx = np.nonzero(counter)
    template[:, idx[0], idx[1]] /= counter[idx]

    # collapsing basis and pitch dimension
    template = template.reshape((-1, 128 * s.BASIS), order='C')

    # plot template
    fig = go.Figure(data=go.Heatmap(z=template))
    try:
        vis = visdom.Visdom()
        vis.plotlyplot(fig)
    except:
        fig.show()

    # saving template
    pickle.dump((template, minpitch, maxpitch), open(s.TEMPLATE_PATH, 'wb'))


if __name__ == "__main__":
    main()
