import pretty_midi as pm
import numpy as np
import pickle
import sys
import plotly.graph_objects as go
from tqdm import trange

SR = 22050
FRAME_SIZE = 16384
HOP_SIZE = 1024
BASIS = 20
BINS = 100
# the number of frames for the attack
ATTACK = 1
# the number of frames for the other basis
BASIS_L = 1
TEMPLATE_PATH = 'nmf_template.pkl'

SCALE_PATH = ['to_be_synthesized/scales.mid', 'audio/pianoteq_scales.mp3']


def main():
    import essentia.standard as esst
    spec = esst.SpectrumCQ(numberBins=BINS, sampleRate=SR, windowType='hann')

    print("Loading midi")
    notes = pm.PrettyMIDI(midi_file=SCALE_PATH[0]).instruments[0].notes
    print("Loading audio")
    audio = esst.EasyLoader(filename=SCALE_PATH[1], sampleRate=SR)()

    # template = np.zeros((FRAME_SIZE // 2 + 1, 128, BASIS))
    template = np.zeros((BINS, 128, BASIS))
    counter = np.zeros((128, BASIS))

    maxpitch = 0
    minpitch = 128

    for i in trange(len(notes)):
        note = notes[i]
        if maxpitch < note.pitch:
            maxpitch = note.pitch
        if minpitch > note.pitch:
            minpitch = note.pitch

        # start and end frame
        start = int(np.round((note.start) * SR))
        end = int(np.round((note.end) * SR))
        ENDED = False

        spd = np.zeros((BINS, BASIS))
        frames = esst.FrameGenerator(audio[start:end],
                                     frameSize=FRAME_SIZE,
                                     hopSize=HOP_SIZE)
        # attack
        for a in range(ATTACK):
            try:
                frame = next(frames)
            except StopIteration:
                print("Error: notes timing not correct")
                print(f"note: {start}, {end}, {len(audio)}")
                sys.exit(99)
            spd[:, 0] += spec(frame)
        counter[note.pitch, 0] += ATTACK

        # other basis except the last one
        for b in range(1, BASIS-1):
            if not ENDED:
                for a in range(BASIS_L):
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
                spd[:, BASIS-1] += spec(frame)
                counter[note.pitch, BASIS-1] += 1
        template[:, note.pitch, :] += spd

    idx = np.nonzero(counter)
    template[:, idx[0], idx[1]] /= counter[idx]

    # collapsing basis and pitch dimension
    template = template.reshape((-1, 128 * BASIS), order='C')

    # plot template
    fig = go.Figure(data=go.Heatmap(z=template))
    fig.show()

    # saving template
    pickle.dump((template, minpitch, maxpitch), open(TEMPLATE_PATH, 'wb'))


if __name__ == "__main__":
    main()
