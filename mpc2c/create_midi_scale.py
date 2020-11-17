#!/usr/bin/env python3
import pretty_midi as pm


def create_scale(duration, silence, velocity, start=0):
    """
    Returns a list of notes in scale from 21 to 108 included.

    Arguments
    ---------
    duration : float
        the duration of each note in seconds
    silence : float
        the duration of the silence in between two notes
    velocity : int
        the velocity of each note
    start : float
        the start time of the scale in seconds

    Returns
    -------
    list :
        list of pretty_midi.Note
    float :
        the end of the scale in seconds
    """
    scale = []
    for i in range(21, 109):
        end = start + duration
        scale.append(pm.Note(velocity, i, start, end))
        start = end + silence

    return scale, end + silence


notes = []
start = 1
N_VELOCITY_LAYERS = 10
STEP = 127 // N_VELOCITY_LAYERS
COUNTOUR = (127 % N_VELOCITY_LAYERS) // 2
for i, v in enumerate(range(COUNTOUR, 127 - COUNTOUR - 1, STEP), start=1):
    print("Creating velocity layer ", i, ": ", v)
    # for 20 levels of velocity:
    new_scale, end = create_scale(0.1, 0, v, start)
    start = end + 1
    notes += new_scale
    new_scale, end = create_scale(0.1, 1, v, start)
    start = end + 1
    notes += new_scale
    new_scale, end = create_scale(0.1, -0.02, v, start)
    start = end + 1
    notes += new_scale
    # new_scale, end = create_scale(0.5, 0, v, start)
    # start = end + 1
    # notes += new_scale
    # new_scale, end = create_scale(0.5, 1, v, start)
    # start = end + 1
    # notes += new_scale
    # new_scale, end = create_scale(0.5, -0.02, v, start)
    # start = end + 1
    notes += new_scale
    new_scale, end = create_scale(1.5, 0, v, start)
    start = end + 1
    notes += new_scale
    new_scale, end = create_scale(1.5, 1, v, start)
    start = end + 1
    notes += new_scale
    new_scale, end = create_scale(1.5, -0.02, v, start)
    start = end + 1
    notes += new_scale

h = end // 3600
m = int((end % 3600) / 60)
s = (end % 3600) % 60
print("Total duration: ", h, "hh ", m, "mm ", s, "ss ")
print("Total number of notes: ", len(notes))
my_midi = pm.PrettyMIDI(initial_tempo=60)
my_midi.instruments = [pm.Instrument(0)]
my_midi.instruments[0].notes = notes
my_midi.write('to_be_synthesized/scales.mid')
