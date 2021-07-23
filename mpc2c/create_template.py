#!/usr/bin/env python3
from pathlib import Path
import pretty_midi as pm
import mido
import numpy as np

from .pycarla import pycarla
from . import make_template
from . import settings as s

MIDI_PATH = Path(s.SCALE_DIR) / s.SCALE_PATH[0]
AUDIO_PATH = Path(s.SCALE_DIR) / s.SCALE_PATH[1]


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


def make_midi():
    # for some reason, it cannot find s without the following line
    from . import settings as s
    notes = []
    start = 1
    width = (s.MAX_VEL - s.MIN_VEL) // s.N_VELOCITY_LAYERS
    for i, v in enumerate(range(s.MIN_VEL + width // 2, s.MAX_VEL, width),
                          start=1):
        print("Creating velocity layer ", i, ": ", v)
        for dur in s.NOTE_DURATION:
            for silence in s.NOTE_SILENCE:
                new_scale, end = create_scale(dur, silence, v, start)
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
    my_midi.write(str(MIDI_PATH))


def synth_scale():
    try:
        carla = pycarla.Carla(
            Path(s.SCALE_DIR) / s.SCALE_PROJ,
            ['-d', 'alsa', '-n', '2', '-r', '48000', '-p', '256', '-X', 'seq'],
            min_wait=4)
        carla.start()
        recorder = pycarla.AudioRecorder()
        player = pycarla.MIDIPlayer()
        print("Playing and recording " + str(MIDI_PATH) + "...")
        midifile = mido.MidiFile(MIDI_PATH)
        recorder.start(midifile.length + 1, condition=player.is_ready)
        player.synthesize_midi_file(midifile,
                                    sync=False,
                                    condition=recorder.is_ready)
        player.wait(in_fw=True, out_fw=True)
        recorder.wait(in_fw=True, out_fw=False)
        if np.all(recorder.recorded == 0):
            raise RuntimeWarning("Recorded file is empty!")
        recorder.save_recorded(AUDIO_PATH)
    except Exception as e:
        print(e)
        carla.kill()


def main():
    import os
    if not os.path.exists(MIDI_PATH):
        make_midi()
    if not os.path.exists(AUDIO_PATH):
        synth_scale()
    make_template.main()


if __name__ == "__main__":
    main()
