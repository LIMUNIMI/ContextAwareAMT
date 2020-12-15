import json
import pathlib
import random
import shutil
from typing import List, Optional

import mido
import numpy as np
import pycarla
from asmd import asmd


def group_split(datasets: List[str],
                contexts: List[str],
                context_splits: List[int],
                groups: List[str] = ['train', 'validation', 'test']) -> dict:
    """
    Given a list of ASMD datasets which support groups, split each group in
    sub groups, each corresponding to a context.

    Returns a new dict object representing an ASMD definition with the new
    groups.
    """

    dataset = asmd.Dataset().filter(datasets=datasets)
    new_definition = {"songs": [], "name": "new_def"}
    for i, group in enumerate(groups):
        d = dataset.filter(groups=[group], copy=True)
        songs = d.get_songs()

        # shuffle songs to randomize the splits
        random.seed(i)
        random.shuffle(songs)

        end: Optional[int]
        start: int = 0
        for context in contexts:
            if context == contexts[-1]:
                # the last context
                end = None
            else:
                # a new context
                # chose a section of size s.CONTEXT_SPLITS[i]
                end = start + context_splits[i]

            selected_songs = songs[start:end]

            # change attribute 'groups' of each selected song
            for song in selected_songs:
                song['groups'].append(context)
            # save the dataset in the returned definition
            new_definition['songs'] += selected_songs

            start = end  # type: ignore
    return new_definition


def synthesize_song(midi_path: str, audio_path: str, final_decay: float = 3):
    """
    Given a path to a midi file, synthesize it and returns the numpy array

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)
    """
    player = pycarla.MIDIPlayer()
    recorder = pycarla.AudioRecorder()
    print("Playing and recording " + midi_path + "...")
    midifile = mido.MidiFile(midi_path)
    print("Total duration: ", midifile.length)
    recorder.start(midifile.length + final_decay)
    player.synthesize_midi_file(midifile, sync=True, progress=True)
    recorder.wait()
    print()
    if not np.any(recorder.recorded != 0):
        raise RuntimeWarning("Recorded file is empty!")
    recorder.save_recorded(audio_path)
    del player, recorder


def split_resynth(datasets: List[str], carla_proj: pathlib.Path,
                  output_path: pathlib.Path, context_splits: List[int],
                  final_decay: float):
    """
    Go trhough the datasets, and using the projects in `carla_proj`, create a
    new resynthesized dataset in `output_path`.

    `context_splits` is a list containing the size for each group of each
    context. Groups should be 3 (train, validation and test).

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)

    After the execution of this function, one can load the new definition file
    by using:

    >>> asmd.Dataset(paths=[output_path], metadataset='metadataset.json')
    """
    server = pycarla.JackServer(['-R', '-d', 'alsa'])
    glob = list(carla_proj.glob("**/*.carxp"))

    # take the name of the contexts
    contexts = [p.stem for p in glob] + ['orig']

    # split the Pathdataset Pathin contexts and save the new definition
    new_def = group_split(datasets, contexts, context_splits=context_splits)

    # create output_path if it doesn't exist and save the new_def
    output_path.mkdir(parents=True, exist_ok=True)
    json.dump(new_def, open(output_path / "new_dataset.json", "wt"))

    # load the new dataset
    dataset = asmd.Dataset(paths=[output_path])
    old_install_dir = pathlib.Path(dataset.install_dir)

    # prepare and save the new metadataset
    dataset.install_dir = str(output_path)
    dataset.metadataset['install_dir'] = str(output_path)
    json.dump(dataset.metadataset, open("metadataset.json", "wt"))
    for i, group in enumerate(contexts):
        # for each context
        # load the preset in Carla
        if group != "orig":
            # if this is a new context, start Carla
            proj = glob[i]
            carla = pycarla.Carla(proj, server, min_wait=4)
            carla.start()

        # get the song with this context
        d = dataset.filter(groups=[group], copy=True)
        for i in range(len(d)):

            # for each song in this context, get the new audio_path
            audio_path = output_path / d.paths[i][0][0]
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path = str(audio_path)
            old_audio_path = str(old_install_dir / d.paths[i][0][0])
            if group != "orig":
                # if this is a new context, resynthesize...
                midi_path = old_audio_path.replace('.wav', '.midi')
                synthesize_song(midi_path, audio_path, final_decay=final_decay)
            else:
                # if this is the original context, copy it!
                # copy the original audio path to the new audio_path
                shutil.copy(old_audio_path, audio_path)
        carla.kill_carla()
        del carla
    server.kill()
