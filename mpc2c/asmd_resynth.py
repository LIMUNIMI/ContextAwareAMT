import shutil
import json
import pathlib
import random

import jack_synth
from asmd import asmd


def group_split(datasets: list[str],
                contexts: list[str],
                context_splits: list[int],
                groups: list[str] = ['train', 'validation', 'test']) -> dict:
    """
    Given a list of ASMD datasets which support groups, split each group in
    sub groups each corresponding to a context

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

        start = 0
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
                song['groups'] += context
            # save the dataset somehow!
            new_definition['songs'] += selected_songs
    return new_definition


def synthesize_song(midi_path: str, audio_path: str, final_decay: float = 3):
    """
    Given a path to a midi file, synthesize it and returns the numpy array

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb,,,)
    """
    player = jack_synth.MIDIPlayer()
    recorder = jack_synth.MIDIRecorder()
    print("Playing and recording " + midi_path + "..")
    duration = jack_synth.get_smf_duration(midi_path)
    recorder.start(audio_path, duration + final_decay)
    player.synthesize_midi_file(midi_path)
    player.wait()
    recorder.wait()


def split_resynth(datasets: list[str],
                  carla_proj: pathlib.Path,
                  output_path: pathlib.Path,
                  context_splits: list[int],
                  final_decay: float):
    """
    Go trhough the datasets, and using the projects in `carla_proj`, create a
    new resynthesized dataset in `output_path`.

    `context_splits` is a list containing the size for each group of each
    context. Groups should be 3 (train, validation and test).

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb,,,)

    After the execution of this function, one can load the new definition file
    by using:

    >>> asmd.Dataset(paths=[output_path], metadataset='metadataset.json')
    """
    server = jack_synth.JackServer(['-R', '-d', 'alsa'])
    glob = list(carla_proj.glob("**/*.carxp"))

    # take the name of the contexts
    contexts = [p.stem for p in glob] + ['orig']

    # split the dataset in contexts and save the new definition
    new_def = group_split(datasets, contexts, context_splits=context_splits)
    json.dump(new_def, output_path / "new_dataset.json")

    # load the new dataset
    dataset = asmd.Dataset(paths=[output_path])
    old_install_dir = pathlib.Path(dataset.install_dir)

    # prepare and save the new metadataset
    dataset.install_dir = output_path
    dataset.metadataset['install_dir'] = output_path
    json.dump(dataset.metadataset, "metadataset.json")
    for i, group in enumerate(contexts):
        proj = glob[i]
        carla = jack_synth.Carla(proj, server, min_wait=4)
        carla.start()
        d = dataset.filter(groups=[group], copy=True)
        for i in range(len(d)):
            __import__('ipdb').set_trace()
            audio_path = output_path / d.paths[i][0][0]
            if group != "orig":
                # a new context, resynthesize...
                midi_path = audio_path.replace('.wav', '.midi')
                synthesize_song(midi_path, audio_path, final_decay=final_decay)
            else:
                # the original context, copy it!
                # copy the original audio path to the new audio_path
                old_audio_path = old_install_dir / d.paths[i][0][0]
                shutil.copy(old_audio_path, audio_path)

