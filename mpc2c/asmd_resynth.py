import json
import pathlib
import random
import shutil
from typing import List, Optional, Dict

import mido
import numpy as np

import pycarla


def group_split(datasets: List[str],
                contexts: Dict[str, str],
                context_splits: List[int],
                groups: List[str] = ['train', 'validation', 'test']) -> dict:
    """
    Given a list of ASMD datasets which support groups, split each group in
    sub groups, each corresponding to a context. 'orig' can be used to
    reference to the original context.

    Returns a new dict object representing an ASMD definition with the new
    groups.
    """

    from .asmd.asmd import asmd
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
        for context, _ in contexts:
            if context == 'orig':
                # the original context
                end = None
                ext = '.wav'
            else:
                # a new context
                # chose a section of size s.CONTEXT_SPLITS[i]
                end = start + context_splits[i]
                ext = '.flac'

            selected_songs = songs[start:end]

            # change attribute 'groups' of each selected song
            for song in selected_songs:
                song['groups'].append(context)
                song['recording']['path'] = [
                    i[:-4] + ext for i in song['recording']['path']
                ]
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


def trial(contexts, dataset, output_path, old_install_dir, final_decay):
    try:
        for group, proj in contexts:
            print("\n------------------------------------")
            print("Working on context ", group)
            print("------------------------------------\n")
            # for each context
            # load the preset in Carla
            if group != "orig":
                server = pycarla.JackServer(['-R', '-d', 'alsa'])
                # if this is a new context, start Carla
                carla = pycarla.Carla(proj, server, min_wait=8)
                carla.start()

            # get the song with this context
            d = dataset.filter(groups=[group], copy=True)
            for j in range(len(d)):

                # for each song in this context, get the new audio_path
                audio_path = output_path / d.paths[j][0][0]
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    print(f"{audio_path} already exists")
                    continue
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path = str(audio_path)
                if group != "orig":
                    old_audio_path = str(
                        old_install_dir / d.paths[j][0][0])[:-5] + '.wav'
                    # if this is a new context, resynthesize...
                    midi_path = old_audio_path[:-4] + '.midi'
                    # check that Carla is still alive..
                    if not carla.exists():
                        print("Carla doesn't exists... restarting everything")
                        carla.restart()
                    synthesize_song(midi_path,
                                    audio_path,
                                    final_decay=final_decay)
                else:
                    old_audio_path = str(old_install_dir / d.paths[j][0][0])
                    print(f"Orig context, {old_audio_path} > {audio_path}")
                    shutil.copy(old_audio_path, audio_path)
            if group != "orig":
                # if this is a new context, close Carla
                carla.kill_carla()
                del carla
    except Exception as e:
        print("Exception occured while processing group " + group)
        print(e)
        if group != "orig":
            print(
                "There was an error while synthesizing, restarting the procedure"
            )
            carla.restart()
        return False
    else:
        server.kill()
        return True


def get_contexts(carla_proj: pathlib.Path):
    glob = list(carla_proj.glob("**/*.carxp"))

    # take the name of the contexts
    contexts = {}
    for p in glob:
        contexts[p.stem] = p
    contexts['orig'] = None
    return contexts


def split_resynth(datasets: List[str], carla_proj: pathlib.Path,
                  output_path: pathlib.Path, metadataset_path: pathlib.Path,
                  context_splits: List[int], final_decay: float):
    """
    Go trhough the datasets, and using the projects in `carla_proj`, create a
    new resynthesized dataset in `output_path`.

    `context_splits` is a list containing the size for each group of each
    context. Groups should be 3 (train, validation and test).

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)

    After the execution of this function, one can load the new definition file
    by using:

    >>> asmd.Dataset(paths=[output_path], metadataset_path='metadataset.json')
    """
    from .asmd.asmd import asmd
    contexts = get_contexts(carla_proj)

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
    json.dump(dataset.metadataset, open(metadataset_path, "wt"))
    for i in range(100):
        if trial(contexts, dataset, output_path, old_install_dir,
                 final_decay):
            break

    print("Copying ground-truth files...")
    for dataset in datasets:
        for old_file in old_install_dir.glob(f"{dataset}/**/*.json.gz"):
            new_file = output_path / old_file.relative_to(old_install_dir)
            print(f":::\n::: {old_file.name} > {new_file.name}")
            shutil.copy(old_file, new_file)
