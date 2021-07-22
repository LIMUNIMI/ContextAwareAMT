import json
import os
import shutil
import time
import typing as t
from pathlib import Path
from tqdm import tqdm

import mido
import jack
import numpy as np

from .asmd.asmd import asmd, dataset_utils
from .pycarla import pycarla
from .clustering import cluster_choice

SAVED_ = "asmd_resynth.txt"


def group_split(datasets: t.List[str],
                contexts: t.Dict[str, t.Any],
                context_splits: t.List[int],
                cluster_func: t.Callable,
                groups: t.List[str] = ['train', 'validation', 'test']) -> dict:
    """
    Given a list of ASMD datasets which support groups, split each group in
    sub groups, each corresponding to a context. 'orig' can be used to
    reference to the original context.

    `cluster_func` is a function which takes a dataset and the number of
    clusters and returns a list of clusters; each cluster is a List[int].
    Each group of the dataset is clustered in `len(contexts)` sets.
    One song is taken from each new set so that each group is split in
    contexts.

    N.B. the `orig` context, if used, must come after everything!

    Returns a new dict object representing an ASMD definition with the new
    groups.
    """

    dataset = dataset_utils.filter(asmd.Dataset(), datasets=datasets)
    new_definition = {"songs": [], "name": "new_def"}
    clusters_groups = []
    songs_groups = []
    for i, group in enumerate(groups):
        print(f"Splitting group {group}")
        d = dataset_utils.filter(dataset, groups=[group], copy=True)
        songs = d.get_songs()
        songs_groups.append(songs)

        clusters = cluster_func(d,
                                context_splits[i],
                                len(contexts),
                                plot=False)
        minlen = min(len(c) for c in clusters)
        print("Cardinality of clusters:", [len(c) for c in clusters])
        if minlen < len(contexts):
            raise Exception(
                f"Error trying to split {group}. Try to reduce the number of songs per this split!"
            )
        clusters_groups.append(clusters)

    for i, clusters in enumerate(clusters_groups):
        for j, context in enumerate(contexts):
            cluster: t.List[int]
            if context == 'orig':
                # the original context
                # use all the remaining songs
                cluster = sum([cluster[j:] for cluster in clusters], [])
                ext = '.wav'
            else:
                # a new context
                cluster = [cluster[j] for cluster in clusters]
                ext = '.flac'

            selected_songs: t.List[dict] = [
                songs_groups[i][idx] for idx in cluster
            ]

            # change attribute 'groups' of each selected song
            for song in selected_songs:
                song['groups'].append(context)
                song['recording']['path'] = [
                    i[:-4] + ext for i in song['recording']['path']
                ]
            # save the dataset in the returned definition
            new_definition['songs'] += selected_songs  # type: ignore

    return new_definition


def synthesize_song(midi_path: str, audio_path: str,
                    final_decay: float) -> bool:
    """
    Given a path to a midi file, synthesize it and returns the numpy array

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)

    Return `True` if timeout was reached while recording (probably some frame
    was lost!)
    """
    with pycarla.AudioRecorder() as recorder, pycarla.MIDIPlayer() as player:
        # activating clients need to be done without freewheeling
        print("Playing and recording " + midi_path + "...")
        midifile = mido.MidiFile(midi_path)
        print("Total duration: ", midifile.length)
        recorder.start(midifile.length + final_decay,
                       condition=player.is_ready)
        player.synthesize_midi_file(midifile,
                                    sync=False,
                                    condition=recorder.is_ready)
        player.wait(in_fw=True, out_fw=False)
        timeout = recorder.wait(in_fw=True, out_fw=False)
        print()
        if np.all(recorder.recorded == 0):
            raise RuntimeWarning("Recorded file is empty!")
        recorder.save_recorded(audio_path)
    return timeout


class BackupManager():
    def __init__(self, save_path: str):
        # loading data about already synthesized songs
        self.save_path = save_path
        if os.path.exists(save_path):
            with open(save_path, "rt") as f:
                lines = f.readlines()
                self.backup_i = int(lines[0])
                self.backup_j = int(lines[1])
        else:
            self.backup_i, self.backup_j = -1, -1
            self.write()

    def add_song(self, j: int):
        self.backup_j = j
        self.write()

    def write(self):
        with open(self.save_path, "wt") as f:
            f.writelines(
                [str(self.backup_i) + "\n",
                 str(self.backup_j) + "\n"])

    def add_group(self, i: int):
        self.backup_i = i
        self.backup_j = -1
        self.write()

    def test_song(self, j: int):
        if j < self.backup_j:
            return True
        else:
            return False

    def test_group(self, i: int):
        if i <= self.backup_i:
            return True
        else:
            return False


def trial(contexts: t.Mapping[str, t.Optional[Path]], dataset: asmd.Dataset,
          output_path: Path, old_install_dir: Path,
          final_decay: float) -> bool:
    """
    Try to synthesize the provided contexts from dataset. If success returns
    True, otherwise False.
    """
    backup = BackupManager(SAVED_)
    try:
        for i, (group, proj) in enumerate(contexts.items()):
            if backup.test_group(i):
                print(f"`{group}` was already synthesized")
                continue
            print("\n------------------------------------")
            print("Working on context ", group)
            print("------------------------------------\n")
            # for each context
            # load the preset in Carla
            if group != "orig":
                # if this is a new context, start Carla and jack
                server = pycarla.JackServer([
                    '-d', 'alsa', '-n', '2', '-r', '48000', '-p', '256', '-X',
                    'seq'
                ])
                server.start()
                carla = pycarla.Carla(proj, server, min_wait=8)
                carla.start()

            # get the songs with this context
            d = dataset_utils.filter(dataset, groups=[group], copy=True)
            for j in range(len(d)):
                if backup.test_song(j):
                    print(f"song `{j}` was already synthesized")
                    continue

                # for each song in this context, get the new audio_path
                audio_path = output_path / d.paths[j][0][0]
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    if correctly_synthesized(j, d):
                        backup.add_song(j)
                        print(
                            f"song `{j}` was already synthesized but not in the BackupManager"
                        )
                        continue
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if group != "orig":
                    # if this is a new context, resynthesize...
                    midi_path = (old_install_dir /
                                 d.paths[j][0][0]).with_suffix('.midi')
                    timeout = False
                    while not correctly_synthesized(j, d) or timeout:
                        # delete file if it exists (only python >= 3.8)
                        timeout = resynthesize(audio_path, carla, midi_path,
                                               final_decay)
                else:
                    old_audio_path = str(old_install_dir / d.paths[j][0][0])
                    print(f"Orig context, {old_audio_path} > {audio_path}")
                    shutil.copy(old_audio_path, audio_path)

                # saving this song as synthesized
                backup.add_song(j)
            # saving this group as synthesized
            backup.add_group(i)
            if group != "orig":
                # if this is a new context, close Carla and jack
                carla.kill()
    except Exception as e:
        print("Exception occured while processing group " + group)
        print("    ", e)
        if group != "orig":
            print(
                "There was an error while synthesizing, restarting the procedure"
            )
            carla.kill()
        return False
    else:
        return True


def resynthesize(audio_path, carla, midi_path, final_decay):
    # delete file if it exists (only python >= 3.8)
    audio_path.unlink(missing_ok=True)
    # check that Carla is still alive..
    if not carla.exists():
        print("Carla doesn't exists... restarting everything")
        carla.restart()
    timeout = synthesize_song(str(midi_path), str(audio_path), final_decay)
    time.sleep(2)
    return timeout


def get_contexts(carla_proj: Path) -> t.Dict[str, t.Optional[Path]]:
    """
    Loads contexts and Carla project files from the provided directory

    Returns a dictionary which maps context names to the corresponding carla
    project file.
    """
    glob = list(carla_proj.glob("**/*.carxp"))

    # take the name of the contexts
    contexts: t.Dict[str, t.Optional[Path]] = {}
    for p in glob:
        contexts[p.stem] = p
    return contexts


def correctly_synthesized(i: int, dataset: asmd.Dataset) -> bool:
    """
    Check if audio file in a song in a dataset is correctly loadable and
    synthesized.
    """
    try:
        audio, sr = dataset.get_mix(i)
    except Exception:
        print(f"Song {i} check: couldn't correctly load the recorded song")
        return False

    midi = dataset_utils.get_score_mat(
        dataset, i, score_type=['precise_alignment', 'broad_alignment'])

    # check duration
    if len(audio) / sr < midi[:, 2].max():
        print(f"Song {i} check: audio duration < midi duration!")
        return False

    # check silence
    pr = dataset.get_pianoroll(
        i,
        score_type=['precise_alignment', 'broad_alignment'],
        resolution=1.0,
        onsets=False,
        velocity=False)

    # remove trailing silence in audio
    # note: here silence is exactly 0 because there is no background noise
    start: int = np.argmax(audio != 0)  # type: ignore
    stop: int = np.argmax(audio[::-1] != 0)  # type: ignore
    audio = audio[start:-stop]

    L = min(pr.shape[1], audio.shape[0] // sr)
    silence = audio[:L * sr].reshape((sr, L))
    silence = silence.sum(axis=0) == 0
    notes = pr[:, :L].sum(axis=(0)) > 0

    # if there is silence in audio but not in pianoroll
    # if np.count_nonzero(np.logical_and(silence[:L], notes[:L])) > diff:
    if np.any(np.logical_and(silence[:L], notes[:L])):
        print(f"Song {i} check: uncorrect synthesis!!")
        return False

    return True


def split_resynth(datasets: t.List[str], carla_proj: Path, output_path: Path,
                  metadataset_path: Path, context_splits: t.List[int],
                  final_decay: float):
    """
    Go through the datasets, and using the projects in `carla_proj`, create a
    new resynthesized dataset in `output_path`.

    `context_splits` is a list containing the size for each group of each
    context. Groups should be 3 (train, validation and test).

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)

    After the execution of this function, one can load the new definition file
    by using:

    >>> asmd.Dataset(paths=[output_path], metadataset_path='metadataset.json')
    """
    contexts = get_contexts(carla_proj)

    # split the dataset in contexts and save the new definition
    new_def_fname = output_path / "new_dataset.json"
    if not os.path.exists(new_def_fname):
        new_def = group_split(datasets, contexts, context_splits,
                              cluster_choice)

        # create output_path if it doesn't exist and save the new_def
        output_path.mkdir(parents=True, exist_ok=True)
        json.dump(new_def, open(new_def_fname, "wt"))

    # load the new definition while retaining the old install_dir
    dataset = asmd.Dataset(definitions=[output_path])
    # update the install_dir
    old_install_dir = Path(dataset.install_dir)
    dataset.install_dir = str(output_path)
    dataset.metadataset['install_dir'] = str(output_path)

    # save the new metadataset
    json.dump(dataset.metadataset, open(metadataset_path, "wt"))

    # print("Copying ground-truth files...")
    for dataset_name in datasets:
        for old_file in tqdm(
                old_install_dir.glob(f"{dataset_name}/**/*.json.gz")):
            new_file = output_path / old_file.relative_to(old_install_dir)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(new_file):
                shutil.copy(old_file, new_file)

    print("Synthesizing contexts")
    for i in range(4):
        if not trial(contexts, dataset, output_path, old_install_dir,
                     final_decay):
            time.sleep(4)
        else:
            break
