import json
import shutil
import time
import typing as t
from pathlib import Path

import mido
import numpy as np
from tqdm import tqdm

from .asmd.asmd import asmd
from .pycarla import pycarla
from .clustering import cluster_choice


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
    For each group, the group of the whole dataset is clustered; then, for each
    context/group combination, one song is taken from each cluster of that
    group to form the context-specific group. This means that each group is
    clustered with `context_splits` clusters.

    N.B. the `orig` context, if used,must come after everything!

    Returns a new dict object representing an ASMD definition with the new
    groups.
    """

    dataset = asmd.Dataset().filter(datasets=datasets)
    new_definition = {"songs": [], "name": "new_def"}
    clusters_groups = []
    songs_groups = []
    for i, group in enumerate(groups):
        print(f"Splitting group {group}")
        d = dataset.filter(groups=[group], copy=True)
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
                    server: pycarla.JackServer, final_decay: float):
    """
    Given a path to a midi file, synthesize it and returns the numpy array

    `final_decay` is the time that is waited before of stopping the recording
    (e.g. if there is a long reverb)
    """
    recorder = pycarla.AudioRecorder(blocksize=server.client.blocksize)
    player = pycarla.MIDIPlayer()
    print("Playing and recording " + midi_path + "...")
    midifile = mido.MidiFile(midi_path)
    print("Total duration: ", midifile.length)
    server.toggle_freewheel()
    recorder.start(midifile.length + final_decay)
    player.synthesize_midi_file(midifile, sync=True, progress=False)
    recorder.wait()
    server.toggle_freewheel()
    print()
    if np.all(recorder.recorded == 0):
        raise RuntimeWarning("Recorded file is empty!")
    recorder.save_recorded(audio_path)
    del player, recorder


def trial(contexts: t.Mapping[str, t.Optional[Path]], dataset: asmd.Dataset,
          output_path: Path, old_install_dir: Path,
          final_decay: float) -> bool:
    """
    Try to synthesize the provided contexts from dataset. If success returns
    True, otherwise False.
    """
    try:
        for group, proj in contexts.items():
            print("\n------------------------------------")
            print("Working on context ", group)
            print("------------------------------------\n")
            # for each context
            # load the preset in Carla
            if group != "orig":
                # if this is a new context, start Carla
                server = pycarla.JackServer([
                    '-R', '-d', 'alsa', '-n', '2', '-r', '48000', '-p', '256',
                    '-X', 'seq'
                ])
                server.start()
                carla = pycarla.Carla(proj, server, min_wait=8)
                carla.start()

            # get the songs with this context
            d = dataset.filter(groups=[group], copy=True)
            for j in range(len(d)):

                # for each song in this context, get the new audio_path
                audio_path = output_path / d.paths[j][0][0]
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    if correctly_synthesized(j, d):
                        print(f"{audio_path} already exists")
                        continue
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if group != "orig":
                    # if this is a new context, resynthesize...
                    midi_path = (old_install_dir /
                                 d.paths[j][0][0]).with_suffix('.midi')
                    while not correctly_synthesized(j, d):
                        # delete file if it exists (only python >= 3.8)
                        audio_path.unlink(missing_ok=True)
                        # check that Carla is still alive..
                        if not carla.exists():
                            print(
                                "Carla doesn't exists... restarting everything"
                            )
                            carla.restart()
                        synthesize_song(str(midi_path), str(audio_path),
                                        server, final_decay)
                else:
                    old_audio_path = str(old_install_dir / d.paths[j][0][0])
                    print(f"Orig context, {old_audio_path} > {audio_path}")
                    shutil.copy(old_audio_path, audio_path)
            if group != "orig":
                # if this is a new context, close Carla
                carla.kill_carla()
                server.kill()
                del carla
    except Exception as e:
        print("Exception occured while processing group " + group)
        print(e)
        if group != "orig":
            print(
                "There was an error while synthesizing, restarting the procedure"
            )
            carla.kill_carla()
            server.kill()
        return False
    else:
        return True


def get_contexts(carla_proj: Path) -> t.Dict[str, t.Optional[Path]]:
    """
    Loads contexts and Carla project files from the provided directory

    Returns a dictionary which maps context names to the corresponding carla
    project file. The additional context 'orig' with project `None` is added.
    """
    glob = list(carla_proj.glob("**/*.carxp"))

    # take the name of the contexts
    contexts: t.Dict[str, t.Optional[Path]] = {}
    for p in glob:
        contexts[p.stem] = p
    contexts['orig'] = None
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

    midi = dataset.get_score(
        i, score_type=['precise_alignment', 'broad_alignment'])

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
    new_def = group_split(datasets, contexts, context_splits, cluster_choice)

    # create output_path if it doesn't exist and save the new_def
    output_path.mkdir(parents=True, exist_ok=True)
    json.dump(new_def, open(output_path / "new_dataset.json", "wt"))

    # load the new dataset
    dataset = asmd.Dataset(paths=[output_path])
    old_install_dir = Path(dataset.install_dir)

    # prepare and save the new metadataset
    dataset.install_dir = str(output_path)
    dataset.metadataset['install_dir'] = str(output_path)
    json.dump(dataset.metadataset, open(metadataset_path, "wt"))

    print("Copying ground-truth files...")
    for dataset_name in datasets:
        for old_file in tqdm(
                old_install_dir.glob(f"{dataset_name}/**/*.json.gz")):
            new_file = output_path / old_file.relative_to(old_install_dir)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(old_file, new_file)

    print("Synthesizing contexts")
    for i in range(10):
        if not trial(contexts, dataset, output_path, old_install_dir,
                     final_decay):
            time.sleep(2)
        else:
            break
