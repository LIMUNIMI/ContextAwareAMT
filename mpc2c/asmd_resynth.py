import json
import pathlib
import shutil
import typing as t

import mido
import numpy as np
import pycarla
from scipy.stats import entropy, gennorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .asmd.asmd import asmd


def extract_velocity_features(vel: np.array):
    return *gennorm.fit(vel), entropy(vel)


def extract_pedaling_features(ped: np.array):
    idx_0 = ped == ped.min()
    idx_127 = ped == ped.max()
    ratio_0 = np.count_nonzero(idx_0) / ped.shape[0]
    ratio_127 = np.count_nonzero(idx_127) / ped.shape[0]
    ped = ped[np.where(~idx_0 * ~idx_127)]
    if len(ped) > 0:
        distr = gennorm.fit(ped)
    else:
        distr = (0, 0, 0)
    return ratio_0, ratio_127, *distr, entropy(ped)


def cluster_choice(dataset: asmd.Dataset,
                   n_clusters: int) -> t.List[t.List[int]]:
    # prepare the dataset by reading velocities and pedaling of each song and
    # fitting a gamma distribution
    def proc(i, dataset):
        velocities = dataset.get_score(
            i, score_type=['precise_alignment', 'broad_alignment'])[:, 3]
        vel_data = extract_velocity_features(velocities)

        pedaling = dataset.get_pedaling(i, frame_based=True)[0]
        ped_data1 = extract_pedaling_features(pedaling[:, 1])
        # ped_data2 = extract_pedaling_features(pedaling[:, 2])
        ped_data3 = extract_pedaling_features(pedaling[:, 3])

        # return np.concatenate([vel_data, ped_data1, ped_data2, ped_data3])
        return np.concatenate([vel_data, ped_data1, ped_data3])

    data = dataset.parallel(proc, n_jobs=-1)
    data = np.array(data)

    # PCA
    _old_vars = data.shape[1]
    data = StandardScaler().fit_transform(data)
    pca_computer = PCA(n_components=_old_vars // 2)
    data = pca_computer.fit_transform(data)
    explained_variance = sum(pca_computer.explained_variance_ratio_)
    print(f"Retained {data.shape[1]} variables out of {_old_vars}")
    print(f"Explained variance: {explained_variance:.2f}")

    # outliers
    outlier_detector = IsolationForest(n_estimators=200,
                                       random_state=1992,
                                       bootstrap=True).fit(data)
    outliers = outlier_detector.predict(data)
    _data_no_outlier = data[outliers == 1]
    print(f"Found {data.shape[0] - _data_no_outlier.shape[0]} outliers")

    # clustering
    _data_no_outlier = StandardScaler().fit_transform(_data_no_outlier)
    cluster_computer = KMeans(n_clusters=n_clusters, random_state=1992)
    cluster_computer.fit(_data_no_outlier)
    distances = cluster_computer.transform(data)
    # labels = cluster_computer.predict(data)

    # creating the output structure
    return distribute_clusters(distances)


def distribute_clusters(transformed_data: np.ndarray) -> t.List[t.List[int]]:
    """
    >>> n_samples, n_clusters = transformed_data.shape
    """
    n_samples, n_clusters = transformed_data.shape
    sorted = np.stack(
        [np.argsort(transformed_data[:, i]) for i in range(n_clusters)])
    not_used_samples = np.ones(n_samples, dtype=np.bool)
    clusters = list(range(n_clusters))
    counters = [0] * n_clusters

    seed = 1992
    assigned = 0
    while assigned < n_samples:
        np.random.seed(seed + assigned)
        np.random.shuffle(clusters)
        for cluster in clusters:
            for sample_idx in range(counters[cluster], n_samples):
                sample = sorted[cluster, sample_idx]
                if not_used_samples[sample]:
                    # use that
                    not_used_samples[sample] = False
                    assigned += 1
                    counters[cluster] += 1
                    break
                else:
                    # skip it
                    sorted[cluster, sample_idx] = -1
                    counters[cluster] += 1

    # put remaining indices to -1
    for cluster in clusters:
        sorted[cluster, counters[cluster]:] = -1

    out = [sorted[i, sorted[i] > -1] for i in range(n_clusters)]
    print(sum(len(c) for c in out))
    return out


def group_split(datasets: t.List[str],
                contexts: t.Dict[str, str],
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
    for i, group in enumerate(groups):
        print(f"Splitting group {group}")
        d = dataset.filter(groups=[group], copy=True)
        songs = d.get_songs()

        clusters = cluster_func(d, context_splits[i])
        minlen = min(len(c) for c in clusters)
        print(f"The most little cluster has cardinality {minlen}")
        print([len(c) for c in clusters])
        if minlen < len(contexts):
            raise Exception(
                f"Error trying to split {group}. Try to reduce the number of songs per this split!"
            )
        clusters_groups.append(clusters)

    for clusters in clusters_groups:
        for j, (context, _) in enumerate(contexts):
            if context == 'orig':
                # the original context
                # use all the remaining songs
                cluster = sum(cluster[j:] for cluster in clusters)
                ext = '.wav'
            else:
                # a new context
                cluster = [cluster[j] for cluster in clusters]
                ext = '.flac'

            selected_songs = songs[clusters[cluster]]

            # change attribute 'groups' of each selected song
            for song in selected_songs:
                song['groups'].append(context)
                song['recording']['path'] = [
                    i[:-4] + ext for i in song['recording']['path']
                ]
            # save the dataset in the returned definition
            new_definition['songs'] += selected_songs

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


def split_resynth(datasets: t.List[str], carla_proj: pathlib.Path,
                  output_path: pathlib.Path, metadataset_path: pathlib.Path,
                  context_splits: t.List[int], final_decay: float):
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
    new_def = group_split(datasets, contexts, context_splits, cluster_choice)

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
        if trial(contexts, dataset, output_path, old_install_dir, final_decay):
            break

    print("Copying ground-truth files...")
    for dataset in datasets:
        for old_file in old_install_dir.glob(f"{dataset}/**/*.json.gz"):
            new_file = output_path / old_file.relative_to(old_install_dir)
            print(f":::\n::: {old_file.name} > {new_file.name}")
            shutil.copy(old_file, new_file)
