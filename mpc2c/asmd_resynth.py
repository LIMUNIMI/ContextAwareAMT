import json
import shutil
import typing as t
from pathlib import Path

import essentia.standard as esst
import mido
import numpy as np
import plotly.express as px
from scipy.stats import entropy, gennorm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .pycarla import pycarla

from .asmd.asmd import asmd
from .mytorchutils.context import vis


def extract_velocity_features(vel: np.ndarray):
    return *gennorm.fit(vel), entropy(vel)


def extract_pedaling_features(ped: np.ndarray):
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


def parallel_feature_extraction(i, dataset):
    velocities = dataset.get_score(
        i, score_type=['precise_alignment', 'broad_alignment'])[:, 3]
    vel_data = extract_velocity_features(velocities)

    pedaling = dataset.get_pedaling(i, frame_based=True)[0]
    ped_data1 = extract_pedaling_features(pedaling[:, 1])
    ped_data2 = extract_pedaling_features(pedaling[:, 2])
    ped_data3 = extract_pedaling_features(pedaling[:, 3])

    return np.concatenate([vel_data, ped_data1, ped_data2, ped_data3])
    # return np.concatenate([vel_data, ped_data1, ped_data3])


def cluster_choice(dataset: asmd.Dataset,
                   n_clusters: int,
                   target_cardinality: int,
                   plot: bool = True) -> t.List[t.List[int]]:
    # prepare the dataset by reading velocities and pedaling of each song and
    # fitting a gamma distribution

    data = dataset.parallel(parallel_feature_extraction,
                            n_jobs=-1, backend='multiprocessing')
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
    # _data_no_outlier = StandardScaler().fit_transform(_data_no_outlier)
    cluster_computer = KMeans(n_clusters=n_clusters, random_state=1992)
    cluster_computer.fit(_data_no_outlier)

    # creating the output structure
    distances = cluster_computer.transform(data)
    labels = cluster_computer.predict(data)
    mode = 'robinhood'
    distributed_clusters = redistribute(distances,
                                        labels,
                                        mode=mode,
                                        target_cardinality=target_cardinality)

    # plotting
    if plot:
        _plot_clusters(data[:, :2], _data_no_outlier[:, :2], n_clusters,
                       target_cardinality, mode)
    return distributed_clusters


def _plot_clusters(points: np.ndarray, data_to_cluster: np.ndarray,
                   n_clusters: int, target_cardinality: int, title: str):
    cluster_computer = KMeans(n_clusters=n_clusters, random_state=1992)
    cluster_computer.fit(data_to_cluster)
    cluster1 = cluster_computer.predict(points)
    distances = cluster_computer.transform(points)
    cluster2 = np.copy(cluster1)
    _ = redistribute(distances,
                     cluster2,
                     target_cardinality=target_cardinality)

    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        color=[
            str(cl) for cl in AgglomerativeClustering(
                n_clusters=n_clusters, linkage='ward').fit_predict(points)
        ],
        title=f"{title} agglomerative clustering")
    vis.plotlyplot(fig)

    fig = px.scatter(x=points[:, 0],
                     y=points[:, 1],
                     color=[str(cl) for cl in cluster1],
                     category_orders=dict(color=[str(cl) for cl in cluster1]),
                     title=f"{title} color: standard")
    vis.plotlyplot(fig)

    fig = px.scatter(x=points[:, 0],
                     y=points[:, 1],
                     color=[str(cl) for cl in cluster2],
                     category_orders=dict(color=[str(cl) for cl in cluster1]),
                     title=f"{title} color: distributed")
    vis.plotlyplot(fig)


def redistribute(*args, mode='robinhood', **kwargs) -> t.List[t.List[int]]:

    if mode == 'robinhood':
        return robinhood(*args, **kwargs)
    elif mode == 'notpope':
        return notpope(*args, **kwargs)
    else:
        raise RuntimeError("mode not known for redistributing clustering")


def notpope(transformed_data: np.ndarray, labels: np.ndarray,
            **kwargs) -> t.List[t.List[int]]:
    """
    >>> n_samples, n_clusters = transformed_data.shape
    """
    n_samples, n_clusters = transformed_data.shape
    sorted = np.stack(
        [np.argsort(transformed_data[:, i]) for i in range(n_clusters)])
    not_used_samples = np.full(n_samples, -1, dtype=np.int32)
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
                if not_used_samples[sample] < 0:
                    # use that
                    not_used_samples[sample] = cluster
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

    # overwrite labels
    labels[:] = not_used_samples[:]

    out = [sorted[i, sorted[i] > -1].tolist() for i in range(n_clusters)]
    return out


def robinhood(
        transformed_data: np.ndarray,
        labels: np.ndarray,
        *args,
        target_cardinality: t.Optional[int] = None) -> t.List[t.List[int]]:
    """
    >>> n_samples, n_clusters = transformed_data.shape
    """
    n_samples, n_clusters = transformed_data.shape
    if not target_cardinality:
        target_cardinality = n_samples // n_clusters
    cardinalities = np.array(
        [np.count_nonzero(labels == cl) for cl in range(n_clusters)])
    poors = np.where(cardinalities < target_cardinality)[0]
    poors_points = np.where(np.isin(labels, poors))[0]
    # transformed_data = transformed_data[rich_points, :]
    sorted = np.stack(
        [np.argsort(transformed_data[:, i]) for i in range(n_clusters)])
    clusters = [
        i for i in range(n_clusters) if cardinalities[i] < target_cardinality
    ]
    counters = [0] * n_clusters
    not_used = np.ones(n_samples, dtype=np.bool8)
    not_used[poors_points] = False

    seed = 1992
    while np.any(cardinalities < target_cardinality):
        np.random.seed(seed + np.sum(counters))
        np.random.shuffle(clusters)
        for cluster in clusters:
            if cardinalities[cluster] >= target_cardinality:
                # don't add points to rich clusters
                continue
            for sample_idx in range(counters[cluster], n_samples):
                sample = sorted[cluster, sample_idx]
                sample_cluster = labels[sample]
                if cardinalities[
                        sample_cluster] > target_cardinality and not_used[
                            sample_idx]:
                    # the point belong to a rich cluster, steal it
                    labels[sample] = cluster
                    cardinalities[sample_cluster] -= 1
                    cardinalities[cluster] += 1
                    counters[cluster] += 1
                    not_used[sample_idx] = False
                    break
                else:
                    # skip it
                    counters[cluster] += 1

    # creating list of clusters
    out: t.List[t.List[int]] = [[] for i in range(n_clusters)]
    for i, label in enumerate(labels):
        out[label].append(i)
    return out


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

        clusters = cluster_func(
            d, context_splits[i], len(contexts), plot=False)
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


def trial(contexts: t.Mapping[str, t.Optional[Path]],
          dataset: asmd.Dataset,
          output_path: Path,
          old_install_dir: Path,
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
                server = pycarla.JackServer(['-R', '-d', 'alsa'])
                # if this is a new context, start Carla
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
                            print("Carla doesn't exists... restarting everything")
                            carla.restart()
                        synthesize_song(str(midi_path),
                                        str(audio_path),
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
            carla.kill_carla()
            server.kill()
        return False
    else:
        carla.kill_carla()
        server.kill()
        return True


def get_contexts(
        carla_proj: Path) -> t.Dict[str, t.Optional[Path]]:
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
        return False
    midi = dataset.get_score(
        i, score_type=['precise_alignment', 'broad_alignment'])

    # check duration
    if len(audio) / sr >= midi[:, 2].max():
        return False

    # check silence
    processer = esst.InstantPower()
    fs, hs = sr, sr // 2
    pr = dataset.get_pianoroll(i,
                               score_type=[
                                   'precise_alignment', 'broad_alignment'],
                               resolution=1.0,
                               onsets=False,
                               velocity=False)

    for j, frame in enumerate(esst.FrameGenerator(audio,
                                                  frameSize=fs,
                                                  hopSize=hs,
                                                  startFromZero=True)):
        power = processer(frame)

        if j < pr.shape[1]:
            if power == 0:
                if pr[:, (j, j-1)].sum() > 0:
                    print(f"Song {i} was uncorrectly synthesized!!")
                    return False
    return True


def split_resynth(datasets: t.List[str], carla_proj: Path,
                  output_path: Path, metadataset_path: Path,
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
    for i in range(10):
        if trial(contexts, dataset, output_path, old_install_dir, final_decay):
            break

    print("Copying ground-truth files...")
    for dataset in datasets:
        for old_file in old_install_dir.glob(f"{dataset}/**/*.json.gz"):
            new_file = output_path / old_file.relative_to(old_install_dir)
            print(f":::\n::: {old_file.name} > {new_file.name}")
            shutil.copy(old_file, new_file)
