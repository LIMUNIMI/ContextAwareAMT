import typing as t

import numpy as np
import plotly.express as px
from scipy.stats import entropy, gennorm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .asmd.asmd import asmd, dataset_utils
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
    velocities = dataset_utils.get_score_mat(
        dataset, i, score_type=['precise_alignment', 'broad_alignment'])[:, 3]
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
                            n_jobs=-1,
                            backend='multiprocessing')
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
