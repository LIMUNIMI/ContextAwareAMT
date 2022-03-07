import pathlib
import pickle
from copy import copy, deepcopy
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset as TorchDataset

# import heartrate
# heartrate.trace(port=8080)


class DatasetDump(TorchDataset):
    """
    This class is useful to save preprocessed datasets.

    It takes a dataset (an `asmd.Dataset` object) and a path where reprocessed
    dataset samples will be saved.  You should then call the method 'dump'
    which preprocess the dataset and dumps it to files.  Finally, you should
    iterate over this object (it's an Iterable).

    This allows for fast dumping in loop of complex objects (such as
    collections of samples where the cardinality of each collections is
    variable and each sample has a huge number of features).

    TODO: add ability to dump samples on the first run.

    """
    def __init__(self,
                 dataset,
                 root: str,
                 dumped: bool = False,
                 bins=np.arange(0, 1, 0.01)):
        super().__init__()
        self.dataset = dataset
        self.root = pathlib.Path(root)

        self.dumped = dumped
        if dumped:
            # inverted is maps the target to the input indices according to the
            # unfiltered dataset
            self.inverted = pickle.load(open(self.root / 'inverted.pkl', 'rb'))
            self.original_inverted = self.inverted
            self.lengths = pickle.load(open(self.root / 'lengths.pkl', 'rb'))
            self.bins = pickle.load(open(self.root / 'bins.pkl', 'rb'))
        else:
            self.bins = bins

        # the list of the included songs
        self.songs = self.dataset.get_songs()

        self.included = np.ones(len(self.dataset), dtype=np.bool8)

    def dump(self, process_fn, *args, **kwargs):
        """
        Preprocess data and dumps them to file.

        Arguments
        ---------

        `process_fn` : Callable
            a function which is used to preprocess each song in the dataset;
            this should respect the asmd.Dataset.parallel structure: its first
            argument should be an int (the index of the song), and the second
            argument should be the dataset itself

        `*args` : any
            additional arguments for `process_fn`

        `**kwargs` : any
            additional key-word arguments for `process_fn` and joblib
        """
        print(f"Dumping data to {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

        self.dataset.dumped_samples = 0
        def pickle_fn(i, dataset, get_data_fn, *args, **kwargs):
            d = {}
            xx, yy = get_data_fn(i, dataset, *args, **kwargs)
            for j in range(len(xx)):
                x, y = xx[j], yy[j]
                bin = self.get_bin(y)
                if bin in d:
                    d[bin].append((i, j))
                else:
                    d[bin] = [(i, j)]
                dest_path = self.make_sample_path(i, j)
                path_x = dest_path + "x.npz"
                path_y = dest_path + "y.npz"
                np.savez(path_x, x)
                np.savez(path_y, y)
            return d, len(xx)

        # the following returns one dict for each song
        dicts = self.dataset.parallel(pickle_fn, process_fn, *args, **kwargs)

        # merging dictionaries
        self.lengths = []
        self.inverted = {}
        for d, l in dicts:
            for bin in d:
                if bin in self.inverted:
                    self.inverted[bin] += d[bin]
                else:
                    self.inverted[bin] = d[bin]
            self.lengths.append(l)
        for k, v in self.inverted.items():
            self.inverted[k] = np.asarray(v)
        self.lengths = np.asarray(self.lengths)
        pickle.dump(self.inverted, open(self.root / 'inverted.pkl', 'wb'))
        pickle.dump(self.lengths, open(self.root / 'lengths.pkl', 'wb'))
        pickle.dump(self.bins, open(self.root / 'bins.pkl', 'wb'))
        self.original_inverted = self.inverted
        self.dumped = True

    def get_bin(self, y):
        return np.searchsorted(self.bins, y)

    def set_operation(self, func, *args, **kwargs):
        """
        Apply `func` to this object's `dataset` to do set operations, using the
        given `args` and `kwargs`. Returns a new dataset, leaving this instance unchanged
        """
        # here copy is needed to avoid copying the whole `inverted` and
        # `lengths` objects (13 seconds about)
        ret = copy(self)
        ret.dataset = func(deepcopy(ret.dataset), *args, **kwargs)
        # parse all datasets and songs and remember the "included" ones
        ret.included = np.zeros_like(ret.included)
        i = 0
        for d in ret.dataset.datasets:
            for s in d['songs']:
                if s['included']:
                    ret.included[i] = True
                i += 1

        ret.inverted = {}
        for bin, v in ret.original_inverted.items():
            keep = ret.included[v[:, 0]]
            ret.inverted[bin] = v[keep, :] #.tolist()

        # the above does the same as following, but the following takes one whole second
        # while the above 0.1 seconds
        # ret.inverted = {
        #     bin: [(i, j) for i, j in ret.original_inverted[bin] if ret.included[i]]
        #     for bin in ret.original_inverted.keys()
        # }
        # print(f"needed time: {time.time() - ttt:.3f}")

        ret.songs = ret.dataset.get_songs()

        return ret

    def _get_song_indices(self, index: int, filtered: bool):
        """
        Returns the song index and the song sample index relative to the unfiltered dataset

        Usable only if dataset was already dumped
        """
        # compute the song containing sample number `index` given the "included" songs
        k = 0
        song = -1
        if filtered:
            included = np.nonzero(self.included)[0]
        else:
            included = range(self.lengths.shape[0]) # type: ignore

        for s in included:
            if k + self.lengths[s] > index:
                song = s
                break
            else:
                k += self.lengths[s]

        # computing the original index of sample `index`
        song_idx = index - k  # the index of the sample in the song

        return song, song_idx

    def make_sample_path(self, song: int, song_idx: int):
        """
        builds the path for a sample being in song `song` and having index `song_idx` within the song
        """
        dir = self.root / str(song)
        dir.mkdir(parents=True, exist_ok=True)
        out = str(dir / str(song_idx))
        return out

    def get_folder(self, index: int, filtered: bool = False):
        """
        Returns the `Path` of a folder where a certain index has been saved.

        This is useful for managing large indices..

        This function takes into account possible filtering of the dataset
        """
        # get the path containing the `original` index
        song, song_idx = self._get_song_indices(index, filtered)
        return self.make_sample_path(song, song_idx)

    def get_target(self, i, song_idx=None, filtered=True):
        """
        get target `i`.

        If `song_idx` is not None, then `i` should be the index of the song
        relative to the dumped dataset and `song_index` the sample index
        referred to the song. If `song_idx` is None, then `i` should be the
        index of the sample referred to the dataset.

        In the latter case, if `filtered` is True, `i` must be referred to the
        filtered dataset, otherwise to the dumped dataset (not filtered in any
        way).

        `filtered` has no effect if `song_idx` is not None.
        """
        if song_idx:
            y = np.load(self.make_sample_path(i, song_idx) + "y.npz")['arr_0']
        else:
            y = np.load(self.get_folder(i, filtered=filtered) + "y.npz")['arr_0']
        y = torch.from_numpy(y)
        return y

    def get_input(self, i, song_idx=None, filtered=True):
        """
        get input, see `get_target`
        """
        if song_idx:
            path = self.make_sample_path(i, song_idx)
        else:
            path = self.get_folder(i, filtered=filtered)
        x = torch.from_numpy(np.load(path + "x.npz")['arr_0'])
        return x, path

    def __getitem__(self, i):
        assert self.dumped, "Dataset not dumped!"
        x = self.get_input(i)
        y = self.get_target(i)
        return x, y

    def __len__(self):
        return self.lengths[self.included].sum() # type: ignore

    def itertargets(self):
        for i in range(len(self)):
            yield self.get_target(i)

    def iterinputs(self):
        for i in range(len(self)):
            yield self.get_input(i)


def pad_collate(batch):
    xx, yy = zip(*batch)
    xx = list(xx)
    yy = list(yy)
    lens = [x.shape[-1] for x in xx]

    m = max(lens)
    with torch.no_grad():
        for i in range(len(xx)):
            pad = m - xx[i].shape[-1]
            xx[i] = F.pad(xx[i], (0, 0, 0, pad), 'constant', 0)
            yy[i] = F.pad(yy[i], (0, 0, 0, pad), 'constant', 0)
    x_pad = torch.stack(xx)
    y_pad = torch.stack(yy)

    return [
        x_pad,
    ], [
        y_pad,
    ], [
        lens,
    ]


def dummy_collate(batch):
    xx, yy = zip(*batch)
    return [
        torch.stack(xx),
    ], [
        torch.stack(yy),
    ], [
        torch.tensor(False),
    ]


def no_batch_collate(batch):
    xx, yy = zip(*batch)
    xx = list(xx)
    yy = list(yy)
    return xx, yy, [
        torch.tensor([
            xx[0].shape[-1],
        ] * xx[0].shape[0]),
    ]
