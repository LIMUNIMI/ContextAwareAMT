"""
A new API for spectrograms with Essentia
"""
import essentia.standard as esst
import numpy as np
import scipy


def peaks_enhance(arr: np.ndarray,
                  alpha: float,
                  th: float,
                  axis=None) -> np.ndarray:
    """
    Apply a function to enhance peaks and remove other points.
    If `axis` is specified, it works only along that axis

    The function is defined as follows:

                        alpha
                       x
    f(x) = --------------------------------
                                 alpha-1
            alpha * (th * max(x))

    The idea is to apply a function that has derivative
        * f'(x) < 1 for values < th * max(x)
        * f'(x) >= 1 for values >= th * max(x)

    Denominator is adjusted by adding 1e-15 to prevent divisions by 0.

    Then, everything < th * max(f(x)) is put to 0.
    """

    # apply peaks_enhancing function
    arr = (arr**alpha) / (alpha *
                          (th * arr.max(axis=axis))**(alpha - 1) + 1e-15)

    # remove everything less than 0.25 of the peak
    critical_point = arr.max(axis=axis) / 4
    arr[arr < critical_point] = 0

    return arr


def midi_pitch_to_f0(midi_pitch: np.ndarray) -> np.ndarray:
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch - 69) / 12)


class EssentiaClass(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._state = (args, kwargs)

    def __repr__(self):
        return "(" + self.__class__.__name__ + ", " + str(self._state) + ")"

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])

    def __getstate__(self):
        return self._state

    def __eq__(self, other):
        try:
            return self._state == other._state
        except Exception:
            return False


class ClassCollection(type):
    def __iter__(self):
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                yield value

    def __repr__(self):
        classname = self.__dict__['__dict__'].__qualname__[:-8]
        repr = self.__module__ + classname + "\n"
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                repr += "\t" + attr + ": " + str(value) + "\n"
        return repr


class Transform(metaclass=ClassCollection):
    """
    Possible transform types.

    Each should accept `size` and `sample_rate` arguments and return the
    transformed array. If the field `bin_width` is not None, it can be used
    for retuning
    """
    class FFT(EssentiaClass):
        def __init__(self, size, sample_rate, **kwargs):
            super().__init__(size, sample_rate, **kwargs)
            self.spec = esst.Spectrum(size=size, **kwargs)
            self.bin_width = (sample_rate // 2) / (size // 2 + 1)

        def __call__(self, x):
            return self.spec(x)

    class CQT(EssentiaClass):
        def __init__(self, size, sample_rate, **kwargs):
            super().__init__(size, sample_rate, **kwargs)
            self.spec = esst.SpectrumCQ(sampleRate=sample_rate, **kwargs)
            self.bin_width = None

        def __call__(self, x):
            return self.spec(x)

    class DCT(EssentiaClass):
        def __init__(self, size, sample_rate, **kwargs):
            super().__init__(size, sample_rate, **kwargs)
            self.spec = esst.DCT(inputSize=size, **kwargs)
            self.bin_width = None

        def __call__(self, x):
            return self.spec(x)


class ProcTransform(metaclass=ClassCollection):
    """

    Possible methods to process an FFT spectrum.

    Each of these should have `sample_rate` and `spectrum_size` parameter, plus
    any other keyword parameter for that particular post-processing method

    Each of these should accept the spectrum

    Each of these should return the transformed array
    """
    class LOG(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.LogSpectrum(sampleRate=sample_rate,
                                         frameSize=spectrum_size,
                                         **kwargs)

        def __call__(self, x):
            return self.spec(x)[0]

    class CENT(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.SpectrumToCent(sampleRate=sample_rate,
                                            inputSize=spectrum_size,
                                            **kwargs)

        def __call__(self, x):
            return self.spec(x)[0]

    class PITCH_BANDS(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, log=False, **kwargs):
            """
            This uses `FrequencyBands` because 128 bands are too many for
            `TriangularBands`; however, it accept a keyword parameter `log`.
            """
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.FrequencyBands(sampleRate=sample_rate,
                                            frequencyBands=[
                                                midi_pitch_to_f0(pitch - 0.5)
                                                for pitch in range(1, 130)
                                            ])
            self.log = log

        def __call__(self, x):
            if self.log:
                return np.log10(1 + self.spec(x))
            else:
                return self.spec(x)

    class BARK(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.TriangularBarkBands(
                sampleRate=sample_rate,
                inputSize=spectrum_size,
                highFrequencyBound=sample_rate // 2,
                **kwargs)

        def __call__(self, x):
            return self.spec(x)

    class BFCC(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.BFCC(sampleRate=sample_rate,
                                  inputSize=spectrum_size,
                                  highFrequencyBound=sample_rate // 2,
                                  **kwargs)

        def __call__(self, x):
            return self.spec(x)[1]

    class MEL(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.MelBands(sampleRate=sample_rate,
                                      inputSize=spectrum_size,
                                      highFrequencyBound=sample_rate // 2,
                                      **kwargs)

        def __call__(self, x):
            return self.spec(x)

    class MFCC(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.MFCC(sampleRate=sample_rate,
                                  inputSize=spectrum_size,
                                  highFrequencyBound=sample_rate // 2,
                                  **kwargs)

        def __call__(self, x):
            return self.spec(x)[1]

    class ERB(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.ERBBands(sampleRate=sample_rate,
                                      inputSize=spectrum_size,
                                      highFrequencyBound=sample_rate // 2,
                                      **kwargs)

        def __call__(self, x):
            return self.spec(x)

    class GFCC(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            self.spec = esst.GFCC(sampleRate=sample_rate,
                                  inputSize=spectrum_size,
                                  highFrequencyBound=sample_rate // 2,
                                  **kwargs)

        def __call__(self, x):
            return self.spec(x)[1]

    class NONE(EssentiaClass):
        def __init__(self, sample_rate, spectrum_size, **kwargs):
            super().__init__(sample_rate, spectrum_size, **kwargs)
            pass

        def __call__(self, x):
            return x


class Spectrometer():
    """
    Creates an object to compute spectrograms with given parameters.
    It combines `Transform` and `ProcTransform` objects to compute
    spectrogram-like representations of arrays; optionally, it can retune the
    arrays by estimating the original tuning with the Essentia algorithm.
    """

    SUB_BINS = 100

    def __call__(self, frame: np.array):
        return self.apply(frame)

    def apply(self,
              frame: np.array,
              retuning_step: float = 0.0,
              sub_bins: np.array = None,
              sub_bins_width: np.array = None,
              new_bins: np.array = None):
        """
        If `retuning_step` is != 0, this function also tries to retune "audio"
        by increasing/decreasing of the specified frequency step; this is only
        supported if using FFT.

        `sub_bins` and `sub_bins_width` must be provided if `retuning_step` !=
        0
        """
        data = self.transform(frame)
        if self.transform.bin_width is not None and retuning_step != 0:
            # this works with fixed bin width only. Simply uses
            # `scipy.ndimage.shift` with `mode='constant', cval=0, order=1`.
            # The shift is computed as `retuning_step / bin_width`.
            data = scipy.ndimage.shift(data,
                                       retuning_step /
                                       self.transform.bin_width,
                                       order=1,
                                       mode='constant',
                                       cval=0)

        return self.proctransform(data)

    def spectrogram(self,
                    audio: np.array,
                    hop: int = 0,
                    retuning: float = 440.0):
        """
        Process a full audio array using the specified hop size
        If `retuning` is > 0, this function also tries to retune "audio" to the
        specified value; this is only supported if using FFT.

        If `hop` is 0 (default), the one provided in the constructor is used.
        If that one is 0 too, an exception is raised
        """
        if hop == 0:
            if self.hop_size == 0:
                raise RuntimeError("Hop-size specified is 0!")
            else:
                hop = self.hop_size

        if retuning > 0 and self.transform.bin_width is not None:
            retuning_step = retuning - esst.TuningFrequencyExtractor(
                frameSize=self.frame_size, hopSize=hop)(audio).mean()

        else:
            retuning_step = 0

        spectrogram = []
        for frame in esst.FrameGenerator(audio,
                                         frameSize=self.frame_size,
                                         hopSize=hop,
                                         startFromZero=True):
            spectrogram.append(self.apply(frame, retuning_step=retuning_step))

        return np.array(spectrogram, dtype=np.float32).T

    def __init__(self,
                 frame_size: int,
                 sample_rate: int = 44100,
                 transform: Transform = Transform.FFT,
                 proctransform: ProcTransform = ProcTransform.LOG,
                 transform_params: dict = {},
                 proctransform_params: dict = {},
                 hop: int = 0):
        """
        Arguments
        ---------

        `frame_size` : int
            the expected size of the frames
        `sample_rate` : int
            the expected size of the sample rates (not needed for FFT and DCT
            with no post-process)
        `transform` : Transform
            the kind of transform to use: FFT, CQT or DCT
        `proctransform` : ProcTransform
            the kind of post-processing to use: none, log, bark, mel, erb,
            mfcc, bfcc, gfcc, cent; it is recommended to use these with FFT
            only
        `transform_params` : dict
            any other arameter to be passed to the transform function
        `proctransform_params` : dict
            any other parameter to be passed to the postprocess function
        `hop` : int
            the hop size used for the spectrogram. If 0 (default) you should
            specify it when calling `spectrogram` method
        """
        self.frame_size = int(frame_size)
        self.sample_rate = int(sample_rate)
        self.hop_size = hop
        self.transform_params = transform_params
        self.proctransform_params = transform_params
        self.hop = hop

        self.transform = transform(size=frame_size,
                                   sample_rate=sample_rate,
                                   **transform_params)

        self.proctransform = proctransform(spectrum_size=frame_size // 2 + 1,
                                           sample_rate=sample_rate,
                                           **proctransform_params)

    def __repr__(self):
        return f"""<Spectrometer:
    frame_size: {self.frame_size},
    sample_rate: {self.sample_rate},
    hop_size: {self.hop},
    transform: {self.transform},
    proctransform: {self.proctransform},
    transform_params: {self.transform_params},
    proctransform_params: {self.proctransform_params}"""
