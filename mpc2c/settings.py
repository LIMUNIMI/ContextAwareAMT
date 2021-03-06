import torch
from torch import nn

from . import essentiaspec

# PATHS
VELOCITY_DATA_PATH = '/datasets/mpc2c/resynth/velocity/'
PEDALING_DATA_PATH = '/datasets/mpc2c/resynth/pedaling/'
RESYNTH_DATA_PATH = '/datasets/mpc2c/resynth/'
METADATASET_PATH = './metadataset.json'
SCALE_PATH = ['scales.mid', 'pianoteq_scales.flac']
SCALE_DIR = './to_be_synthesized'
CARLA_PROJ = './carla_proj'
TEMPLATE_PATH = 'nmf_template.pkl'
IMAGES_PATH = './images/'
RESULT_PATH = './result/'

# resynthesis of the datasets
DATASETS = ["Maestro"]
CONTEXT_SPLITS = [160, 22, 29]
RESYNTH_FINAL_DECAY = 4

# GENERIC
SR = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512
#: number of jobs used
NJOBS = 10
# number of mfcc
BINS = 13
SPEC = essentiaspec.Spectrometer(FRAME_SIZE,
                                 SR,
                                 'hann',
                                 hop=HOP_SIZE,
                                 transform=essentiaspec.Transform.Spectrum,
                                 proctransform=essentiaspec.ProcTransform.NONE)
MFCC = essentiaspec.ProcTransform.MFCC(SR,
                                       FRAME_SIZE // 2 + 1,
                                       logType='dbamp')
RETUNING = False

# NMF
#: epsilon value used inside the nmf to prevent divisons by 0
EPS = 1e-32
#: width of mini spectrograms: the number of frames after the onset; if the
#: note lasts longer than this value, it is trimmed, if it lasts less it is
#: padded with `PADDING_VALUE`
SPEC_LEN = 30
#: value used for padding mini specs when their width is < MINI_SPEC_SIZE
PADDING_VALUE = 1
#: cost function used in the NMF
NMF_COST_FUNC = 'EucDist'
EPS_RANGE = 0
#: value used for range around activations
EPS_ACTIVATIONS = 0

# MAKE_TEMPLATE
#: number of velocity layers in the scale
N_VELOCITY_LAYERS = 20
MIN_VEL = 10
MAX_VEL = 120
#: different note durantion
NOTE_DURATION = [0.1, 1.5]
#: silence between the notes
NOTE_SILENCE = [0.5, 1.5]
#: a carla project to synthesize the scale
SCALE_PROJ = 'scale.carxp'
#: how many basis use in total (except the first and release)
BASIS_FRAMES = {
    #: the number of basis for the attack
    'attack_b': 1,
    #: the number of basis for the release
    'release_b': 15,
    #: the number of basis for the inner
    'inner_b': 14,
    #: the number of frames for the attack basis
    'attack_f': 1,
    #: the number of frames for the release basis
    'release_f': 1,
    #: the number of frames for the inner basis
    'inner_f': 2,
}

#: on of "pad" or "stretch": the strategy used to have midi and audio with the
#: same length; just use "pad" for Maestro
PREPROCESSING = "pad"

# NN
MAX_LAYERS = 30
DEVICE = 'cuda'
GPUS = 1
EPOCHS = 40
VEL_HYPERPARAMS = {
    "enc_k1": 6,
    "enc_k2": 8,
    "enc_kernel": 3,
    "spec_k1": 6,
    "spec_k2": 8,
    "spec_kernel": 5,
    "activation": nn.ReLU(),
}
PED_HYPERPARAMS = {
    "enc_k1": 6,
    "enc_k2": 8,
    "enc_kernel": 3,
    "spec_k1": 6,
    "spec_k2": 8,
    "spec_kernel": 5,
    "activation": nn.ReLU(),
}
VEL_BATCH_SIZE = 10
PED_BATCH_SIZE = 10
EARLY_STOP = 20
EARLY_RANGE = 1e-4
EMA_PERIOD = 15
TRAIN_DROPOUT = 0.1
DTYPE = torch.float32
PRECISION = 32
#: percentage of the dataset to use
VEL_DATASET_LEN = 1e-3
PED_DATASET_LEN = 3.5e-3
MAX_TIME_CONV_STACK = 30
MAX_SIZE_CONV_STACK = (1 * (2**30)) // 4  # ~ 1 GB
SWA = False

GRIDSPACE = {
    'enc_k1': [4],
    'enc_k2': [1, 2, 3],
    'enc_kernel': [3, 5],
    'spec_k1': [4],
    'spec_k2': [1, 2, 4],
    'spec_kernel': [3, 5],
}

#: If compiling code with cython in pure-python mode
BUILD = False
#: If cleaning cython files before building
CLEAN = True
