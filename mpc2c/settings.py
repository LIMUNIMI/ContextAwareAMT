import torch
from skopt import space
from torch import nn

from . import essentiaspec
from .feature_extraction import AbsLayer

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
NJOBS = 4
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
MINI_SPEC_SIZE = 30
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
EPOCHS = 500
VEL_HYPERPARAMS = {
    'lstm_layers': 2,
    'lstm_hidden_size': 7,
    'encoder_features': 5,
    "kernel_0": 3,
    "kernel_1": 3,
    "middle_activation": nn.ReLU,
    'latent_features': 3,
    'performer_features': 7,
    'performer_layers': 3
}
# TODO: redo pedaling
PED_HYPERPARAMS = {
    'lstm_layers': 1,
    'lstm_hidden_size': 5,
    'middle_features': 0,
    "kernel_0": 4,
    "middle_activation": nn.Tanh  # or AbsLayer?
}
VEL_BATCH_SIZE = 100
PED_BATCH_SIZE = 1
EARLY_STOP = 20
EARLY_RANGE = 1e-8
TRAIN_DROPOUT = 0.1
PLOT_LOSSES = True
DTYPE = torch.float32
PRECISION = 32
WD = 0
#: percentage of the dataset to use, use it for debugging or for skopt
DATASET_LEN = 1
LR_K = 5

# SKOPT
# TODO: SKSPACE!
PED_SKSPACE = [
    space.Integer(0, 4, name='lstm_layers'),
    space.Integer(0, 7, name='lstm_hidden_size'),
    space.Integer(0, 7, name='latent_features'),
    space.Integer(0, 7, name='encoder_features'),
    space.Integer(3, 6, name='kernel_0'),
    space.Categorical([nn.ReLU, nn.Identity, AbsLayer, nn.Tanh],
                      name='middle_activation'),
    space.Integer(3, 6, name='performer_layers'),
    space.Integer(0, 7, name='performer_features'),
]
VEL_SKSPACE = PED_SKSPACE + [space.Integer(3, 10, name='kernel_1')]
SKITERATIONS = (0, 40)
PLOT_GRAPHS = True
COMPLEXITY_PENALIZER = 0 # 1e-6

#: If compiling code with cython in pure-python mode
BUILD = False
#: If cleaning cython files before building
CLEAN = True
