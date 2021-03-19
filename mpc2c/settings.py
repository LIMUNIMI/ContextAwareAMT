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
SCALE_PATH = ['scales.mid', 'pianoteq_scales.mp3']
CARLA_PROJ = './carla_proj'
TEMPLATE_PATH = 'nmf_template.pkl'
IMAGES_PATH = './images/'
RESULT_PATH = './result/'

# resynthesis of the datasets
DATASETS = ["Maestro"]
CONTEXT_SPLITS = [120, 15, 25]
RESYNTH_FINAL_DECAY = 4

# GENERIC
SR = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512
#: number of jobs used
NJOBS = 5
# number of mfcc
BINS = 13
SPEC = essentiaspec.Spectrometer(
    FRAME_SIZE,
    SR,
    'hann',
    hop=HOP_SIZE,
    transform=essentiaspec.Transform.PowerSpectrum,
    proctransform=essentiaspec.ProcTransform.NONE)
MFCC = essentiaspec.ProcTransform.MFCC(SR,
                                       FRAME_SIZE // 2 + 1,
                                       logType='dbpow')
RETUNING = False

# NMF
#: epsilon value used inside the nmf to prevent divisons by 0
EPS = 1.0**-52
#: width of mini spectrograms: the number of frames after the onset; if the
#: note lasts longer than this value, it is trimmed, if it lasts less it is
#: padded with `PADDING_VALUE`
MINI_SPEC_SIZE = 30
#: value used for padding mini specs when their width is < MINI_SPEC_SIZE
PADDING_VALUE = 0
#: cost function used in the NMF
NMF_COST_FUNC = 'EucDist'
EPS_RANGE = 0
#: value used for range around activations
EPS_ACTIVATIONS = 0

# MAKE_TEMPLATE
#: how many basis use in total (except the last one)
BASIS = 19
#: the number of frames for the attack
ATTACK = 1
#: the number of frames for the other basis
BASIS_L = 1

#: on of "pad" or "stretch": the strategy used to have midi and audio with the
#: same length; just use "pad" for Maestro
PREPROCESSING = "pad"

# NN
MAX_LAYERS = 30
DEVICE = 'cuda'
EPOCHS = 500
VEL_HYPERPARAMS = {
    'lstm_layers': 0,
    'lstm_hidden_size': 2,
    'middle_features': 0,
    "kernel_0": 4,
    "kernel_1": 10,
    "middle_activation": AbsLayer,
    "sigmoid_last": True
}
PED_HYPERPARAMS = {
    'lstm_layers': 0,
    'lstm_hidden_size': 3,
    'middle_features': 0,
    "kernel_0": 6,
    "middle_activation": nn.Identity,  # or AbsLayer?
    # "sigmoid_last": False
    "sigmoid_last": True
    # skopt returned `False`, but dependency graph showed `True`; moreover,
    # training on original context showed a slightly lower loss for `True`
    # (about 1e-3 lower than `False`)
}
VEL_BATCH_SIZE = 500
PED_BATCH_SIZE = 1
EARLY_STOP = 20
EARLY_RANGE = 1e-8
TRAIN_DROPOUT = 0.1
PLOT_LOSSES = True
DTYPE = torch.float64
WD = 0
#: percentage of the dataset to use, use it for debugging or for skopt
DATASET_LEN = 1
LR_K = 5

# Transfer-learning
# 0%, 69%, 91% = 0, 2, all but 1 conv layers
PED_STEP = [0, 5, 8]
# 0%, 66%, 94% = 0, 3, all but 1 conv layers
VEL_STEP = [0, 8, 17]
TRANSFER_WD = 0
TRANSFER_DROPOUT = 0.1
TRANSFER_VEL_BATCH_SIZE = 500
TRANSFER_LR_K = 5

# SKOPT
VEL_SKSPACE = [
    space.Integer(0, 2, name='lstm_layers'),
    space.Integer(0, 7, name='lstm_hidden_size'),
    space.Integer(0, 7, name='middle_features'),
    space.Integer(3, 6, name='kernel_0'),
    space.Integer(3, 10, name='kernel_1'),
    space.Categorical([nn.ReLU, nn.Identity, AbsLayer, nn.Tanh],
                      name='middle_activation'),
    space.Categorical([True, False], name='sigmoid_last'),
]
PED_SKSPACE = [
    space.Integer(0, 4, name='lstm_layers'),
    space.Integer(0, 7, name='lstm_hidden_size'),
    space.Integer(0, 7, name='middle_features'),
    space.Integer(3, 6, name='kernel_0'),
    space.Categorical([nn.ReLU, nn.Identity, AbsLayer, nn.Tanh],
                      name='middle_activation'),
    space.Categorical([True, False], name='sigmoid_last'),
]
SKCHECKPOINT = 'skopt_checkpoint.pkl'
SKITERATIONS = (0, 80)
PLOT_GRAPHS = True
COMPLEXITY_PENALIZER = 1e-5

#: If compiling code with cython in pure-python mode
BUILD = False
