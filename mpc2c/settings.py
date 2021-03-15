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
EPS = 2.0**-52
#: width of mini spectrograms: the number of frames after the onset; if the
#: note lasts longer than this value, it is trimmed, if it lasts less it is
#: padded with `PADDING_VALUE`
MINI_SPEC_SIZE = 5
#: value used for padding mini specs when their width is < MINI_SPEC_SIZE
PADDING_VALUE = -1e-15
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
    'lstm_hidden_size': 5,
    'middle_features': 0,
    "kernel_0": 3,
    "kernel_1": 5,
    "middle_activation": nn.Tanh,
    "sigmoid_last": True
}
PED_HYPERPARAMS = {
    'lstm_layers': 0,
    'lstm_hidden_size': 3,
    'middle_features': 0,
    "kernel_0": 6,
    "middle_activation": nn.Identity,  # or AbsLayer?
    "sigmoid_last":
    True  # skopt returned `False`, but dependency graph showed `True`
}
VEL_BATCH_SIZE = 5
PED_BATCH_SIZE = 1
EARLY_STOP = 10
EARLY_RANGE = 1e-4
TRAIN_DROPOUT = 0.1
PLOT_LOSSES = True
DTYPE = torch.float32
WD = 0
#: percentage of the dataset to use, use it for debugging or for skopt
DATASET_LEN = 1
LR_K = 5

# Transfer-learning
# 100%, 35%, 70% = 0, 1, all but 2 conv layers
PED_STEP = [0, 3, 5]
# 100%, 47%, 84% = 0, 1, all but 2 conv layers
VEL_STEP = [0, 3, 14]
TRANSFER_WD = 0
TRANSFER_DROPOUT = 0.1
TRANSFER_EARLY_RANGE = 1e-4
TRANSFER_EARLY_STOP = 20
TRANSFER_VEL_BATCH_SIZE = 500

# SKOPT
VEL_SKSPACE = [
    space.Integer(0, 2, name='lstm_layers'),
    space.Integer(0, 7, name='lstm_hidden_size'),
    space.Integer(0, 7, name='middle_features'),
    space.Integer(3, 6, name='kernel_0'),
    space.Integer(3, 5, name='kernel_1'),
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
