import torch
from skopt import space

# PATHS
TEMPLATE_PATH = 'nmf_template.pkl'
VELOCITY_DATA_PATH = '/datasets/mpc2c/velocity/'
PEDALING_DATA_PATH = '/datasets/mpc2c/pedaling/'
SCALE_PATH = ['scales.mid', 'pianoteq_scales.mp3']

# NMF
#: epsilon value used inside the nmf to prevent divisons by 0
EPS = 2.0**-52
#: width of mini spectrograms centered around the maximum value
MINI_SPEC_SIZE = 5
#: value used for padding mini specs when their width is < MINI_SPEC_SIZE
PADDING_VALUE = -1e-15
#: cost function used in the NMF
NMF_COST_FUNC = 'EucDist'
EPS_RANGE = 0
#: value used for range around activations
EPS_ACTIVATIONS = 0
#: if True, recreate data (also set by --redump)
REDUMP = False
# NN
DEVICE = 'cuda'
EPOCHS = 500
VEL_HYPERPARAMS = {
    "lr": 0.5,
    "wd": 0.01,
    "kernel": 4,
    "stride": 1,
    "dilation": 5
}
PED_HYPERPARAMS = {
    "lr": 0.5,
    "wd": 0.01,
    "kernel": 4,
    "stride": 1,
    "dilation": 5
}
VEL_BATCH_SIZE = 600
PED_BATCH_SIZE = 1
EARLY_STOP = 10
PLOT_LOSSES = True
DTYPE = torch.float32
#: percentage of the dataset to use, use it for debugging or for skopt
DATASET_LEN = 1

# SKOPT
SKSPACE = [
    space.Real(0.0001, 2, name='lr'),
    space.Real(0.0001, 1, name='wd'),
    space.Integer(1, 32, name='kernel'),
    space.Integer(1, 32, name='stride'),
    space.Integer(1, 16, name='dilation')
]
SKCHECKPOINT = 'skopt_checkpoint.pkl'
SKITERATIONS = 10**4

# MAKE_TEMPLATE
#: how many basis use in total
BASIS = 20
#: number of bins expected from the spectrogram (this depends on the number of
#: bins per semitone...)
BINS = 256
#: the number of frames for the attack
ATTACK = 1
#: the number of frames for the other basis
BASIS_L = 1

# GENERIC
SR = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512
#: datasets used for training the model
DATASETS = ["Maestro"]
#: number of jobs used
NJOBS = 10

#: on of "pad" or "stretch": the strategy used to have midi and audio with the
#: same length; just use "pad" for Maestro
PREPROCESSING = "pad"

#: If compiling code with cython in pure-python mode
BUILD = False
