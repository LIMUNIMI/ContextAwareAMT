import torch

# NMF
#: epsilon value used inside the nmf to prevent divisons by 0
EPS = 2.0**-52
#: width of mini spectrograms centered around the maximum value
MINI_SPEC_SIZE = 5
#: value used for padding mini specs when their width is < MINI_SPEC_SIZE
PADDING_VALUE = -1e-15
#: cost function used in the NMF
NMF_COST_FUNC = 'EucDist'
#: number of songs considered for training
NUM_SONGS_FOR_TRAINING = 1000
EPS_RANGE = 0
#: value used for range around activations
EPS_ACTIVATIONS = 0
#: percentage of the dataset to use, use it for debugging
DATASET_LEN = 1

# NN
DEVICE = 'cuda'
EPOCHS = 500
BATCH_SIZE = 5
EARLY_STOP = 10
BRANCHES = 16
LR_VELOCITY = 1
LR_PEDALING = 1
KERNEL = 3
STRIDE = 1
DILATION = 5
PLOT_LOSSES = True
DTYPE = torch.float64

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

# PATHS
TEMPLATE_PATH = 'nmf_template.pkl'
MINI_SPEC_PATH = 'mini_spec.pkl.gz'
DIFF_SPEC_PATH = 'diff_spec.pkl.gz'
SCALE_PATH = ['scales.mid', 'pianoteq_scales.mp3']

#: on of "pad" or "stretch": the strategy used to have midi and audio with the
#: same length; just use "pad" for Maestro
preprocessing = "pad"

#: use the following for debugging
BUILD = False
