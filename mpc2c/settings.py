# TODO: move to yaml or json

# NMF
NMF_DATASETS = ["Maestro"]
EPS = 2.0**-52
MINI_SPEC_SIZE = 14
PADDING_VALUE = -1
DEVICE = 'cuda'
NJOBS = 5
NUM_SONGS_FOR_TRAINING = 100
EPOCHS = 500
BATCH_SIZE = 400
EARLY_STOP = 10
BRANCHES = 16
DATASET_LEN = 1  # use this for debugging
EPS_RANGE = 0.1

# MAKE_TEMPLATE
SR = 22050
FRAME_SIZE = 16384
HOP_SIZE = 1024
BASIS = 20
BINS = 100
# the number of frames for the attack
ATTACK = 1
# the number of frames for the other basis
BASIS_L = 1
TEMPLATE_PATH = 'nmf_template.pkl'

SCALE_PATH = ['to_be_synthesized/scales.mid', 'audio/pianoteq_scales.mp3']
