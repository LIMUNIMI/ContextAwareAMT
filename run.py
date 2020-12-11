# from mpc2c import nmf
import argparse

# from cylang import cylang
# cylang.compile()
from Cython.Build import Cythonize

from mpc2c import settings as s

if s.BUILD:
    Cythonize.main(["mpc2c/[!data_management.py]**.py", "-3", "--inplace"])


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for running experiments")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Create the initial template from `pianoteq_scales.mp3` and `scales.mid`"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Create the midi file that must be synthesized for creating the template."
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Prepare the datasets by splitting the various contexts and resynthesizing them"
    )
    parser.add_argument(
        "--train-velocity",
        action="store_true",
        help="Train the neural network for velocity estimation.")
    parser.add_argument(
        "--train-pedaling",
        action="store_true",
        help="Train the neural network for pedaling estimation.")
    parser.add_argument(
        "--skopt",
        action="store_true",
        help="If activated, instead of a full training, performs various little training cycles to look  for hyper-parameters using skopt."
    )
    parser.add_argument(
        "--redump",
        action="store_true",
        help="If used, it pre-processes the full dataset and dumps it before of starting training procedure."
    )
    return parser.parse_args()


def load_nmf_params():
    import pickle
    nmf_params = pickle.load(open(s.TEMPLATE_PATH, 'rb'))
    print("using minpitch: ", nmf_params[1])
    print("using maxpitch: ", nmf_params[2])
    return nmf_params


def main():
    args = parse_args()
    if args.skopt:
        # if we are hyper-optimizing, change some settings
        from mpc2c.mytorchutils import hyperopt
        s.DATASET_LEN = 0.01
        s.PLOT_LOSSES = False
    s.REDUMP = args.redump

    if args.template:
        from mpc2c import make_template
        make_template.main()
    if args.scale:
        from mpc2c import create_midi_scale
        create_midi_scale.main()
    if args.datasets:
        from mpc2c.asmd_resynth import split_resynth
        split_resynth(s.DATASETS, s.CARLA_PROJ, s.RESYNTH_DATA_PATH,
                      s.CONTEXT_SPLITS, s.RESYNTH_FINAL_DECAY)
    if args.train_pedaling:
        from mpc2c import training
        nmf_params = load_nmf_params()
        if args.skopt:
            hyperopt(s.SKSPACE, s.SKCHECKPOINT, s.SKITERATIONS,
                     lambda x: training.train_pedaling(nmf_params, x))
        else:
            training.train_pedaling(nmf_params, s.VEL_HYPERPARAMS)

    if args.train_velocity:
        from mpc2c import training
        nmf_params = load_nmf_params()
        if args.skopt:
            hyperopt(s.SKSPACE, s.SKCHECKPOINT, s.SKITERATIONS,
                     lambda x: training.train_velocity(nmf_params, x))
        else:
            training.train_velocity(nmf_params, s.PED_HYPERPARAMS)


if __name__ == "__main__":
    main()
