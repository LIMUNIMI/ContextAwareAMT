# from mpc2c import nmf
import argparse

# from cylang import cylang
# cylang.compile()
from Cython.Build import Cythonize

from mpc2c import settings as s

if s.BUILD:
    Cythonize.main(["mpc2c/**.py", "-3", "--inplace"])


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
    parser.add_argument("--datasets",
                        action="store_true",
                        help="Create the datasets using NMF.")
    parser.add_argument("--train-velocity",
                        action="store_true",
                        help="Train the neural network for velocity estimation.")
    parser.add_argument("--train-pedaling",
                        action="store_true",
                        help="Train the neural network for pedaling estimation.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.template:
        from mpc2c import make_template
        make_template.main()
    if args.scale:
        from mpc2c import create_midi_scale
        create_midi_scale.main()
    if args.datasets:
        from mpc2c import nmf
        import pickle
        nmf_params = pickle.load(open(s.TEMPLATE_PATH, 'rb'))
        nmf.create_datasets(nmf_params, s.MINI_SPEC_PATH, s.DIFF_SPEC_PATH,
                            "train")
        nmf.create_datasets(nmf_params, s.MINI_SPEC_PATH, s.DIFF_SPEC_PATH,
                            "valid")
    if args.train_pedaling:
        from mpc2c import training
        import pickle
        nmf_params = pickle.load(open(s.TEMPLATE_PATH, 'rb'))
        training.train_pedaling(nmf_params)

    if args.train_velocity:
        from mpc2c import training
        import pickle
        nmf_params = pickle.load(open(s.TEMPLATE_PATH, 'rb'))
        training.train_velocity(nmf_params)


if __name__ == "__main__":
    main()
