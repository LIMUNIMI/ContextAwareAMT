import argparse

import torch
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
        "-sc",
        "--scale",
        action="store_true",
        help="Create the midi file that must be synthesized for creating the template."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        action="store_true",
        help="Prepare the datasets by splitting the various contexts and resynthesizing them"
    )
    parser.add_argument(
        "-v",
        "--velocity",
        action="store_true",
        help="Perform actions for velocity estimation (note-wise prediction).")
    parser.add_argument(
        "-p",
        "--pedaling",
        action="store_true",
        help="Perform actions for pedaling estimation (frame-wise prediction)."
    )
    parser.add_argument("-t",
                        "--train",
                        action="store_true",
                        help="Train a model.")
    parser.add_argument(
        "-sk",
        "--skopt",
        action="store_true",
        help="Perform various little training cycles to look  for hyper-parameters using skopt."
    )
    parser.add_argument("-r",
                        "--redump",
                        action="store_true",
                        help="Pre-process the full dataset and dumps it")
    parser.add_argument(
        "-c",
        "--context",
        action="store",
        type=str,
        default=None,
        help="Limit the action to only the specified context (e.g. `-c pianoteq0`, `-c salamander1`, `-c orig`)"
    )
    parser.add_argument(
        "-pt",
        "--checkpoint",
        action="store",
        type=str,
        default=None,
        help="Load parameters from this checkpoint and freeze the initial weights if training."
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        default=None,
        nargs=2,
        help="Expects two inputs, namely a path to MIDI file and a path to audio file."
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

    if args.input:
        raise NotImplementedError(
            "Not yet implemented transcription from files")

    if args.template:
        from mpc2c import make_template
        make_template.main()
    if args.scale:
        from mpc2c import create_midi_scale
        create_midi_scale.main()
    if args.datasets:
        from pathlib import Path

        from mpc2c.asmd_resynth import split_resynth
        split_resynth(s.DATASETS,
                      Path(s.CARLA_PROJ), Path(s.RESYNTH_DATA_PATH),
                      Path(s.METADATASET_PATH), s.CONTEXT_SPLITS,
                      s.RESYNTH_FINAL_DECAY)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)['state_dict']
    else:
        checkpoint = None

    nmf_params = load_nmf_params()
    if args.skopt:
        from mpc2c import training
        from mpc2c.mytorchutils import hyperopt

        if args.pedaling:
            s.DATASET_LEN = 0.1
            space = s.PED_SKSPACE

            def objective(x):
                return training.train_pedaling(nmf_params, x, s.WD,
                                               args.context)

            space_constraint = training.model_test(
                training.build_pedaling_model, torch.rand(1, s.BINS - 1, 100))
        elif args.velocity:
            s.DATASET_LEN = 0.015
            space = s.VEL_SKSPACE
            space_constraint = training.model_test(
                training.build_velocity_model,
                torch.rand(1, s.BINS - 1, s.MINI_SPEC_SIZE))

            def objective(x):
                return training.train_velocity(nmf_params, x, s.WD,
                                               args.context)

        hyperopt(space,
                 s.SKCHECKPOINT,
                 s.SKITERATIONS,
                 objective,
                 space_constraint=space_constraint,
                 plot_graphs=False)

    if args.train:
        from mpc2c import training
        if args.pedaling:
            training.train_pedaling(nmf_params,
                                    s.PED_HYPERPARAMS,
                                    s.WD,
                                    context=args.context,
                                    state_dict=checkpoint)

        elif args.train_velocity:
            training.train_velocity(nmf_params,
                                    s.VEL_HYPERPARAMS,
                                    s.WD,
                                    context=args.context,
                                    state_dict=checkpoint)

    if args.redump:
        from mpc2c import data_management
        if args.pedaling:
            for split in ['train', 'validation', 'test']:
                data_management.get_loader(
                    [split, args.context] if args.context is not None else
                    [split], nmf_params, 'pedaling', True)
        elif args.velocity:
            for split in ['train', 'validation', 'test']:
                data_management.get_loader(
                    [split, args.context] if args.context is not None else
                    [split], nmf_params, 'velocity', True)


if __name__ == "__main__":
    main()
