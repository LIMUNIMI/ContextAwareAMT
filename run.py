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
        "-tv",
        "--train-velocity",
        action="store_true",
        help="Train the neural network for velocity estimation.")
    parser.add_argument(
        "-tp",
        "--train-pedaling",
        action="store_true",
        help="Train the neural network for pedaling estimation.")
    parser.add_argument(
        "-sk",
        "--skopt",
        action="store_true",
        help="If activated, instead of a full training, performs various little training cycles to look  for hyper-parameters using skopt."
    )
    parser.add_argument(
        "-r",
        "--redump",
        action="store_true",
        help="If used, it pre-processes the full dataset and dumps it, then exits")
    parser.add_argument(
        "-c",
        "--context",
        action="store",
        type=str,
        default=None,
        help="If used, limits the processing to only the specified context (e.g. `-c pianoteq0`, `-c salamander1`, `-c orig`)"
    )
    parser.add_argument(
        "-gm",
        "--generic-model",
        action="store",
        type=str,
        default=None,
        help="If used, load parameters from the generic model and fix the initial weights if training."
    )
    parser.add_argument(
        "-cm",
        "--context-model",
        action="store",
        type=str,
        default=None,
        help="If used, load parameters from the specific model into th final weights. If `-gm` is used, this applies afterwards and overwrites the final part of the generic model weights."
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
    if args.skopt:
        # if we are hyper-optimizing, change some settings
        from mpc2c.mytorchutils import hyperopt
        if args.train_velocity:
            s.DATASET_LEN = 0.015
        elif args.train_pedaling:
            s.DATASET_LEN = 0.1
        s.PLOT_LOSSES = False
    s.REDUMP = args.redump

    if args.generic_model:
        generic_model = torch.load(args.generic_model)['state_dict']
    else:
        generic_model = None

    if args.context_model:
        context_model = torch.load(args.context_model)['state_dict']
    else:
        context_model = None

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
    if args.train_pedaling:
        from mpc2c import training
        nmf_params = load_nmf_params()
        if args.skopt:
            hyperopt(s.PED_SKSPACE,
                     s.SKCHECKPOINT,
                     s.SKITERATIONS,
                     lambda x: training.train_pedaling(nmf_params, x, s.WD,
                                                       args.context),
                     space_constraint=training.model_test(
                         training.build_pedaling_model,
                         torch.rand(1, s.BINS - 1, 100)),
                     plot_graphs=s.PLOT_GRAPHS)
        else:
            training.train_pedaling(nmf_params,
                                    s.PED_HYPERPARAMS,
                                    s.WD,
                                    context=args.context,
                                    state_dict=generic_model)

    if args.train_velocity:
        from mpc2c import training
        nmf_params = load_nmf_params()
        if args.skopt:
            hyperopt(s.VEL_SKSPACE,
                     s.SKCHECKPOINT,
                     s.SKITERATIONS,
                     lambda x: training.train_velocity(nmf_params, x, s.WD,
                                                       args.context),
                     space_constraint=training.model_test(
                         training.build_velocity_model,
                         torch.rand(1, s.BINS - 1, s.MINI_SPEC_SIZE)),
                     plot_graphs=s.PLOT_GRAPHS)
        else:
            training.train_velocity(nmf_params,
                                    s.VEL_HYPERPARAMS,
                                    s.WD,
                                    context=args.context,
                                    state_dict=generic_model)


if __name__ == "__main__":
    main()
