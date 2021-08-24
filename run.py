import argparse
import pickle
from pathlib import Path

import torch

from mpc2c import create_template, data_management, evaluate
from mpc2c import settings as s
from mpc2c import training
from mpc2c.asmd_resynth import split_resynth, get_contexts
from mpc2c.mytorchutils import hyperopt
from mpc2c import build

build.build()


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for running experiments")
    parser.add_argument(
        "-sc",
        "--scale",
        action="store_true",
        help=
        "Create the midi file containing the scales for the template; syntehsize it and make the template."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        action="store_true",
        help=
        "Prepare the datasets by splitting the various contexts and resynthesizing them"
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
        "-g",
        "--generic",
        action="store_true",
        help="Use generic indipendence instead of specific indipendence")

    parser.add_argument(
        "-sk",
        "--skopt",
        action="store_true",
        help=
        "Perform various little training cycles to look  for hyper-parameters using skopt."
    )
    parser.add_argument("-r",
                        "--redump",
                        action="store_true",
                        help="Pre-process the full dataset and dumps it")
    parser.add_argument(
        "-pt",
        "--checkpoint",
        action="store",
        type=str,
        default=None,
        help=
        "Load parameters from this checkpoint and freeze the initial weights if training."
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store",
        type=str,
        default=None,
        nargs='+',
        help=
        "Evaluate the error distribution of model checkpoints given as argument. All contexts available in `settings.CARLA_PROJ` will be used, plus the 'orig' context. All models are evaluated on all contexts."
    )
    parser.add_argument(
        "-cp",
        "--compare",
        action="store_true",
        help=
        "Only valid if `--evaluate` is used. Using this option, you can name your models starting with the context on which they were trained (e.g. `pianoteq0_vel.pt`); in this way, one more plot is created, representing the `orig` model compared to the other models on their specific context."
    )
    parser.add_argument(
        "-cf",
        "--csv-file",
        action="store",
        type=str,
        default=None,
        nargs='+',
        help=
        "Expects at least one input, namely paths to csv files containing the saved tests that should be plotted"
    )
    return parser.parse_args()


def load_nmf_params():
    nmf_params = pickle.load(open(s.TEMPLATE_PATH, 'rb'))
    print("using minpitch: ", nmf_params[1])
    print("using maxpitch: ", nmf_params[2])
    return nmf_params


def main():

    args = parse_args()

    if args.scale:
        create_template.main()
    if args.datasets:

        split_resynth(s.DATASETS,
                      Path(s.CARLA_PROJ), Path(s.RESYNTH_DATA_PATH),
                      Path(s.METADATASET_PATH), s.CONTEXT_SPLITS,
                      s.RESYNTH_FINAL_DECAY)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)['state_dict']
        s.VEL_BATCH_SIZE = s.TRANSFER_VEL_BATCH_SIZE
    else:
        checkpoint = None

    if args.pedaling:
        mode = 'pedaling'
        hpar = s.PED_HYPERPARAMS
    elif args.velocity:
        mode = 'velocity'
        hpar = s.VEL_HYPERPARAMS

    nmf_params = load_nmf_params()
    if args.skopt:

        contexts = list(get_contexts(s.CARLA_PROJ).keys())

        def objective(x):
            return training.train(x, mode, generic=args.generic)

        if args.pedaling:
            s.DATASET_LEN = 0.1
            space = s.PED_SKSPACE
            test_sample = torch.rand(1, s.BINS, 100)
            checkpoint_path = "ped_skopt.pt"

        elif args.velocity:
            s.DATASET_LEN = 1e-3
            space = s.VEL_SKSPACE
            test_sample = torch.rand(1, s.BINS, s.MINI_SPEC_SIZE)
            checkpoint_path = "vel_skopt.pt"
        else:
            raise RuntimeError("Velocity or pedaling must be set")

        space_constraint = training.model_test(
            lambda x: training.build_model(x, mode, contexts), test_sample)

        hyperopt(space,
                 checkpoint_path,
                 s.SKITERATIONS,
                 objective,
                 space_constraint=space_constraint,
                 plot_graphs=True)

    if args.train:

        print("----------------")
        training.train(hpar,
                       mode,
                       copy_checkpoint=Path("models") / f"{mode}.pt",
                       generic=args.generic)

    if args.redump:
        contexts = list(get_contexts(s.CARLA_PROJ).keys())
        for split in ['train', 'validation', 'test']:
            data_management.get_loader(split,
                                       redump=True,
                                       contexts=contexts,
                                       mode=mode,
                                       nmf_params=nmf_params)

    if args.evaluate or args.csv_file:
        compare = args.compare
        if args.csv_file:
            for fname in args.csv_file:
                evaluate.plot_from_file(fname,
                                        compare=compare,
                                        mode=mode,
                                        ext='.svg')
        else:

            dfs = evaluate.evaluate(args.evaluate, mode, Path(s.RESULT_PATH))

            for i, df in enumerate(dfs):
                evaluate.plot(df,
                              compare,
                              mode=mode,
                              save=Path(s.IMAGES_PATH) / f"{mode}_eval.{i}")


if __name__ == "__main__":
    main()
