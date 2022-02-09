import argparse
import pickle
import shutil
import subprocess
from logging import error
from pathlib import Path

import mlflow

from mpc2c import build, create_template, data_management
from mpc2c import settings as s
from mpc2c import training, evaluate
from mpc2c.asmd_resynth import get_contexts, split_resynth

build.build()


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for running experiments")
    parser.add_argument(
        "-cm",
        "--clean-mlflow",
        action="store_true",
        help=
        "If used, mlflow experiment is cleaned before of running experiments")
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
        "-p",
        "--pedaling",
        action="store_true",
        help="Perform actions for pedaling estimation (window-wise prediction)."
    )
    parser.add_argument(
        "-v",
        "--velocity",
        action="store_true",
        help="Perform actions for velocity estimation (note-wise prediction).")
    parser.add_argument("-t",
                        "--train",
                        action="store_true",
                        help="Train a model.")
    parser.add_argument(
        "-cs",
        "--contextspecific",
        action="store_true",
        help=
        "Train a specializer against context specificity on the same latent space used for performance regression"
    )

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
        "-pc",
        "--printcontexts",
        action="store_true",
        help="Print contexts in the order with the labels shown in mlflow log")
    parser.add_argument("-e",
                        "--evaluate",
                        action="store_true",
                        help="Evaluate configurations")
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

    contexts = list(get_contexts(s.CARLA_PROJ).keys())

    if args.printcontexts:
        for i, c in enumerate(contexts):
            print(f"{i}: {c}")

    if args.pedaling:
        mode = 'pedaling'
        hpar = s.PED_HYPERPARAMS
    elif args.velocity:
        mode = 'velocity'
        hpar = s.VEL_HYPERPARAMS
    else:
        error("Please specify -p or -v")
        return

    nmf_params = load_nmf_params()

    if args.redump:
        contexts = list(get_contexts(s.CARLA_PROJ).keys())
        for split in ['train', 'validation', 'test']:
            data_management.get_loader(split,
                                       redump=True,
                                       contexts=contexts,
                                       one_context_per_batch=False,
                                       mode=mode,
                                       nmf_params=nmf_params)
    if args.skopt:

        def objective(x):
            l1, model = training.train(x, mode, False, False, test=True)
            # note: deepcopy causes some weakref errors...
            # saving a copy to disk instead
            try:
                pickle.dump(model, open("_model.pkl", "wb"))
            except Exception as e:
                raise RuntimeError("Error pickling model:" + str(e))
            model = pickle.load(open("_model.pkl", "rb"))
            l3, _ = training.train(x,
                                   mode,
                                   True,
                                   True,
                                   test=True,
                                   start_from_model=model)
            model = pickle.load(open("_model.pkl", "rb"))
            l2, _ = training.train(x,
                                   mode,
                                   True,
                                   False,
                                   test=True,
                                   start_from_model=model)
            model = pickle.load(open("_model.pkl", "rb"))
            l4, _ = training.train(x,
                                   mode,
                                   False,
                                   True,
                                   test=True,
                                   start_from_model=model)
            return (l1 + l2 + l3 + l4) / 4

        if args.pedaling:
            # test_sample = torch.rand(1, s.BINS, 100)
            checkpoint_path = "ped_grid.pt"

        elif args.velocity:
            # test_sample = torch.rand(1, s.BINS, s.MINI_SPEC_SIZE)
            checkpoint_path = "vel_grid.pt"
        else:
            return  # not reachable, here to shutup the pyright

        # space_constraint = training.model_test(
        #     lambda x: training.build_model(x, contexts), test_sample)
        exp = mlflow.get_experiment_by_name(mode)
        if exp and args.clean_mlflow:
            if exp.lifecycle_stage == 'deleted':
                exp_path = Path(
                    mlflow.get_registry_uri()) / '.trash' / exp.experiment_id
            else:
                exp_path = exp.artifact_location
            shutil.rmtree(exp_path)

        training.grid_search(s.GRIDSPACE, objective, checkpoint_path)

        exp = mlflow.get_experiment_by_name(mode)
        subprocess.run([
            'mlflow', 'experiments', 'csv', '-x', exp.experiment_id, '-o',
            f'{mode}_results.csv'
        ])

    if args.train:

        print("----------------")
        training.train(hpar,
                       mode,
                       args.contextspecific,
                       True,
                       copy_checkpoint=Path("models") /
                       f"{mode}_{args.contextspecific}.pt",
                       test=True)

    if args.evaluate:
        evaluate.main(mode)


if __name__ == "__main__":
    main()
