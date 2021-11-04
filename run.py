import argparse
import pickle
import shutil
import subprocess
from logging import error
from pathlib import Path

import mlflow
import skopt

from mpc2c import build, create_template, data_management, evaluate
from mpc2c import settings as s
from mpc2c import training
from mpc2c.asmd_resynth import get_contexts, split_resynth
from mpc2c.mytorchutils import hyperopt

build.build()


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for running experiments")
    parser.add_argument(
        "-sc",
        "--scale",
        action="store_true",
        help="Create the midi file containing the scales for the template; syntehsize it and make the template."
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
        help="TODO Perform actions for pedaling estimation (frame-wise prediction)."
    )
    parser.add_argument("-t",
                        "--train",
                        action="store_true",
                        help="Train a model.")
    parser.add_argument(
        "-cs",
        "--contextspecific",
        action="store_true",
        help="Train a specializer against context specificity on the same latent space used for performance regression"
    )

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
        "-pc",
        "--printcontexts",
        action="store_true",
        help="Print contexts in the order with the labels shown in mlflow log")
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
    if args.skopt:

        def objective(x):
            l3 = training.train(x, mode, True, True, test=True)
            l1 = training.train(x, mode, False, False, test=True)
            l2 = training.train(x, mode, True, False, test=True)
            l4 = training.train(x, mode, False, True, test=True)
            return (l1 + l2 + l3 + l4) / 4

        if args.pedaling:
            # test_sample = torch.rand(1, s.BINS, 100)
            checkpoint_path = "ped_skopt.pt"

        elif args.velocity:
            # test_sample = torch.rand(1, s.BINS, s.MINI_SPEC_SIZE)
            checkpoint_path = "vel_skopt.pt"
        else:
            return  # not reachable, here to shutup the pyright

        # space_constraint = training.model_test(
        #     lambda x: training.build_model(x, contexts), test_sample)
        exp = mlflow.get_experiment_by_name(mode)
        if exp:
            if exp.lifecycle_stage == 'deleted':
                exp_path = Path(
                    mlflow.get_registry_uri()) / '.trash' / exp.experiment_id
            else:
                exp_path = exp.artifact_location
            shutil.rmtree(exp_path)

        hyperopt(
            s.SKSPACE,
            checkpoint_path,
            s.SKITERATIONS,
            objective,
            skoptimizer_kwargs=dict(
                # space_constraint=space_constraint,
                plot_graphs=False,
                optimization_method=skopt.dummy_minimize),
            optimize_kwargs=dict(max_loss=20.0,
                                 initial_point_generator="grid"))
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
                       copy_checkpoint=Path("models") / f"{mode}.pt")

    if args.redump:
        contexts = list(get_contexts(s.CARLA_PROJ).keys())
        for split in ['train', 'validation', 'test']:
            data_management.get_loader(split,
                                       redump=True,
                                       contexts=contexts,
                                       one_context_per_batch=False,
                                       mode=mode,
                                       nmf_params=nmf_params)


if __name__ == "__main__":
    main()
