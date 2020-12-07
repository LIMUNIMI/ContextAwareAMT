import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import visdom

import skopt
from skopt import load, plots
from skopt.callbacks import CheckpointSaver, VerboseCallback


def hyperopt(*args):
    """
    A functional interface to `SKOtimizer`.
    Just all the args to `SKOptimizer` and run `optimize`.
    """
    skoptimizer = SKOptimizer(*args)
    skoptimizer.optimize()


@dataclass
class SKOptimizer(object):
    """
    A class to ease the hyper-parameters search.

    Fields
    ------

    `space` is a skopt space of named hyperparams

    `checkpoint_path` is a path where checkpoints are saved

    `num_iter` is the number of iterations

    `to_minimize` is a callable that accepts hyperparams in `space` as a dict
        and which returns one loss

    `optimization_method` a callable that implements he skopt interface;
        defaults to `skopt.gbrt_minimize`

    Methods
    -------

    `plot` opens a `visdom` instance and plots the `self.res` object in this
        instance

    `optimize` load a checkpoint if it exists and starts the optimization
        procedure; calls the `plot` method after checkpoint loading  and before
        of exiting.
    """

    space: dict
    checkpoint_path: str
    num_iter: int
    to_minimize: callable
    optimization_method = skopt.gbrt_minimize

    def _make_objective_func(self):
        @skopt.utils.use_named_args(self.space)
        def objective(**hyperparams):

            print("--------------------")
            print("Testing hyperparams:")
            print(hyperparams)

            try:
                loss = self.to_minimize(hyperparams)
            except (ValueError, Exception, RuntimeError) as e:
                # the following 2 are for debugging
                # import traceback
                # traceback.print_exc(e)
                print("Detected runtime error: ", e)
                loss = 1.0
            return loss
        return objective

    def plot(self):
        print("Plotting a res object, open visdom on localhost!")
        # plottings
        vis = visdom.Visdom()
        fig = plt.figure()
        plots.plot_convergence(self.res)
        vis.matplot(fig)

        # the previous method doesn't work here (matplotlib sucks)
        axes = plots.plot_objective(self.res)
        vis.matplot(axes.flatten()[0].figure)
        axes = plots.plot_evaluations(self.res)
        vis.matplot(axes.flatten()[0].figure)

    def optimize(self):
        if os.path.exists(self.checkpoint_path):
            print("Loading and plotting previous checkpoint...")
            self.res = load(self.checkpoint_path)
            x0 = self.res.x_iters
            y0 = self.res.func_vals
            self.plot()
        else:
            print("Starting new optimization from scratch...")
            x0, y0 = None

        verbose_callback = VerboseCallback(1)
        checkpoint_saver = CheckpointSaver(self.checkpoint_path)
        print("\n=================================\n")
        res = self.optimization_method(
            self._make_objective_func(),
            x0=x0,  # already examined values for x
            y0=y0,  # observed values for x0
            dimensions=self.space,
            callback=[verbose_callback, checkpoint_saver],
            n_calls=self.num_iter)
        skopt.utils.dump(res, "skopt_result.pkl")
        print("\n=================================\n")

        self.plot()
