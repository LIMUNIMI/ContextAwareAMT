import time
import os
import sys
import traceback
from dataclasses import dataclass
from pprint import pprint
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

import skopt
from skopt import load, plots
from skopt.callbacks import CheckpointSaver, VerboseCallback
import numpy as np

import mlflow
from . import context


def hyperopt(*args, skoptimizer_kwargs={}, optimize_kwargs={}):
    """
    A functional interface to `SKOtimizer`.
    Just pass all the args to `SKOptimizer` and run `optimize`.
    """
    skoptimizer = SKOptimizer(*args, **skoptimizer_kwargs)
    skoptimizer.optimize(**optimize_kwargs)


class EarlyStop(object):
    def __init__(self, n_iters):
        """
        Stops the Optimization if for `n_iters` there are no improvements
        """

        self.n_iters = n_iters
        self.counter = 0

    def __call__(self, res):
        """
        Return True if there was `self.n_iters` without improvements, otherwise
        returns False.
        """
        if res.fun == res.func_vals[-1]:
            # last iteration was the best one
            self.counter = 0
            return False
        elif self.counter >= self.n_iters:
            # n_iters were passed
            return True
        else:
            # n_iters were not passed
            self.counter += 1
            return False


@dataclass
class SKOptimizer(object):
    """
    A class to ease the hyper-parameters search.

    Fields
    ------

    `space` is a skopt space of named hyperparams

    `checkpoint_path` is a path where checkpoints are saved

    `num_iter` is a tuple[int] containing the number of iterations for the two
        procedures: the first is the number of iterations to do using the
        uniform random choice, the second is the number of iterations to do
        using the `optimization_method` specified in this call

    `to_minimize` is a callable that accepts hyperparams in `space` as a dict
        and which returns one loss

    `optimization_method` a callable that implements he skopt interface;
        defaults to `skopt.dummy_minimize`

    `space_constraint` a callable that implements a constraint for the space:
        return True if the hyperparameters are valid

    `plot_graphs` a boolean default to True; if False, no plot is produced

    `early_stop` a tuple representing the number of iterations without
        improvement after which the optimization will be stopped even if
        `num_iter` has not been reached yet; by default, it is set to `num_iter
        * 0.4`. To disable, simply put this equal to `num_iter`.

    Methods
    -------

    `plot` opens a `MLFLow` instance and plots the `self.res` object in this
        instance

    `optimize` load a checkpoint if it exists and starts the optimization
        procedure; calls the `plot` method after checkpoint loading  and before
        of exiting. First, this performs `num_iter[0]` iterations using a
        uniform radnom sampler, then it performs `num_iter[1]` iterations using
        the method specified in the constructor
    """

    space: list
    checkpoint_path: str = 'skopt_checkpoint.pkl'
    num_iter: Tuple[int, int] = (0, 100)
    to_minimize: Callable = lambda x: 1
    optimization_method: Callable = skopt.forest_minimize
    space_constraint: Optional[Callable] = None
    plot_graphs: bool = True
    seed: int = 1992
    early_stop: Tuple[int,
                      int] = (4 * num_iter[0] // 10, 4 * num_iter[1] // 10)

    def _make_objective_func(self, max_loss=1.0):
        global objective

        @skopt.utils.use_named_args(self.space)
        def objective(**hyperparams):

            print("--------------------")
            print("Testing hyperparams:")
            pprint(hyperparams)

            try:
                loss = self.to_minimize(hyperparams)
            except (ValueError, Exception, RuntimeError) as e:
                if context.DEBUG:
                    # the following 2 are for debugging
                    traceback.print_exc(e)
                print("Detected runtime error: ", e, file=sys.stderr)
                print("To view this error, set `context.DEBUG` to False")
                loss = max_loss

            return loss

        return objective

    def _make_constraint(self):
        global constraint

        if self.space_constraint is not None:

            @skopt.utils.use_named_args(self.space)
            def constraint(**hyperparams):
                return self.space_constraint(hyperparams)

            return constraint

        else:
            return None

    def plot(self):
        if not self.plot_graphs:
            return
        print("Plotting a res object, open MLFlow on localhost!")
        # plottings
        fig = plt.figure()
        plots.plot_convergence(self.res)
        mlflow.log_figure(fig, str(int(time.time())) + '.png')

        # the previous method doesn't work here (matplotlib sucks)
        # previous comment was for visdom...
        axes = plots.plot_objective(self.res)
        fig = axes.flatten()[0].figure
        mlflow.log_figure(fig, str(int(time.time())) + '.png')
        axes = plots.plot_evaluations(self.res)
        fig = axes.flatten()[0].figure
        mlflow.log_figure(fig, str(int(time.time())) + '.png')

    def optimize(self, max_loss=1.0, **kwargs):
        self.res = None
        if self.load_res():
            x0 = self.res.x_iters
            y0 = self.res.func_vals
            prev_iters = len(x0)
            random_state = self.res.random_state
        else:
            print("Starting new optimization from scratch...")
            x0 = y0 = None
            prev_iters = 0
            random_state = self.seed

        verbose_callback = VerboseCallback(1)
        checkpoint_saver = CheckpointSaver(self.checkpoint_path)
        if prev_iters < self.num_iter[0]:
            print("\n=================================")
            print("Uniform random init")
            print("=================================\n")
            kwargs_ = dict(
                func=self._make_objective_func(max_loss),
                dimensions=self.space,
                x0=x0,  # already examined values for x
                y0=y0,  # observed values for x0
                callback=[
                    verbose_callback, checkpoint_saver,
                    EarlyStop(self.early_stop[0])
                ],
                random_state=random_state,
                n_calls=self.num_iter[0] - prev_iters,
                **kwargs
            )
            if self.space_constraint is not None:
                kwargs_["space_constraint"]=self._make_constraint()
            self.res = skopt.dummy_minimize(**kwargs_)
            x0 = self.res.x_iters
            y0 = self.res.func_vals
            prev_iters = len(x0)
            random_state = self.res.random_state

        if prev_iters - self.num_iter[0] < self.num_iter[1]:
            print("\n=================================")
            print("Specific method optimization")
            print("=================================\n")
            kwargs_ = dict(
                func=self._make_objective_func(max_loss),
                dimensions=self.space,
                x0=x0,  # already examined values for x
                y0=y0,  # observed values for x0
                callback=[
                    verbose_callback, checkpoint_saver,
                    EarlyStop(self.early_stop[1])
                ],
                random_state=random_state,
                model_queue_size=1,
                n_calls=self.num_iter[1] + self.num_iter[0] - prev_iters,
                **kwargs
            )
            if self.space_constraint is not None:
                kwargs_["space_constraint"]=self._make_constraint()
            self.res = self.optimization_method(**kwargs_)

        if self.res is None:
            print("No iteration!")
            return

        print("\n=================================\n")

        self.plot()

        print("Best hyperparams:")
        print("x:", self.res.x)
        print("f(x):", self.res.fun)

    def load_res(self, result_fname=None):
        """
        Simply loads checkpoint if `result_fname` is None, otherwise it loads
        `result_fname`.

        `self.res` is set.

        Returns True if something is loaded, False otherwise
        """
        def _load(fname):
            self._make_objective_func()
            self._make_constraint()
            print("Loading checkpoint...")
            self.res = load(fname)
            return True

        if result_fname is None:
            if os.path.exists(self.checkpoint_path):
                return _load(self.checkpoint_path)

        elif os.path.exists(result_fname):
            return _load(result_fname)

        return False
