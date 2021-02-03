import typing as T
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from . import settings as s
from .asmd_resynth import get_contexts
from .data_management import multiple_splits_one_context
from .mytorchutils import make_loss_func, test
from .train import build_pedaling_model, build_velocity_model


def evaluate(checkpoints: T.Dict[str, T.Any], mode: str, fname: str):

    evaluation = []
    contexts: T.List[str] = get_contexts(s.CARLA_PROJ)
    for checkpoint in checkpoints:
        errors = pd.DataFrame()
        if mode == 'velocity':
            model = build_velocity_model(s.VEL_HYPERPARAMS)
        elif mode == 'pedaling':
            model = build_pedaling_model(s.PED_HYPERPARAMS)
        else:
            raise RuntimeError(f"Cannot evaluate mode `{mode}`")

        model.load_state_dict(torch.load(checkpoint)['state_dict'])

        for context in contexts.keys():
            errors[context] = eval_model_context(model, context, mode)
        errors['checkpoint'] = [Path(checkpoint).stem] * errors.shape[0]
        evaluation.append(errors)

    # concatenate dataframes
    evaluation = pd.concat(evaluation)
    evaluation.to_csv(fname)

    return evaluation


def eval_model_context(model: torch.nn.Module, context: str, mode: str):
    testloader = multiple_splits_one_context(['test'], context, mode, False)
    loss, predictions = test(model,
                             testloader,
                             make_loss_func(F.l1_loss),
                             device=s.DEVICE,
                             dtype=s.DTYPE,
                             return_predictions=True)
    if mode == 'velocity':
        # TODO : check the following
        return np.concatenate(predictions)[-1]
    elif mode == 'pedaling':
        # TODO : check the following
        return np.concatenate(predictions)[-1]


def plot_dash(figs, port):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()
    app.layout = html.Div([dcc.Graph(figure=fig) for fig in figs])
    app.run_server(port=port, debug=False, use_reloader=False)


def plot(df: pd.DataFrame, compare: bool):
    """
    `df` is a dataframe with columns the contexts + one column where the
    checkpoint of each result is stored.

    TODO: add p-values
    """
    import plotly.express as px
    figs = []
    # plotting all contexts for each checkpoint
    checkpoints = df['checkpoint'].unique()
    for checkpoint in checkpoints:
        figs.append(
            px.violin(df[df['checkpoint'] == checkpoint],
                      x=[
                          context for context in df.columns
                          if context != 'checkpoint'
            ],
                title=f"{checkpoint}"))

    # plotting all checkpoints for each context
    for context in df.columns:
        if context == 'checkpoint':
            continue
        figs.append(
            px.violin(df[context, 'checkpoint'],
                      x='checkpoint',
                      title=f"{context}"))

    if not compare:
        return figs
    # plotting generic vs specific model
    # creating a new dataframe where rows are only kept if the checkpoint
    # string representation starts with the context string representation or
    # with 'orig'.  Deleted values are set to nan. The 'orig' column is also
    # deleted.
    del df['orig']
    for checkpoint in checkpoints:
        cols_to_delete = [
            col for col in df.columns()
            if col != 'checkpoint' and checkpoint.startswith(col)
        ]
        # TODO check that the following effectively modifies df
        df[df['checkpoint'] == checkpoint][cols_to_delete] = None

    figs.append(
        px.violin(
            df,
            x=[context for context in df.columns if context != 'checkpoint'],
            groups='checkpoint',
            title="transfer-learning effect"))

    return figs


def plot_from_file(fname, compare, port):
    df = pd.from_csv(fname)
    figs = plot(df, compare)
    plot_dash(figs, port)
