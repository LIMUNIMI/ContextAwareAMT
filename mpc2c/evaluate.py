import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from . import settings as s
from .asmd_resynth import get_contexts
from .data_management import multiple_splits_one_context
from .mytorchutils import make_loss_func, test
from .training import build_pedaling_model, build_velocity_model


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
    """
    import plotly.express as px
    figs = []
    contexts = [context for context in df.columns if context != 'checkpoint']
    checkpoints = df['checkpoint'].unique()

    # plotting all contexts for each checkpoint
    for checkpoint in checkpoints:
        # this is needed because colors are given by a column with labels,
        # otherwise all violins have the same color
        cdf = df[df['checkpoint'] ==
                 checkpoint][contexts].stack().reset_index()
        cdf = cdf.rename({0: 'value', 'level_1': 'context'}, axis='columns')
        figs.append(
            px.violin(
                cdf,
                y='value',
                color='context',
                box=True,
                # points='all',
                range_y=[0, 1],
                title=f"{checkpoint}"))

    # plotting all checkpoints for each context
    for context in contexts:
        figs.append(
            px.violin(
                df[[context, 'checkpoint']],
                y=context,
                # x='checkpoint',
                color='checkpoint',
                box=True,
                # points='all',
                range_y=[0, 1],
                title=f"{context}"))

    if compare:
        # plotting generic vs specific model
        # creating a new dataframe where rows are only kept if the checkpoint
        # string representation starts with the context string representation
        # or with 'orig'.  Deleted values are set to nan. The 'orig' column is
        # also deleted.
        del df['orig']
        contexts = [
            context for context in df.columns if context != 'checkpoint'
        ]
        for checkpoint in checkpoints:
            cols_to_delete = [
                col for col in df.columns if col != 'checkpoint'
                and not checkpoint.startswith((col, 'orig'))
            ]
            df.loc[df['checkpoint'] == checkpoint, cols_to_delete] = None
            # changing name to this checkpoint as it will only be used for its
            # own context
            if len(cols_to_delete) > 0:
                df.loc[df['checkpoint'] == checkpoint,
                       'checkpoint'] = 'transfer-learning'

        fig = px.violin(
            df,
            # x=contexts,
            color='checkpoint',
            box=True,
            # points='all',
            range_y=[0, 1],
            title="transfer-learning effect")

        # adding pvals
        for n, context in enumerate(contexts):
            cdf = df[[context, 'checkpoint']].dropna()
            x = cdf[context]
            y = cdf[cdf['checkpoint'].str.startswith('orig')][context]
            L = min(len(x), len(y))
            _stat, pval = wilcoxon(x[:L], y[:L])
            fig.add_annotation(x=n,
                               y=1,
                               align='center',
                               text=f"p={pval:.4f}",
                               showarrow=False)

        figs.append(fig)

    # TODO: save figs to svgs...

    return figs


def plot_from_file(fname, compare, port):
    df = pd.from_csv(fname)
    figs = plot(df, compare)
    plot_dash(figs, port)
