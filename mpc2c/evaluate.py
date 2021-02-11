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


def evaluate(checkpoints: T.Dict[str, T.Any], mode: str,
             out_dir: Path) -> T.List[pd.DataFrame]:
    # TODO: Not yet tested!

    contexts: T.List[str] = get_contexts(Path(s.CARLA_PROJ))
    evaluation: T.List[T.List[pd.DataFrame]]
    if mode == 'velocity':
        evaluation = evaluate_velocity(checkpoints, contexts)
    elif mode == 'pedaling':
        evaluation = evaluate_pedaling(checkpoints, contexts)

    # transpose evaluation to Tuple[List[DataFrame]] and concat the dataframes
    # in each list
    print("Writing to file...")
    out_dir.mkdir(parents=True, exist_ok=True)
    ret: T.List[pd.DataFrame] = []
    for i, eval in enumerate(tuple(zip(*evaluation))):
        eval = pd.concat(eval)
        ret.append(eval)
        eval.to_csv(out_dir / f"{mode}_eval.{i}.csv")

    return ret


def evaluate_velocity(checkpoints: T.Dict[str, T.Any],
                      contexts: T.List[str]) -> T.List[T.List[pd.DataFrame]]:
    evaluation: T.List[T.List[pd.DataFrame]] = []

    for checkpoint in checkpoints:
        errors = [
            pd.DataFrame(),
        ]
        model = build_velocity_model(s.VEL_HYPERPARAMS)

        model.load_state_dict(torch.load(checkpoint)['state_dict'])

        for context in contexts.keys():
            print(f"\nEvaluating {checkpoint} on {context}")
            res = eval_model_context(model, context, 'velocity')
            errors[0] = errors[0].append(
                pd.DataFrame(
                    dict(values=res,
                         context=[context] * res.shape[0],
                         checkpoint=[Path(checkpoint).stem] * res.shape[0])))
        errors[0]['checkpoint'] = [Path(checkpoint).stem] * errors.shape[0]
        evaluation.append(errors)

    return evaluation


def evaluate_pedaling(checkpoints: T.Dict[str, T.Any],
                      contexts: T.List[str]) -> T.List[T.List[pd.DataFrame]]:
    evaluation: T.List[T.List[pd.DataFrame]] = []

    for checkpoint in checkpoints:
        errors = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        model = build_pedaling_model(s.PED_HYPERPARAMS)

        model.load_state_dict(torch.load(checkpoint)['state_dict'])

        for context in contexts.keys():
            print(f"\nEvaluating {checkpoint} on {context}")
            res = eval_model_context(model, context, 'pedaling')
            for i in range(len(errors)):
                errors[i] = errors[i].append(
                    pd.DataFrame(data=dict(values=res[i],
                                           context=[context] * res[i].shape[0],
                                           checkpoint=[Path(checkpoint).stem] *
                                           res[0].shape[0])))

        evaluation.append(errors)

    return evaluation


def eval_model_context(model: torch.nn.Module, context: str, mode: str):

    testloader = multiple_splits_one_context(['test'], context, mode, False)
    loss, predictions = test(model,
                             testloader,
                             make_loss_func(F.l1_loss),
                             device=s.DEVICE,
                             dtype=s.DTYPE,
                             return_predictions=True)

    # computing L1 errors for each prediction
    errors = []
    for i, (inputs, targets, lens) in enumerate(testloader):
        if lens == torch.tensor(False):
            errors.append(torch.abs(targets[0] - predictions[i][..., 0, 0]))
        else:
            for batch, L in enumerate(lens[0]):
                errors.append(
                    torch.abs(targets[0][batch, ..., :L] -
                              predictions[i][0][batch, ..., :L]))

    return np.concatenate(errors, -1)


def plot_dash(figs, port):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()
    app.layout = html.Div([dcc.Graph(figure=fig) for fig in figs])
    app.run_server(port=port, debug=False, use_reloader=False)


def plot(df: pd.DataFrame, compare: bool, save: T.Optional[Path]):
    """
    `df` is a dataframe with columns the 'values', 'context' and 'checkpoint'.

    `compare` to add one more plots with `orig` vs all other context

    `save` is the path to the dir where figs will be saved (if `None`, no
    figure will be saved)
    """
    import plotly.express as px
    figs = []
    contexts = df['context'].unique()
    checkpoints = df['checkpoint'].unique()

    # plotting all contexts for each checkpoint
    print(" 1. Plotting checkpoints")
    for checkpoint in checkpoints:
        # this is needed because colors are given by a column with labels,
        # otherwise all violins have the same color
        figs.append(
            px.violin(
                df[df['checkpoint'] == checkpoint],
                y='values',
                color='context',
                box=True,
                # range_y=[0, 1],
                title=f"{checkpoint}"))

    # plotting all checkpoints for each context
    print(" 2. Plotting contexts")
    for context in contexts:
        figs.append(
            px.violin(
                df[df['context'] == context],
                y='values',
                color='checkpoint',
                box=True,
                # range_y=[0, 1],
                title=f"{context}"))

    if compare:
        print(" 3.1 Preparing orig vs all")
        # plotting generic vs specific model
        # creating a new dataframe where rows are only kept if the checkpoint
        # string representation starts with the context string representation
        # or with 'orig'.

        # removing rows with 'orig' context
        df = df[df['context'] != 'orig']
        contexts = df['context'].unique()

        # selcting rows for which the checkpoint starts with the context or
        # with 'orig'
        cdf = pd.DataFrame()
        cdf = cdf.append(df[df['checkpoint'].str.startswith('orig')])
        for context in contexts:
            cdf = cdf.append(df[df['checkpoint'].str.startswith(context)])

        # renaming checkpoints
        new_checkpoint = 'transfer-learnt'
        idx = cdf['checkpoint'].str.startswith('orig')
        # ~ is for logical not
        cdf.loc[~idx, 'checkpoint'] = new_checkpoint

        # plotting
        print(" 3.2 Plotting orig vs all")
        fig = px.violin(
            cdf,
            y='values',
            x='context',
            color='checkpoint',
            box=True,
            # range_y=[0, 1],
            title="transfer-learning effect")
        __import__('ipdb').set_trace()
        fig.write_image('test.svg')

        # adding pvals
        print(" 3.3 Computing pvals")
        for n, context in enumerate(contexts):
            data = cdf[cdf['context'] == context]
            x = data.loc[data['checkpoint'] == new_checkpoint, 'values']
            y = data.loc[data['checkpoint'] == 'orig', 'values']
            L = min(len(x), len(y))
            _stat, pval = wilcoxon(x[:L], y[:L])
            fig.add_annotation(x=n,
                               y=1,
                               align='center',
                               text=f"p={pval:.4f}",
                               showarrow=False)

        figs.append(fig)

    # saving figures to svg files
    if save:
        print(" 4. Saving figures")
        save.mkdir(parents=True, exist_ok=True)
        for fig in figs:
            fig.write_image(save / fig.layout.title.text.replace(' ', '_') +
                            '.svg')

    return figs


def plot_from_file(fname, compare, port):
    fname = Path(fname)
    print("Reading from file...")
    df = pd.read_csv(fname)
    print("Creating figures...")
    figs = plot(df, compare, save=Path(s.IMAGES_PATH) / fname.stem)
    print("Starting dash...")
    plot_dash(figs, port)