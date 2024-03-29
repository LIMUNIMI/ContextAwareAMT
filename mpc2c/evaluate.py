import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import wilcoxon, f_oneway, ttest_rel, kruskal, shapiro
from statsmodels.stats.multitest import multipletests

import re

SHOW_HTML = False
OUTPUT_DIR = "imgs"
regex = re.compile(".*True.*")


def is_context_aware(method):
    return regex.match(method)


def myplot(title, *args, **kwargs):

    fig = px.box(
        *args,
        **kwargs,
        # box=True,
        title=title,
        points=False)
    fig.update_traces(boxmean='sd')
    return fig


def add_multi_index(df: pd.DataFrame, initp="", initm=""):
    """
    Add a multi-index consisting of the hyper-parameters of the runs
    """

    hyperparams = df[[
        initp + 'enc_k1', initp + 'enc_k2', initp + 'enc_kernel',
        initp + 'spec_k1', initp + 'spec_k2', initp + 'spec_kernel'
    ]].astype(str).apply(lambda x: '_'.join(x), axis=1)
    df['params'] = hyperparams

    multiple_performers = df[initp + 'multiple_performers'].astype('string')
    context_specific = df[initp + 'context_specific'].astype('string')
    context_specific = context_specific.astype('string')
    df['method'] = multiple_performers + '-' + context_specific
    methods = df['method'].unique()

    # removing runs that have not ended
    df = df[df[initm + 'perfm_test_avg'].notna()]
    # keep only the 4 most recent runs for each set of params
    # (if one set had a stopped run, it is restarted from scratch)
    df = df.groupby('params').head(4)
    # remove params with less than 4 runs
    params_ok = df.groupby('params')[initm + 'perfm_test_avg'].count() == 4
    runs_ok = df['params'].isin(params_ok[params_ok].index)
    df = df[runs_ok]

    df = df.set_index(['method', 'params'])
    print(f"Number of valid runs: {df.shape[0]}")
    return df, methods, hyperparams.unique()


def corrected_pvals(distributions, stat_test=wilcoxon):
    """
    Computes Wilcoxon p-values and correct them. Returns non-corrected p-values
    and rejection result with corrected pvalues.
    """

    L = len(distributions)
    methods = list(distributions.keys())
    pvals = pd.DataFrame(np.zeros((L, L), dtype=np.float64), methods, methods)

    for m1, vals1 in distributions.items():
        for m2, vals2 in distributions.items():
            if m1 == m2:
                pvals[m1].loc[m2] = np.nan
                continue
            _stat, pval = stat_test(vals1, vals2)
            pvals[m1].loc[m2] = pval

    # only keep upper triangular part
    _pvals = np.triu(pvals.to_numpy())
    _pvals[_pvals == 0] = np.nan
    pvals = pd.DataFrame(data=_pvals.copy(),
                         index=pvals.index,
                         columns=pvals.columns)

    # take valid p-values and correct them
    # Correcting pairwise p-values with confidence at 95%!
    idx = np.where(~np.isnan(_pvals))
    _, _corrected, _, _ = multipletests(_pvals[idx].flatten(), method='holm')
    _pvals[idx] = _corrected
    corrected = pd.DataFrame(data=_pvals,
                             index=pvals.index,
                             columns=pvals.columns)

    return corrected, pvals


def significance_analysis(distributions):
    # Normality test
    print("Normality tests:")
    for key, dist in distributions.items():
        _stat, pval = shapiro(dist)
        print(f"{key}: {pval:.2e}")
    print("-------")

    # Multivariate tests
    _stat, f_pval = f_oneway(*distributions.values())
    print(f"One-way ANOVA: {f_pval:.2e}")

    _stat, f_pval = kruskal(*distributions.values())
    print(f"One-way Kruskal: {f_pval:.2e}")

    print("-------")

    # Pair-wise tests
    print("\nT-test p-values:\n")
    reject, pvals = corrected_pvals(distributions, ttest_rel)
    print(pvals)
    print("\nCorrection reject hypothesis:\n")
    print(reject)

    print("\nWilcoxon p-values:\n")
    reject, pvals = corrected_pvals(distributions, wilcoxon)
    print(pvals)
    print("\nCorrection reject hypothesis:\n")
    print(reject)


def analyze_context_importance(dfs, methods, var="perfm_test_avg", initm=""):
    """
    Analyze the importance of considering context and not. It takes the best
    value for each run among those configurations that consider context and
    compares them to the configuration that doesn't consider context.

    `dfs`: List[Tuple[str, pd.DataFrame]], where `str` is the mode.
    """

    var = initm + var
    title = "Test Avg By Context"
    dists = {mode: df['reward'] for mode, df in dfs}
    oracles = {}

    # adding the oracles
    for mode in dists:
        oracles[mode] = []
        for m in methods:
            if is_context_aware(m):
                oracles[mode].append(dists[mode].loc[m])
        # now oracles[mode] contains one array (the results) for each
        # context_aware method
        # I contatenate these results and compute the maximum for each
        # hyperparameter point
        oracles[mode] = pd.concat(oracles[mode], axis=1).max(axis=1)

    print("Plotting " + title)

    print("\nComparison of all the methods across modality")
    significance_analysis(dists)

    print("\nComparison of the best method across modality")
    significance_analysis(oracles)

    fig = myplot(title + " oracles",
                 pd.DataFrame(oracles).melt(),
                 x='variable',
                 y='value',
                 color='variable')
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.write_html(f"imgs/{title.replace(' ', '_')}.html")
    if SHOW_HTML:
        fig.show()


def point_to_point_by_context(dfs, var='perfm_test_avg', initm=""):
    """
    Creates point-to-point plot for each df in `dfs`, but only using the best
    method among the ones that are context-aware
    """
    var = initm + var

    fig = make_subplots(rows=1,
                        cols=len(dfs),
                        shared_yaxes=True,
                        horizontal_spacing=0.01,
                        subplot_titles=[df[0] for df in dfs])

    for idx, (mode, methods, df) in enumerate(dfs):
        unaware_method = [m for m in methods if not is_context_aware(m)][0]
        methods = [m for m in methods if is_context_aware(m)]

        reference = df.loc[unaware_method, var].reset_index()
        reference['color'] = "e''"

        # build the dataframe with the oracle values
        others = df.loc[methods, var].swaplevel()
        oracle_df = reference.copy()
        oracle_df['color'] = "e'"
        for param in reference['params']:
            best_value = others.loc[param].min()
            oracle_df.loc[oracle_df['params'] == param, var] = best_value

        add_point_to_point_subplot(oracle_df, fig, idx, reference, var)

    title = "Error difference by oracles"
    fig.update_layout(title_text=title)
    fig.update_yaxes(tickfont_size=10, dtick=1, tick0=0)
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.write_html(f"imgs/{title.replace(' ', '_')}.html")
    if SHOW_HTML:
        fig.show()


def analyze_methods(df, methods, mode, var="perfm_test_avg", initm=""):
    """
    Analyze each method across the various runs.
    """

    var = initm + var
    title = f"Test Avg By Method ({mode})"
    distributions_by_method = {
        m: df.loc[m][var]
        for m in methods if is_context_aware(m)
    }
    print("Plotting " + title)

    significance_analysis(distributions_by_method)

    fig = myplot(title, df.reset_index(), x='method', y=var, color='method')
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.write_html(f"imgs/{title.replace(' ', '_')}.html")
    if SHOW_HTML:
        fig.show()


def analyze_wins(df, methods, var="perfm_test_avg", initm=""):
    """
    Analyze how many configuration each method is the best

    Returns a dataframe where rows are methods and cols are beated methods
    """
    var = initm + var
    L = len(methods)
    cols = list(methods) + ['all']
    wins = pd.DataFrame(np.zeros((L, L + 1), dtype=np.int32), methods, cols)

    df = df[var].swaplevel()
    confs = df.index.levels[0]
    for conf in confs:
        df_conf = df.loc[conf]
        for m1 in methods:
            for m2 in methods:
                if m1 == m2:
                    wins.loc[m1][m2] = -1
                    continue
                if df_conf.loc[m1] < df_conf[m2]:
                    wins.loc[m1][m2] += 1
                elif df_conf.loc[m2] < df_conf[m1]:
                    pass
                else:
                    print(f"Parity: {conf} - {m1} {m2}!")
            if df_conf.loc[m1] == df_conf.min():
                wins.loc[m1]['all'] += 1

    print("Wins analysis:\n")
    print(wins)
    return wins


def find_best_method(dfs,
                     methods,
                     var='perfm_test_avg',
                     initm="",
                     lower_is_better=True):
    var = initm + var
    print(f"Method         {var:<9} test_avg test_std")
    for method in methods:
        df_method = dfs.loc[method]
        if lower_is_better:
            i = df_method[var].argmin()
        else:
            i = df_method[var].argmax()
        best_var = df_method.iloc[i][var]
        best_test_avg = df_method.iloc[i][initm + 'perfm_test_avg']
        best_test_std = df_method.iloc[i][initm + 'perfm_test_std']
        print(
            f"{method:<11}:        {best_var:.2e}  {best_test_avg:.2e}  {best_test_std:.2e}"
        )


def compute_reward(
    df,
    methods,
    var='perfm_test_avg',
    initm="",
):
    """
    Compute the reward for each method and substitutes it into the dataframe.
    Return the new dataframe modified.

    The method against which the rewqrd will be computed is the one for which
    `is_context_aware()` returns False
    """
    var = initm + var
    new_df = df.copy()
    new_df['reward_' + var] = None
    unaware_method = [m for m in methods if not is_context_aware(m)][0]
    reference = new_df.loc[unaware_method][var]
    for m in methods:
        if m == unaware_method:
            new_df.drop(unaware_method, level=0, axis=0, inplace=True)
        else:
            new_df.loc[m,
                       'reward'] = (reference - new_df.loc[m, var]).to_numpy()
    return new_df


def add_point_to_point_subplot(other, fig, idx, reference, var):
    """
    This function adds a point-to-point subplot to `fig` in column index `idx +
    1`. It takes data from `other` and uses `reference` in each subplot.
    The plotted value is `var` and the subplot title is `title`
    """

    showlegend_neg = True
    showlegend_pos = True
    for i in range(len(other)):
        e2 = reference.iloc[i][var]
        e1 = other.iloc[i][var]
        R = e2 - e1

        hp = reference.iloc[i]['params']
        assert hp == other.iloc[i][
            'params'], "Error, points do not have the same hpeters!"
        hp = hp.split('_')
        hp = ', '.join([hp[1], hp[2], hp[4], hp[5]])
        hp = '[' + hp + ']'

        # point_reward.append(abs(reward))
        if R > 0:
            color = 'blue'
            point_shape = 'circle-open'
            line_dash = None
            name = 'R > 0'
            showlegend = showlegend_pos
            showlegend_pos = False
        else:
            color = 'red'
            point_shape = 'x-open'
            line_dash = 'dash'
            name = 'R ≤ 0'
            showlegend = showlegend_neg
            showlegend_neg = False
        trace = go.Scatter(x=[e1, e2],
                           y=[hp, hp],
                           line=dict(dash=line_dash, width=1, color=color),
                           name=name,
                           marker=dict(symbol=point_shape, line_width=1),
                           legendgroup=name,
                           showlegend=showlegend)
        fig.add_trace(trace, row=1, col=idx + 1)


def point_to_point_by_method(df,
                             methods,
                             mode,
                             var='perfm_test_avg',
                             initm=""):
    """
    Creates a plot with two points for each hyper-parameter combination
    """
    var = initm + var

    unaware_method = [m for m in methods if not is_context_aware(m)][0]
    methods = [m for m in methods if is_context_aware(m)]

    fig = make_subplots(rows=1,
                        cols=len(methods),
                        shared_yaxes=True,
                        horizontal_spacing=0.01,
                        subplot_titles=methods)

    reference = df.loc[unaware_method, var].reset_index()
    reference['color'] = "e''"
    for idx, m in enumerate(methods):
        other = df.loc[m, var].reset_index()
        other['color'] = "e'"
        add_point_to_point_subplot(other, fig, idx, reference, var)
    title = f"Error difference by mode {mode}"
    fig.update_layout(title_text=title)
    fig.update_yaxes(tickfont_size=10, dtick=1, tick0=0)
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.write_html(f"imgs/{title.replace(' ', '_')}.html")
    if SHOW_HTML:
        fig.show()


def __get_inits(df):
    # check if params and metrics are separated with a dot (this depends on
    # the mlflow version)
    if 'enc_k1' in df.columns:
        initp = initm = ''
    else:
        initp = 'params.'
        initm = 'metrics.'
    return initp, initm


def main(metric):

    reward_dfs = []
    metric_dfs = []
    initm = ""
    for mode in ['pedaling', 'velocity']:
        mode_df = pd.read_csv(f"{mode}_results.csv")
        # fixing metric based on the version of MLFlow
        initp, initm = __get_inits(mode_df)
        if metric.startswith(initm):
            metric = metric[len(initm):]
        mode_df, methods, params = add_multi_index(mode_df,
                                                   initp=initp,
                                                   initm=initm)
        mode_df.sort_index(level=['params', 'method'], inplace=True)
        metric_dfs.append((mode, methods, mode_df))
        print("==========================")
        print(f"= Analysis for {mode} =")
        print("==========================")

        print("\n==============\n")
        find_best_method(mode_df, methods, var=metric, initm=initm)

        # print("\n==============\n")
        analyze_wins(mode_df, methods, var=metric, initm=initm)

        point_to_point_by_method(mode_df,
                                 methods,
                                 mode,
                                 var=metric,
                                 initm=initm)

        mode_df = compute_reward(mode_df, methods, var=metric, initm=initm)

        print("\n==============\n")
        analyze_methods(mode_df, methods, mode, var='reward', initm='')

        reward_dfs.append((mode, mode_df))

    print("\n==============\n")
    print("Context importance")

    analyze_context_importance(reward_dfs, methods, var='reward', initm='')
    point_to_point_by_context(metric_dfs, var=metric, initm=initm)
