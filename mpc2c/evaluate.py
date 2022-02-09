import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import wilcoxon, f_oneway, ttest_rel, kruskal, shapiro
from statsmodels.stats.multitest import multipletests

import re

OUTPUT_DIR = "imgs"
ip = im = ''


def myplot(title, *args, **kwargs):

    fig = px.box(
        *args,
        **kwargs,
        # box=True,
        title=title,
        points='all')
    fig.update_traces(boxmean='sd')
    return fig


def add_multi_index(df: pd.DataFrame):
    """
    Add a multi-index consisting of the hyper-parameters of the runs
    """

    hyperparams = df[[
        ip + 'enc_k1', ip + 'enc_k2', ip + 'enc_kernel', ip + 'spec_k1',
        ip + 'spec_k2', ip + 'spec_kernel'
    ]].astype(str).apply(lambda x: '_'.join(x), axis=1)
    df['params'] = hyperparams

    multiple_performers = df[ip + 'multiple_performers'].astype('string')
    context_specific = df[ip + 'context_specific'].astype('string')
    context_specific = context_specific.astype('string')
    df['method'] = multiple_performers + '-' + context_specific
    methods = df['method'].unique()

    # removing runs that have not ended
    df = df[df[im + 'perfm_test_avg'].notna()]
    # keep only the 4 most recent runs for each set of params
    # (if one set had a stopped run, it is restarted from scratch)
    df = df.groupby('params').head(4)
    # remove params with less than 4 runs
    params_ok = df.groupby('params')[im + 'perfm_test_avg'].count() == 4
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


def analyze_context_importance(df, methods, var=im + "perfm_test_avg"):
    """
    Analyze the importance of condidering context and not by taking the best
    value for each run among those configurations that consider context and
    comparing them to the configuration that doesn't consider context.
    """

    title = "Test Avg By Context"
    dists = {'context': [], 'no_context': []}
    regex = re.compile(".*True.*")
    for m in methods:
        data = df.loc[m][var]
        if regex.match(m):
            dists['context'].append(data)
        else:
            dists['no_context'].append(data)

    dists['no_context'] = pd.concat(dists['no_context'])

    dists['context'] = pd.concat(dists['context'], axis=1).min(axis=1)

    print("Plotting " + title)

    significance_analysis(dists)

    fig = myplot(title,
                 pd.DataFrame(dists).melt(),
                 x='variable',
                 y='value',
                 color='variable')
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.show()


def analyze_methods(df, methods, var="perfm_test_avg"):
    """
    Analyze each method across the various runs.
    """

    title = "Test Avg By Method"
    distributions_by_method = {m: df.loc[m][var] for m in methods}
    print("Plotting " + title)

    significance_analysis(distributions_by_method)

    fig = myplot(title, df.reset_index(), x='method', y=var, color='method')
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.show()


def analyze_wins(df, methods, var=im + "perfm_test_avg"):
    """
    Analyze how many configuration each method is the best

    Returns a dataframe where rows are methods and cols are beated methods
    """
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


def find_best_method(df,
                     methods,
                     var=im + 'perfm_test_avg',
                     lower_is_better=True):
    print(f"Method         {var:<9} test_avg test_std")
    for method in methods:
        df_method = df.loc[method]
        if lower_is_better:
            i = df_method[var].argmin()
        else:
            i = df_method[var].argmax()
        best_var = df_method.iloc[i][var]
        best_test_avg = df_method.iloc[i][im + 'perfm_test_avg']
        best_test_std = df_method.iloc[i][im + 'perfm_test_std']
        print(
            f"{method:<11}:        {best_var:.2e}  {best_test_avg:.2e}  {best_test_std:.2e}"
        )


def main(mode):
    import sys

    df = pd.read_csv("{mode}_results.csv")

    global ip, im
    if 'enc_k1' in df.columns:
        ip = im = ''
    else:
        ip = 'params.'
        im = 'metrics.'

    if len(sys.argv) == 1:
        var = im + "perfm_test_avg"
    else:
        var = sys.arg[1]

    df, methods, params = add_multi_index(df)

    print("\n==============\n")

    analyze_wins(df, methods, var=var)

    print("\n==============\n")

    analyze_context_importance(df, methods, var=var)

    print("\n==============\n")

    analyze_methods(df, methods, var=var)

    print("\n==============\n")
    find_best_method(df, methods, var=var)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
