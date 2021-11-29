import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import wilcoxon, f_oneway
from statsmodels.stats.multitest import multipletests

import re

OUTPUT_DIR = "imgs"


def myviolin(title, *args, **kwargs):
    return px.violin(*args, **kwargs, box=True, title=title, points='all')


def add_multi_index(df: pd.DataFrame):
    """
    Add a multi-index consisting of the hyper-parameters of the runs
    """

    hyperparams = df[[
        'enc_k1', 'enc_k2', 'enc_kernel', 'spec_k1', 'spec_k2', 'spec_kernel'
    ]].astype(str).apply(lambda x: '_'.join(x), axis=1)
    df['params'] = hyperparams

    multiple_performers = df['multiple_performers'].astype('string')
    context_specific = df['context_specific'].astype('string')
    df['method'] = multiple_performers + '-' + context_specific
    methods = df['method'].unique()

    df = df.set_index(['method', 'params'])
    return df, methods, hyperparams.unique()


def corrected_pvals(distributions):
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
            _stat, pval = wilcoxon(vals1, vals2)
            pvals[m1].loc[m2] = pval
    _pvals = pvals.to_numpy()
    reject, _, _, _ = multipletests(_pvals[_pvals != 0])

    return reject, pvals


def significance_analysis(distributions):
    _stat, f_pval = f_oneway(*distributions.values())

    reject, pvals = corrected_pvals(distributions)

    print(f"One-way ANOVA: {f_pval:.2e}")
    print("Wilcoxon p-values:\n")
    print(pvals)
    print(f"\nCorrection reject hypothesis: {reject}")


def analyze_context_importance(df, methods, var="perfm_test_avg"):
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

    fig = myviolin(title,
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

    fig = myviolin(title, df.reset_index(), x='method', y=var, color='method')
    fig.write_image(f"imgs/{title.replace(' ', '_')}.svg")
    fig.show()


def analyze_wins(df, methods, var="perfm_test_avg"):
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
                    print(f"Parity: {m1} {m2}!")
            if df_conf.loc[m1] == df_conf.min():
                wins.loc[m1]['all'] += 1

    print("Wins analysis:\n")
    print(wins)
    return wins


def main()
    import sys
    if len(sys.argv) == 1:
        var = "perfm_test_avg"
    else:
        var = sys.arg[1]

    df = pd.read_csv("velocity_results_1.back.csv")
    df = df.loc[2:]  # TODO: remove this row!

    df, methods, params = add_multi_index(df)

    print("\n==============\n")

    analyze_wins(df, methods, var=var)

    print("\n==============\n")

    analyze_context_importance(df, methods, var=var)

    print("\n==============\n")

    analyze_methods(df, methods, var=var)


if __name__ == "__main__":
    main()

