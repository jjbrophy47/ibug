"""
Organize results.
"""
import os
import sys
import time
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
from scipy import stats
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util

c_dict = {}
c_dict['ames'] = 'Ames~\\cite{de2011ames}'
c_dict['bike'] = 'Bike~\\cite{bike_sharing,Dua:2019}'
c_dict['california'] = 'California~\\cite{pace1997sparse}'
c_dict['communities'] = 'Communities~\\cite{Dua:2019,redmond2002data}'
c_dict['concrete'] = 'Concrete~\\cite{yeh1998modeling,Dua:2019}'
c_dict['energy'] = 'Energy~\\cite{tsanas2012accurate,Dua:2019}'
c_dict['facebook'] = 'Facebook~\\cite{Sing1503:Comment,Dua:2019}'
c_dict['kin8nm'] = 'Kin8nm~\\cite{kin8nm}'
c_dict['life'] = 'Life~\\cite{life}'
c_dict['meps'] = 'MEPS~\\cite{cohen2009medical}'
c_dict['msd'] = 'MSD~\\cite{Bertin-Mahieux2011}'
c_dict['naval'] = 'Naval~\\cite{Dua:2019,coraddu2016machine}'
c_dict['news'] = 'News~\\cite{Dua:2019,fernandes2015proactive}'
c_dict['obesity'] = 'Obesity~\\cite{obesity}'
c_dict['power'] = 'Power~\\cite{Dua:2019,kaya2012local,tufekci2014prediction}'
c_dict['protein'] = 'Protein~\\cite{Dua:2019}'
c_dict['star'] = 'STAR~\\cite{Dua:2019,stock2012introduction}'
c_dict['superconductor'] = 'Superconductor~\\cite{Dua:2019,hamidieh2018data}'
c_dict['synthetic'] = 'Synthetic~\\cite{breiman1996bagging,friedman1991multivariate}'
c_dict['wave'] = 'Wave~\\cite{Dua:2019}'
c_dict['wine'] = 'Wine~\\cite{cortez2009modeling,Dua:2019}'
c_dict['yacht'] = 'Yacht~\\cite{Dua:2019}'

metric_names = {
    'rmse': 'Root-Mean-Squared Error',
    'nll': 'Negative-Log Likelihood',
    'crps': 'Continuous Ranked Probability Score',
    'check': 'Check Score',
    'interval': 'Interval Score',
    'rms_cal': 'Root-Mean-Squared Calibration Error',
    'ma_cal': 'Mean-Absolute Calibration Error',
    'miscal_area': 'Miscalibration Area',
}

dmap = {
    'meps': 'MEPS',
    'msd': 'MSD',
    'star': 'STAR'
}

method_map = {
    'cbu': 'CBU',
    'cbu_ibug_3c6e45799d41bd18a1c73419c9a8cedf': 'IBUG+CBU',
    'cbu_ibug_9924f746db4df90f0d794346ed144bbd': 'IBUG-CB+CBU',
    'pgbm_5e13b5ea212546260ba205c54e1d9559': 'PGBM',
    'ngboost': 'NGBoost',
    'knn': 'KNN',
    'knn_fi': 'KNN-FI',
    'ibug_3c6e45799d41bd18a1c73419c9a8cedf': 'IBUG',
    'ibug_9924f746db4df90f0d794346ed144bbd': 'IBUG-CB',
    'bart': 'BART',
    'dataset': 'Dataset',
}

def adjust_sigfigs(v, n=None):
    """
    Adjust number of significant figures.
        - If v < 1, use 3 sigfigs.
        - If 1 <= v < 10, use 3 sigfigs.
        - If 10 <= v < 100, use 1 sigfigs.
        - If 100 <= v, use 0 sigfigs.

    Input
        v: float, Input value.
        n: int, Number of significant figures.

    Return
        str, with appropriate sigfigs.
    """
    assert type(v) == float
    if n is not None:
        res = f'{v:.{n}f}'
    else:
        if v < 10:
            res = f'{v:.3f}'
        elif v < 100:
            res = f'{v:.1f}'
        else:
            res = f'{v:.0f}'
    return res


def get_ax_lims(ax):
    """
    Compute min. and max. of axis limits.

    Input
        ax: Matplotlib.Axes object.

    Return
        List of 2 items, min. value of both axes, max. values of both axes.
    """
    return [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]

def plot_runtime_boxplots(btime_df, ptime_df, out_dir=None, logger=None):
    """
    Plot runtime boxplots for training and prediction.
    """
    if logger:
        logger.info('\nPlotting boxplots...')
    
    util.plot_settings(fontsize=11, libertine=True)
    
    # sanity checks
    assert 'dataset' in btime_df

    # filtering
    btime_df = btime_df.replace(np.inf, np.nan).dropna()
    ptime_df = ptime_df.replace(np.inf, np.nan).dropna()
    assert np.all(btime_df.index == ptime_df.index)

    # get dimensions
    n_methods = len(btime_df.columns) - 1
    n_datasets = len(btime_df.index) - n_methods

    # remove h2h from dataframes
    btime_df = btime_df.iloc[:n_datasets, 1:].astype(np.float32)
    ptime_df = ptime_df.iloc[:n_datasets, 1:].astype(np.float32)

    # method names
    cols = btime_df.columns.to_list()
    col_names = [method_map[c] for c in cols]

    _, axs = plt.subplots(1, 2, figsize=(4 * 2, 2.75 * 1))
    axs = axs.flatten()

    ax = axs[0]
    btime_df.boxplot(cols, ax=ax, return_type='dict')
    ax.set_yscale('log')
    ax.set_xticklabels(col_names)
    ax.set_ylabel('Total Train Time (s)')
    ax.grid(b=False, which='major', axis='x')
    ax.set_axisbelow(True)

    ax = axs[1]
    ptime_df.boxplot(cols, ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(col_names)
    ax.set_ylabel('Avg. Predict Time (ms)')
    ax.grid(b=False, which='major', axis='x')
    ax.set_axisbelow(True)

    if out_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'runtime.pdf'), bbox_inches='tight')


# def plot_runtime(bdf, pdf, out_dir):
#     """
#     Plot train time and avg. predict time per test example scatter plots.

#     Input
#         bdf: pd.DataFrame, Average build time dataframe.
#         pdf: pd.DataFrame, Average predict time dataframe.
#         out_dir: str, Output directory.
#     """

#     # find IBUG, NGBoost, and PGBM columns
#     igb_col = None
#     ngb_col = None
#     pgb_col = None

#     ibug_count = 0
#     for c in bdf.columns:
#         if 'ibug' in c:
#             ibg_col = c
#             ibug_count += 1
#             assert ibg_col in pdf
#         elif 'ngboost' in c:
#             ngb_col = c
#             assert ngb_col in pdf
#         elif 'pgbm' in c:
#             pgb_col = c
#             assert pgb_col in pdf

#     if (ibg_col is None and ngb_col is None and pgb_col is None) or (ibug_count > 1):
#         return None

#     util.plot_settings(fontsize=15, libertine=True)

#     fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))
#     s = 75

#     ax = axs[0]
#     x = bdf[ibg_col]
#     y = bdf[ngb_col]
#     ax.scatter(x, y, marker='1', s=s)
#     ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(1e0, 1e5)
#     ax.set_ylim(1e0, 1e5)
#     lims = get_ax_lims(ax)
#     ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#     ax.set_xlabel('IBUG')
#     ax.set_ylabel('NGBoost')
#     ax.legend()

#     ax = axs[1]
#     x = bdf[ibg_col]
#     y = bdf[pgb_col]
#     ax.scatter(x, y, marker='1', s=s)
#     ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', s=s)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(1e1, 1e6)
#     ax.set_ylim(1e1, 1e6)
#     lims = get_ax_lims(ax)
#     ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#     ax.set_xlabel('IBUG')
#     ax.set_ylabel('PGBM')

#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f'runtime_train.pdf'))
#     plt.close('all')

#     fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))

#     ax = axs[0]
#     x = pdf[ibg_col]
#     y = pdf[ngb_col]
#     ax.scatter(x, y, marker='1', s=s)
#     ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(1e-2, 1e3)
#     ax.set_ylim(1e-2, 1e3)
#     lims = get_ax_lims(ax)
#     ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#     ax.tick_params(axis='both', which='minor')
#     ax.set_xlabel('IBUG')
#     ax.set_ylabel('NGBoost')

#     ax = axs[1]
#     x = pdf[ibg_col]
#     y = pdf[pgb_col]
#     ax.scatter(x, y, marker='1', s=s)
#     ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(1e-2, 1e3)
#     ax.set_ylim(1e-2, 1e3)
#     lims = get_ax_lims(ax)
#     ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#     ax.tick_params(axis='x', which='minor')
#     ax.set_xlabel('IBUG')
#     ax.set_ylabel('PGBM')

#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f'runtime_predict.pdf'))

#     return (ibg_col, ngb_col, pgb_col)


def compute_time_intersect(bdf, pdf, ref_col, skip_cols=[]):
    """
    Computes no. predictions required for the total runtime of
        two methods to intersect (train+predict).

    Input
        bdf: pd.DataFrame, build time dataframe.
        pdf: pd.DataFrame, predict time dataframe.
        ref_col: str, Column to compute relative values to.
        skip_cols: list, Columns to skip when computing relative values.

    Return
        pd.DataFrame with additional columns equal to the number
        of columns (not including `ref_col` or `skip_cols`).
    """
    assert 'n_train' in bdf and 'n_test' in pdf

    res_df = bdf[['dataset']].copy()
    res_df['n'] = bdf['n_train'] + pdf['n_test']
    res_df['p'] = bdf['n_features']

    cols = [c for c in bdf.columns if c not in skip_cols and c != ref_col]
    for c in cols:
        key = f'{ref_col}-{c}'
        res_df[key] = (bdf[c] - bdf[ref_col]) / (pdf[ref_col] - pdf[c])
        res_df[key] = res_df[key].apply(lambda x: f'{int(x):,}')
    return res_df


def compute_relative(df, ref_col, skip_cols=[]):
    """
    Compute values relative to a specified column.

    Input
        df: pd.DataFrame, Input dataframe.
        ref_col: str, Column to compute relative values to.
        skip_cols: list, Columns to skip when computing relative values.

    Return
        pd.DataFrame with relative values.
    """
    res_df = df.copy()
    cols = [c for c in df.columns if c not in skip_cols]
    for c in cols:
        res_df[c] = df[c] / df[ref_col]
    return res_df


def markdown_table(df, logger=None, suffix=''):
    """
    Format dataframe.
    """
    assert 'dataset' in df

    df = df.replace(np.inf, np.nan).dropna()

    num_cols = [c for c in df.columns if c != 'dataset']

    s = '|' + '|'.join([method_map[c] for c in df.columns]) + '|'
    s += '\n| --- | ' + ' | '.join(['---' for _ in num_cols]) + ' |'
    datasets = df['dataset'].values
    num_arr = df[num_cols].values

    num_datasets = len(datasets) - (len(df.columns) - 1)
    min_vals = num_arr[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr in zip(datasets, num_arr):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = '*' + method_map[items[0]] + ' ' + items[1] + '*'
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n|' + dataset_name.replace("w-l", "W-L")
        for v in arr:
            if type(v) == str:
                s += f'| {v}'
            else:
                if v == min_vals[i]:
                    s += f'| **{adjust_sigfigs(v)}**'
                else:
                    s += f'| {adjust_sigfigs(v)}'
        i += 1
        s += '|'
    
    if logger:
        logger.info(f'\n### {suffix}\n{s}\n')
    else:
        print(f'\n### {suffix}\n{s}\n')


def markdown_table_sem(df, sem_df, logger=None, suffix=''):
    """
    Format dataframe.
    """
    assert 'dataset' in df

    df = df.replace(np.inf, np.nan).dropna()
    sem_df = sem_df.replace(np.inf, np.nan).dropna()
    assert np.all(df.index == sem_df.index)

    num_cols = [c for c in df.columns if c != 'dataset']

    s = '|' + '|'.join([method_map[c] for c in df.columns]) + '|'
    s += '\n| --- | ' + ' | '.join(['---' for _ in num_cols]) + ' |'
    datasets = df['dataset'].values
    num_arr = df[num_cols].values
    sem_arr = sem_df[num_cols].values

    num_datasets = len(datasets) - (len(df.columns) - 1)
    min_vals = num_arr[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr, s_arr in zip(datasets, num_arr, sem_arr):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = '*' + method_map[items[0]] + ' ' + items[1] + '*'
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n|' + dataset_name.replace("w-l", "W-L")
        for v, se in zip(arr, s_arr):
            if type(v) == str:
                s += f'| {v}'
            else:
                if v == min_vals[i]:
                    s += f'| **$${adjust_sigfigs(v)}**' + '_{(' + f'{adjust_sigfigs(se)}' + ')}$$'
                else:
                    s += f'| $${adjust_sigfigs(v)}' + '_{(' + f'{adjust_sigfigs(se)}' + ')}$$'
        i += 1
        s += '|'
    
    if logger:
        logger.info(f'\n### {suffix}\n{s}\n')
    else:
        print(f'\n### {suffix}\n{s}\n')


def markdown_table2(df1, df2, logger=None, suffix=''):
    """
    Format dataframe, combining the columns from two dataframes into one column.
    """
    assert 'dataset' in df1

    df1 = df1.replace(np.inf, np.nan).dropna()
    df2 = df2.replace(np.inf, np.nan).dropna()
    assert np.all(df1.index == df2.index)

    num_cols = [c for c in df1.columns if c != 'dataset']

    s = '|' + '|'.join([method_map[c] for c in df1.columns]) + '|'
    s += '\n| --- | ' + ' | '.join(['---' for _ in num_cols]) + ' |'
    datasets = df1['dataset'].values
    num_arr1 = df1[num_cols].values
    num_arr2 = df2[num_cols].values

    num_datasets = len(datasets) - (len(df1.columns) - 1)
    min_vals1 = num_arr1[:num_datasets].min(axis=1)
    min_vals2 = num_arr2[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr1, arr2 in zip(datasets, num_arr1, num_arr2):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = '*' + method_map[items[0]] + ' ' + items[1] + '*'
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n|' + dataset_name.replace("w-l", "W-L")
        for v1, v2 in zip(arr1, arr2):
            if type(v1) == str:
                s += f'| {v1}/{v2}'
            else:
                if v1 == min_vals1[i]:
                    s += f'| **{adjust_sigfigs(v1)}**'
                else:
                    s += f'| {adjust_sigfigs(v1)}'

                if v2 == min_vals2[i]:
                    s += f'/**{adjust_sigfigs(v2)}**'
                else:
                    s += f'/{adjust_sigfigs(v2)}'
        i += 1
        s += '|'
    
    if logger:
        logger.info(f'\n### {suffix}\n{s}\n')
    else:
        print(f'\n### {suffix}\n{s}\n')


def latex_table(df, logger=None, suffix='',
    float_opt='t', caption='', label='', is_center=True, align_str=None):
    """
    Format dataframe.
    """
    assert 'dataset' in df
    df = df.replace(np.inf, np.nan).dropna()
    num_cols = [c for c in df.columns if c != 'dataset']

    # preamble
    s = f'\\begin{{table}}[{float_opt}]'
    s += f'\n\\caption{{{caption}}}'
    s += f'\n\\label{{{label}}}' 
    if is_center:
        s += '\n\\centering'
    s += '\n\\begin{tabular}'
    if align_str:
        s += f'{{{align_str}}}'
    else:
        s += '{l' + 'c' * (len(num_cols) - 1) + 'c}'
    s += '\n\\toprule'

    # column headers
    s += '\n' + ' & '.join([method_map[c] for c in df.columns]) + ' \\\\'
    s += '\n\\midrule'

    # table body
    datasets = df['dataset'].values
    num_arr = df[num_cols].values

    num_datasets = len(datasets) - (len(df.columns) - 1)
    min_vals = num_arr[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr in zip(datasets, num_arr):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = method_map[items[0]] + ' ' + items[1].replace('w-l', 'W-L')
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n' + dataset_name
        for v in arr:
            if type(v) == str:
                s += f' & {v}'
            else:
                if v == min_vals[i]:
                    s += f' & {{\\bfseries {adjust_sigfigs(v)}}}'
                else:
                    s += f' & {adjust_sigfigs(v)}'
        i += 1
        s += ' \\\\'
    
    # bottom
    s += '\n\\bottomrule'
    s += '\n\\end{tabular}'
    s += '\n\\end{table}'
    
    if logger:
        logger.info(f'\n{suffix}\n{s}\n')
    else:
        print(f'\n{suffix}\n{s}\n')


def latex_table_sem(df, sem_df, logger=None, suffix='', float_opt='t', caption='',
    label='', is_center=True, align_str=None):
    """
    Format dataframe.
    """
    assert 'dataset' in df
    df = df.replace(np.inf, np.nan).dropna()
    sem_df = sem_df.replace(np.inf, np.nan).dropna()
    assert np.all(df.index == sem_df.index)

    num_cols = [c for c in df.columns if c != 'dataset']

    # preamble
    s = f'\\begin{{table}}[{float_opt}]'
    s += f'\n\\caption{{{caption}}}'
    s += f'\n\\label{{{label}}}' 
    if is_center:
        s += '\n\\centering'
    s += '\n\\begin{tabular}'
    if align_str:
        s += f'{{{align_str}}}'
    else:
        s += '{l' + 'c' * (len(num_cols) - 1) + 'c}'
    s += '\n\\toprule'

    # column headers
    s += '\n' + ' & '.join([method_map[c] for c in df.columns]) + ' \\\\'
    s += '\n\\midrule'

    # table body
    datasets = df['dataset'].values
    num_arr = df[num_cols].values
    sem_arr = sem_df[num_cols].values

    num_datasets = len(datasets) - (len(df.columns) - 1)
    min_vals = num_arr[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr, s_arr in zip(datasets, num_arr, sem_arr):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = method_map[items[0]] + ' ' + items[1].replace('w-l', 'W-L')
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n' + dataset_name
        for v, se in zip(arr, s_arr):
            if type(v) == str:
                s += f' & {v}'
            else:
                if v == min_vals[i] or v - se <= min_vals[i]:
                    s += f' & {{\\bfseries {adjust_sigfigs(v)}}}'
                else:
                    s += f' & {adjust_sigfigs(v)}'
                s += f'$_{{({adjust_sigfigs(se)})}}$'
        i += 1
        s += ' \\\\'
    
    # bottom
    s += '\n\\bottomrule'
    s += '\n\\end{tabular}'
    s += '\n\\end{table}'
    
    if logger:
        logger.info(f'\n{suffix}\n{s}\n')
    else:
        print(f'\n{suffix}\n{s}\n')


def latex_table2(df1, df2, logger=None, suffix='', float_opt='t', caption='',
    label='', is_center=True, align_str=None):
    """
    Format dataframe.
    """
    assert 'dataset' in df1

    df1 = df1.replace(np.inf, np.nan).dropna()
    df2 = df2.replace(np.inf, np.nan).dropna()
    assert np.all(df1.index == df2.index)

    num_cols = [c for c in df1.columns if c != 'dataset']

    # preamble
    s = f'\\begin{{table}}[{float_opt}]'
    s += f'\n\\caption{{{caption}}}'
    s += f'\n\\label{{{label}}}' 
    if is_center:
        s += '\n\\centering'
    s += '\n\\begin{tabular}'
    if align_str:
        s += f'{{{align_str}}}'
    else:
        s += '{l' + 'c' * (len(num_cols) - 1) + 'c}'
    s += '\n\\toprule'

    # column headers
    s += '\n' + ' & '.join([method_map[c] for c in df1.columns]) + ' \\\\'
    s += '\n\\midrule'

    # table body
    datasets = df1['dataset'].values
    num_arr1 = df1[num_cols].values
    num_arr2 = df2[num_cols].values

    num_datasets = len(datasets) - (len(df1.columns) - 1)
    min_vals1 = num_arr1[:num_datasets].min(axis=1)
    min_vals2 = num_arr2[:num_datasets].min(axis=1)

    i = 0
    for dataset, arr1, arr2 in zip(datasets, num_arr1,  num_arr2):
        if dataset in dmap:
            dataset_name = dmap[dataset]
        elif dataset.split()[0] in method_map:
            items = dataset.split()
            dataset_name = method_map[items[0]] + ' ' + items[1].replace('w-l', 'W-L')
        else:
            dataset_name = f'{dataset.capitalize()}'

        s += '\n' + dataset_name
        for v1, v2 in zip(arr1, arr2):
            if type(v1) == str:
                s += f' & {v1}/{v2}'
            else:
                if v1 == min_vals1[i]:
                    s += f' & {{\\bfseries {adjust_sigfigs(v1)}}}'
                else:
                    s += f' & {adjust_sigfigs(v1)}'

                if v2 == min_vals2[i]:
                    s += f'/{{\\bfseries {adjust_sigfigs(v2)}}}'
                else:
                    s += f'/{adjust_sigfigs(v2)}'
        i += 1
        s += ' \\\\'
    
    # bottom
    s += '\n\\bottomrule'
    s += '\n\\end{tabular}'
    s += '\n\\end{table}'
    
    if logger:
        logger.info(f'\n{suffix}\n{s}\n')
    else:
        print(f'\n{suffix}\n{s}\n')


def append_gmean(data_df, attach_df=None, fmt='int', remove_nan=True):
    """
    Attach geometric mean to the given dataframe.

    Input
        data_df: pd.DataFrame, Dataframe used to compute head-to-head scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.
        fmt: str, Format for the resulting gmean.
        remove_nan: bool, Remove NaN rows in `data_df` before processing.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    res = {'dataset': 'Geo. mean'}

    if remove_nan:
        data_df = data_df.dropna()

    for c in data_df:

        if c in ['dataset', 'fold']:
            continue

        res[c] = stats.gmean(data_df[c])
    gmean_df = pd.DataFrame([res])

    if attach_df is not None:
        result_df = pd.concat([attach_df, gmean_df])
    else:
        result_df = pd.concat([data_df, gmean_df])

    return result_df


def valtest_h2h(val_df, test_df, attach_df=None):
    """
    Attach head-to-head wins, ties, and losses to the given dataframe.

    Input
        val_df: pd.DataFrame, Dataframe with validation scores.
        test_df: pd.DataFrame, Dataframe with test scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    assert 'dataset' in val_df.columns
    assert 'dataset' in test_df.columns
    cols = [c for c in val_df.columns if c not in ['dataset']]

    # fill in NaNs with a large number
    val_df = val_df.fillna(np.inf)
    test_df = test_df.fillna(np.inf)

    res_list = []
    for c1 in cols:
        res = {'dataset': f'{c1} W-L'}

        for c2 in cols:
            if c1 == c2:
                res[c2] = '-'
                continue

            vc1 = val_df[c1] != np.inf
            vc2 = val_df[c2] != np.inf
            tc1 = test_df[c1] != np.inf
            tc2 = test_df[c2] != np.inf

            # wins
            vcw = val_df[c1] < val_df[c2]
            tcw = test_df[c1] < test_df[c2]
            n_wins = np.where((vc1) & (vc2) & (vcw) & (tc1) & (tc2) & (tcw))[0].shape[0]

            # losses
            vcl = val_df[c1] > val_df[c2]
            tcl = test_df[c1] > test_df[c2]
            n_losses = np.where((vc1) & (vc2) & (vcl) & (tc1) & (tc2) & (tcl))[0].shape[0]

            res[c2] = f'{n_wins}-{n_losses}'

        res_list.append(res)
    res_df = pd.DataFrame(res_list)

    if attach_df is not None:
        result_df = pd.concat([attach_df, res_df])
    else:
        result_df = pd.concat([test_df, res_df])
    return result_df


def append_head2head(data_df, attach_df=None):
    """
    Attach head-to-head wins, ties, and losses to the given dataframe.

    Input
        data_df: pd.DataFrame, Dataframe used to compute head-to-head scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    assert 'dataset' in data_df.columns
    cols = [c for c in data_df.columns if c not in ['dataset']]

    # fill in NaNs with a large number
    data_df = data_df.fillna(np.inf)

    res_list = []
    for c1 in cols:
        res = {'dataset': f'{c1} W-L'}

        for c2 in cols:
            if c1 == c2:
                res[c2] = '-'
                continue

            n_wins = len(np.where((data_df[c1] != np.inf) & (data_df[c2] != np.inf) & (data_df[c1] < data_df[c2]))[0])
            n_ties = len(np.where((data_df[c1] == data_df[c2]) | (data_df[c1] == np.inf) | (data_df[c2] == np.inf))[0])
            n_losses = len(np.where((data_df[c1] != np.inf) & (data_df[c2] != np.inf) & (data_df[c1] > data_df[c2]))[0])

            res[c2] = f'{n_wins}-{n_losses}'

        res_list.append(res)
    res_df = pd.DataFrame(res_list)

    if attach_df is not None:
        result_df = pd.concat([attach_df, res_df])
    else:
        result_df = pd.concat([data_df, res_df])
    return result_df


def append_head2head_old(data_df, attach_df=None, include_ties=True,
                     p_val_threshold=0.05):
    """
    Attach head-to-head wins, ties, and losses to the given dataframe.

    Input
        data_df: pd.DataFrame, Dataframe used to compute head-to-head scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.
        skip_cols: list, List of columns to ignore when computing scores.
        include_ties: bool, If True, include ties with scores.
        p_val_threshold: float, p-value threshold to denote statistical significance.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    assert 'dataset' in data_df.columns
    assert 'fold' in data_df.columns
    cols = [c for c in data_df.columns if c not in ['dataset', 'fold']]

    res_list = []
    for c1 in cols:

        if include_ties:
            res = {'dataset': f'{c1} W-T-L'}
        else:
            res = {'dataset': f'{c1} W-L'}

        for c2 in cols:
            if c1 == c2:
                res[c1] = '-'
                continue

            n_wins, n_ties, n_losses = 0, 0, 0
            for dataset, df in data_df.groupby('dataset'):
                t_stat, p_val = ttest_rel(df[c1], df[c2], nan_policy='omit')
                c1_mean = np.mean(df[c1])
                c2_mean = np.mean(df[c2])
                if t_stat < 0 and p_val < p_val_threshold:
                    n_wins += 1
                elif t_stat > 0 and p_val < p_val_threshold:
                    n_losses += 1
                else:
                    if np.isnan(p_val):  # no difference in method values
                        print('NAN P_val', dataset, c1, c2)
                    n_ties += 1

            if include_ties:
                res[c2] = f'{n_wins}-{n_ties}-{n_losses}'
            else:
                res[c2] = f'{n_wins}-{n_losses}'

        res_list.append(res)
    h2h_df = pd.DataFrame(res_list)

    if attach_df is not None:
        result_df = pd.concat([attach_df, h2h_df])
    else:
        result_df = pd.concat([data_df, h2h_df])

    return result_df


def get_param_list(name, result):
    """
    Extract parameters selected to obtain this result.

    Input
        name: str, method name.
        res: dict, result dictionary.

    Return
        - List of selected hyperparameter values.
        - List of selected hyperparameter names.
        - List of selected hyperparameter types.
    """
    param_list = []
    param_names = []
    param_types = []

    if name.startswith('cbu_ibug'):
        return [], [], []

    if 'ibug' in name:
        param_list.append(result['model_params']['base_model_params_']['n_estimators'])
        param_list.append(result['model_params']['base_model_params_']['learning_rate'])
        param_names += ['$T$', r'$\eta$']
        param_types += [int, float]

        if result['train_args']['tree_type'] == 'lgb':
            param_list.append(result['model_params']['base_model_params_']['num_leaves'])
            param_list.append(result['model_params']['base_model_params_']['min_child_samples'])
            param_names += ['$h$', r'$n_{\ell_0}$']
            param_types += [int, int]

        elif result['train_args']['tree_type'] in ['xgb', 'cb']:
            tree_type = result['train_args']['tree_type']
            md = result['model_params']['base_model_params_']['max_depth']
            md = -1 if md is None else md
            min_leaf_name = 'min_child_weight' if tree_type == 'xgb' else 'min_data_in_leaf'
            param_list.append(result['model_params']['base_model_params_'][min_leaf_name])
            param_list.append(md)
            param_names += ['$d$', r'$n_{\ell_0}$']
            param_types += [int, int]

        param_list.append(int(result['model_params']['k']))
        param_list.append(result['model_params']['min_scale'])
        param_names += ['$k$', r'$\rho$']
        param_types += [int, float]

    elif 'constant' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_names += ['$T$', r'$\eta$']
        param_types += [int, float]

        if result['train_args']['tree_type'] == 'lgb':
            param_list.append(result['model_params']['num_leaves'])
            param_list.append(result['model_params']['min_child_samples'])
            param_names += ['$h$', r'$n_{\ell_0}$']
            param_types += [int, int]

        elif result['train_args']['tree_type'] in ['xgb', 'cb']:
            tree_type = result['train_args']['tree_type']
            md = result['model_params']['max_depth']
            md = -1 if md is None else md
            min_leaf_name = 'min_child_weight' if tree_type == 'xgb' else 'min_data_in_leaf'
            param_list.append(md)
            param_list.append(result['model_params']['base_model_params_'][min_leaf_name])
            param_names += ['$d$', r'$n_{\ell_0}$']
            param_types += [int, int]

    elif 'knn_fi' in name:
        param_list.append(result['model_params']['max_feat'])
        param_list.append(result['model_params']['k'])
        param_list.append(result['model_params']['min_scale'])
        param_names += [r'$\upsilon$', '$k$', r'$\rho$']
        param_types += [int, int, float]

    elif 'knn' in name:
        param_list.append(result['model_params']['base_model_params_']['n_neighbors'])
        param_list.append(result['model_params']['k'])
        param_list.append(result['model_params']['min_scale'])
        param_names += ['$k_1$', '$k_2$', r'$\rho$']
        param_types += [int, int, float]

    elif 'ngboost' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_names += ['$T$']
        param_types += [int]

    elif 'pgbm' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_list.append(result['model_params']['max_leaves'])
        param_list.append(result['model_params']['min_data_in_leaf'])
        param_names += ['$T$', r'$\eta$', '$h$', r'$n_{\ell_0}$']
        param_types += [int, float, int, int]

    # delta
    d_val = result['delta']['best_delta']
    d_op = r'$\delta$' if result['delta']['best_op'] == 'add' else r'$\gamma$'
    param_list.append(f'{d_op}:{d_val:.0e}')
    param_names += [r'$\gamma/\delta$']
    param_types += [str]

    return param_list, param_names, param_types


def aggregate_params(param_df, param_names, param_types):
    """
    Combine parameters from multiple folds into one list.

    Input
        param_df: pd.DataFrame, Dataframe with parameters for each fold.
        param_names: dict, List of parameter names, index by method name.
        param_types: dict, List of parameter types, index by method name.

    Return
        Dataframe with combined parameter values.
    """
    param_list = []

    res_list = []
    for dataset, df in param_df.groupby(['dataset']):
        res = {'dataset': dataset}

        for name in df.columns:

            # skip columns
            if name in ['dataset', 'fold']:
                continue

            # error checking
            values_list = [x for x in df[name].values if isinstance(x, list)]
            if len(values_list) > 0:
                values = np.vstack(values_list)

                for i in range(values.shape[1]):  # per parameter
                    param_name = param_names[name][i]
                    param_type = param_types[name][i]
                    param_vals = [param_type(x) for x in values[:, i]]

                    if param_name == r'$\rho$':
                        param_val = np.median(param_vals)
                        if dataset in ['ames', 'news', 'star', 'wave']:
                            param_str = f'{param_val:.0f}'.rstrip('0').rstrip('.')
                        elif dataset == 'naval':
                            param_str = f'{param_val:.0e}'
                        else:
                            param_str = f'{param_val:.3f}'.rstrip('0').rstrip('.')

                    elif param_name == '$T$' and 'NGBoost' in name:
                        param_val = int(np.median(param_vals))
                        param_str = f'{param_val}'

                    elif param_type == float:
                        param_val = stats.mode(param_vals)[0][0]
                        param_str = f'{param_val:.3f}'.rstrip('0').rstrip('.')

                    else:
                        param_val = stats.mode(param_vals)[0][0]
                        param_str = f'{param_val}'

                    res[f'{name}_{param_name}'] = param_str

        res_list.append(res)
    param_df = pd.DataFrame(res_list)

    return param_df


def process(args, in_dir, out_dir, logger):
    # color, ls, label = util.get_plot_dicts()
    # scoring2 = 'crps' if args.scoring == 'nll' else 'nll'

    # get results
    test_point_list = []
    test_prob_list = []
    test_prob2_list = []
    test_prob3_list = []
    test_prob_delta_list = []
    test_prob2_delta_list = []
    test_prob3_delta_list = []

    val_prob_list = []
    val_prob2_list = []
    val_prob3_list = []
    val_prob_delta_list = []
    val_prob2_delta_list = []
    val_prob3_delta_list = []

    btime_list = []
    ptime_list = []
    param_list = []
    param_names = {}
    param_types = {}

    for dataset in args.dataset:
        start = time.time()

        for fold in args.fold:
            exp_dir = os.path.join(in_dir, dataset, args.tuning_metric, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)

            test_point = {'dataset': dataset, 'fold': fold}

            test_prob = test_point.copy()
            test_prob2 = test_point.copy()
            test_prob3 = test_point.copy()
            test_prob_delta = test_point.copy()
            test_prob2_delta = test_point.copy()
            test_prob3_delta = test_point.copy()

            val_prob = test_point.copy()
            val_prob2 = test_point.copy()
            val_prob3 = test_point.copy()
            val_prob_delta = test_point.copy()
            val_prob2_delta = test_point.copy()
            val_prob2_delta = test_point.copy()
            val_prob3_delta = test_point.copy()

            param = test_point.copy()
            btime = test_point.copy()
            ptime = test_point.copy()

            # extract results
            for method, res in results:
                name = method

                test_point[name] = res['metrics']['test']['accuracy'][args.acc_metric]
                test_prob[name] = res['metrics']['test']['scoring_rule'][args.scoring_rule_metric]
                test_prob2[name] = res['metrics']['test']['avg_calibration'][args.cal_metric]
                test_prob3[name] = res['metrics']['test']['sharpness']['sharp']
                test_prob_delta[name] = res['metrics']['test_delta']['scoring_rule'][args.scoring_rule_metric]
                test_prob2_delta[name] = res['metrics']['test_delta']['avg_calibration'][args.cal_metric]
                test_prob3_delta[name] = res['metrics']['test_delta']['sharpness']['sharp']

                val_prob[name] = res['metrics']['val']['scoring_rule'][args.scoring_rule_metric]
                val_prob2[name] = res['metrics']['val']['avg_calibration'][args.cal_metric]
                val_prob3[name] = res['metrics']['val']['sharpness']['sharp']
                val_prob_delta[name] = res['metrics']['val_delta']['scoring_rule'][args.scoring_rule_metric]
                val_prob2_delta[name] = res['metrics']['val_delta']['avg_calibration'][args.cal_metric]
                val_prob3_delta[name] = res['metrics']['val_delta']['sharpness']['sharp']

                btime[name] = res['timing']['tune_train']
                ptime[name] = res['timing']['test_pred_time'] / res['data']['n_test'] * 1000 # milliseconds
                param[name], param_names[name], param_types[name] = get_param_list(name, res)

            test_point_list.append(test_point)
            test_prob_list.append(test_prob)
            test_prob2_list.append(test_prob2)
            test_prob3_list.append(test_prob3)
            test_prob_delta_list.append(test_prob_delta)
            test_prob2_delta_list.append(test_prob2_delta)
            test_prob3_delta_list.append(test_prob3_delta)

            val_prob_list.append(val_prob)
            val_prob2_list.append(val_prob2)
            val_prob3_list.append(val_prob3)
            val_prob_delta_list.append(val_prob_delta)
            val_prob2_delta_list.append(val_prob2_delta)
            val_prob3_delta_list.append(val_prob3_delta)

            btime_list.append(btime)
            ptime_list.append(ptime)
            param_list.append(param)

        logger.info(f'{dataset}...{time.time() - start:.3f}s')

    # compile results
    raw = {}
    raw['test_acc_df'] = pd.DataFrame(test_point_list)
    raw['test_sr_df'] = pd.DataFrame(test_prob_list)
    raw['test_cal_df'] = pd.DataFrame(test_prob2_list)
    raw['test_sharp_df'] = pd.DataFrame(test_prob3_list)
    raw['test_sr_df_delta'] = pd.DataFrame(test_prob_delta_list)
    raw['test_cal_df_delta'] = pd.DataFrame(test_prob2_delta_list)
    raw['test_sharp_df_delta'] = pd.DataFrame(test_prob3_delta_list)

    raw['val_sr_df'] = pd.DataFrame(val_prob_list)
    raw['val_cal_df'] = pd.DataFrame(val_prob2_list)
    raw['val_sharp_df'] = pd.DataFrame(val_prob3_list)
    raw['val_sr_df_delta'] = pd.DataFrame(val_prob_delta_list)
    raw['val_cal_df_delta'] = pd.DataFrame(val_prob2_delta_list)
    raw['val_sharp_df_delta'] = pd.DataFrame(val_prob3_delta_list)

    raw['btime_df'] = pd.DataFrame(btime_list)
    raw['ptime_df'] = pd.DataFrame(ptime_list)
    param_df = aggregate_params(pd.DataFrame(param_list), param_names, param_types)  # aggregate hyperparameters

    # compute mean and and SEM/s.d.
    group_cols = ['dataset']
    mean = {key: df.groupby(group_cols).mean().reset_index().drop(columns=['fold']) for key, df in raw.items()}
    sem = {key: df.groupby(group_cols).sem().reset_index().drop(columns=['fold']) for key, df in raw.items()}
    std = {
        'btime_df': raw['btime_df'].groupby(group_cols).std().reset_index().drop(columns=['fold']),
        'ptime_df': raw['ptime_df'].groupby(group_cols).std().reset_index().drop(columns=['fold'])
    }

    # compute wins/losses
    mean_h2h = {key: append_head2head(df) for key, df in mean.items()}
    sem_h2h = {key: append_head2head(df) for key, df in sem.items()}
    std_h2h = {key: append_head2head(df) for key, df in std.items()}

    # pd.set_option('display.max_columns', None)

    logger.info('\n\nNO VARIANCE CALIBRATION')
    logger.info(f"\n{metric_names[args.acc_metric]} (Accuracy):\n{mean_h2h['test_acc_df']}")
    logger.info(f"\n{metric_names[args.scoring_rule_metric]} (Proper Scoring Rule):\n{mean_h2h['test_sr_df']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{mean_h2h['test_cal_df']}")
    logger.info(f"\nSharpness:\n{mean_h2h['test_sharp_df']}")

    logger.info('\n\nWITH VARIANCE CALIBRATION')
    logger.info(f"\n{metric_names[args.acc_metric]} (Accuracy):\n{mean_h2h['test_acc_df']}")
    logger.info(f"\n{metric_names[args.scoring_rule_metric]} (Proper Scoring Rule):\n{mean_h2h['test_sr_df_delta']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{mean_h2h['test_cal_df_delta']}")
    logger.info(f"\nSharpness:\n{mean_h2h['test_sharp_df_delta']}")

    # delta comparison
    keys = [k for k in mean.keys() if 'delta' not in k and k not in ['test_acc_df', 'btime_df', 'ptime_df']]
    delta = {key: mean[key].merge(mean[key+'_delta'], on='dataset', how='left') for key in keys}
    delta_h2h = {key: append_head2head(df) for key, df in delta.items()}

    logger.info('\n\nW/O VS. WITH VARIANCE CALIBRATION\n')
    logger.info(f"\n{metric_names[args.scoring_rule_metric]} (Proper Scoring Rule):\n{delta_h2h['test_sr_df']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{delta_h2h['test_cal_df']}")

    # val-test comparison
    keys = [k for k in mean.keys() if not any(x in k for x in ['test', 'test_acc_df', 'btime_df', 'ptime_df'])]
    vtest_h2h = {k.replace('val', 'vtest'): valtest_h2h(val_df=mean[k], test_df=mean[k.replace('val', 'test')]) for k in keys}

    logger.info('\n\nVALIDATION AND TEST\n')
    logger.info('\nW/O Calibration')
    logger.info(f"\n{metric_names[args.scoring_rule_metric]} (Proper Scoring Rule):\n{vtest_h2h['vtest_sr_df']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{vtest_h2h['vtest_cal_df']}")
    logger.info(f"\nSharpness:\n{vtest_h2h['vtest_sharp_df']}")

    logger.info('\nW/ Calibration')
    logger.info(f"\n{metric_names[args.scoring_rule_metric]} (Proper Scoring Rule):\n{vtest_h2h['vtest_sr_df_delta']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{vtest_h2h['vtest_cal_df_delta']}")
    logger.info(f"\nSharpness:\n{vtest_h2h['vtest_sharp_df_delta']}")

    # save
    for key, df in mean_h2h.items():
        df.to_csv(f'{out_dir}/mean_{key}.csv', index=False)

    for key, df in delta_h2h.items():
        df.to_csv(f'{out_dir}/dcomp_{key}.csv', index=False)

    for key, df in vtest_h2h.items():
        df.to_csv(f'{out_dir}/vtest_{key}.csv', index=False)

    logger.info(f'\nSaving results to {out_dir}...')

    if 'cbu' in mean['test_cal_df']:
        cbu_cal_ratio = mean['test_cal_df']['cbu'] / mean['test_cal_df_delta']['cbu']
        cbu_sharp_ratio = mean['test_sharp_df']['cbu'] / mean['test_sharp_df_delta']['cbu']
        print(cbu_cal_ratio.mean(), cbu_cal_ratio.median())
        print(cbu_sharp_ratio.mean(), cbu_sharp_ratio.median())

    # print markdown tables
    logger.info('\n\nMARKDOWN TABLES')
    markdown_table(df=mean_h2h['test_acc_df'], logger=logger, suffix=args.acc_metric.upper())
    markdown_table(df=mean_h2h['test_sr_df_delta'], logger=logger, suffix=args.scoring_rule_metric.upper())
    markdown_table2(df1=mean_h2h['test_cal_df_delta'], df2=mean_h2h['test_sharp_df_delta'], logger=logger, suffix='MACE/Sharpness')

    # print latex tables
    logger.info('\n\nLATEX TABLES')
    latex_table_sem(df=mean_h2h['test_acc_df'], sem_df=sem_h2h['test_acc_df'], logger=logger, suffix=args.acc_metric.upper())
    latex_table_sem(df=mean_h2h['test_sr_df'], sem_df=sem_h2h['test_sr_df'], logger=logger,
        suffix=args.scoring_rule_metric.upper())
    latex_table_sem(df=mean_h2h['test_sr_df_delta'], sem_df=sem_h2h['test_sr_df_delta'], logger=logger,
        suffix=args.scoring_rule_metric.upper() + '+Delta')
    latex_table2(df1=mean_h2h['test_cal_df_delta'], df2=mean_h2h['test_sharp_df_delta'], logger=logger, suffix='MACE/Sharpness')

    param_df.to_csv(f'{out_dir}/param_df.csv', index=False)

    # runtime
    logger.info('\n\nRUNTIME')
    logger.info(f'\n Train time (s):\n{append_gmean(mean["btime_df"])}')
    logger.info(f'\n Avg. predict time per test example (s):\n{append_gmean(mean["ptime_df"])}')
    latex_table_sem(df=mean_h2h['btime_df'], sem_df=std_h2h['btime_df'], logger=logger, suffix='Train Time (s)')
    latex_table_sem(df=mean_h2h['ptime_df'], sem_df=std_h2h['ptime_df'], logger=logger, suffix='Avg. Predict Time (ms)')
    plot_runtime_boxplots(btime_df=mean_h2h['btime_df'], ptime_df=mean_h2h['ptime_df'], out_dir=out_dir, logger=logger)

    exit(0)

    # test_point_mean_df = test_point_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # test_prob_mean_df = test_prob_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # test_prob2_mean_df = test_prob2_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # test_prob_delta_mean_df = test_prob_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # test_prob2_delta_mean_df = test_prob2_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # val_prob_mean_df = val_prob_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # val_prob2_mean_df = val_prob2_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # val_prob_delta_mean_df = val_prob_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # val_prob2_delta_mean_df = val_prob2_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # btime_mean_df = btime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    # ptime_mean_df = ptime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])

    # test_point_sem_df = test_point_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # test_prob_sem_df = test_prob_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # test_prob2_sem_df = test_prob2_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # test_prob_delta_sem_df = test_prob_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # test_prob2_delta_sem_df = test_prob2_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # val_prob_sem_df = val_prob_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # val_prob2_sem_df = val_prob2_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # val_prob_delta_sem_df = val_prob_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # val_prob2_delta_sem_df = val_prob2_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    # btime_std_df = btime_df.groupby(group_cols).std().reset_index().drop(columns=['fold'])
    # ptime_std_df = ptime_df.groupby(group_cols).std().reset_index().drop(columns=['fold'])

    # combine mean and sem into one dataframe
    test_point_ms_df = join_mean_sem(test_point_mean_df, test_point_sem_df, metric=args.point_metric)
    test_prob_ms_df = join_mean_sem(test_prob_mean_df, test_prob_sem_df, metric=args.prob_metric1)
    test_prob2_ms_df = join_mean_sem(test_prob2_mean_df, test_prob2_sem_df, metric=args.prob_metric2)
    test_prob_delta_ms_df = join_mean_sem(test_prob_delta_mean_df, test_prob_delta_sem_df, metric=args.prob_metric1)
    test_prob2_delta_ms_df = join_mean_sem(test_prob2_delta_mean_df, test_prob2_delta_sem_df, metric=args.prob_metric2)
    val_prob_ms_df = join_mean_sem(val_prob_mean_df, val_prob_sem_df, metric=args.prob_metric1)
    val_prob2_ms_df = join_mean_sem(val_prob2_mean_df, val_prob2_sem_df, metric=args.prob_metric2)
    val_prob_delta_ms_df = join_mean_sem(val_prob_delta_mean_df, val_prob_delta_sem_df, metric=args.prob_metric1)
    val_prob2_delta_ms_df = join_mean_sem(val_prob2_delta_mean_df, val_prob2_delta_sem_df, metric=args.prob_metric2)
    btime_ms_df = join_mean_sem(btime_mean_df, btime_std_df, metric='btime', exclude_sem=True)
    ptime_ms_df = join_mean_sem(ptime_mean_df, ptime_std_df, metric='ptime', exclude_sem=True)

    # attach head-to-head scores
    test_point_ms2_df = append_head2head(data_df=test_point_df, attach_df=test_point_ms_df)
    test_prob_ms2_df = append_head2head(data_df=test_prob_df, attach_df=test_prob_ms_df)
    test_prob2_ms2_df = append_head2head(data_df=test_prob2_df, attach_df=test_prob2_ms_df)
    test_prob_delta_ms2_df = append_head2head(data_df=test_prob_delta_df, attach_df=test_prob_delta_ms_df)
    test_prob2_delta_ms2_df = append_head2head(data_df=test_prob2_delta_df, attach_df=test_prob2_delta_ms_df)
    val_prob_ms2_df = append_head2head(data_df=val_prob_df, attach_df=val_prob_ms_df)
    val_prob2_ms2_df = append_head2head(data_df=val_prob2_df, attach_df=val_prob2_ms_df)
    val_prob_delta_ms2_df = append_head2head(data_df=val_prob_delta_df, attach_df=val_prob_delta_ms_df)
    val_prob2_delta_ms2_df = append_head2head(data_df=val_prob2_delta_df, attach_df=val_prob2_delta_ms_df)

    # attach g. mean scores
    btime_ms2_df = append_gmean(data_df=btime_df, attach_df=btime_ms_df, fmt='int')
    ptime_ms2_df = append_gmean(data_df=ptime_df, attach_df=ptime_ms_df, fmt='float')

    # format columns
    test_point_ms2_df = format_dataset_names(test_point_ms2_df)
    test_prob_ms2_df = format_dataset_names(test_prob_ms2_df)
    test_prob2_ms2_df = format_dataset_names(test_prob2_ms2_df)
    test_prob_delta_ms2_df = format_dataset_names(test_prob_delta_ms2_df)
    test_prob2_delta_ms2_df = format_dataset_names(test_prob2_delta_ms2_df)
    val_prob_ms2_df = format_dataset_names(val_prob_ms2_df)
    val_prob2_ms2_df = format_dataset_names(val_prob2_ms2_df)
    val_prob_delta_ms2_df = format_dataset_names(val_prob_delta_ms2_df)
    val_prob2_delta_ms2_df = format_dataset_names(val_prob2_delta_ms2_df)
    btime_ms2_df = format_dataset_names(btime_ms2_df)
    ptime_ms2_df = format_dataset_names(ptime_ms2_df)

    # merge specific dataframes
    test_prob_point_delta_ms2_df = test_prob_delta_ms2_df.merge(test_point_ms2_df, on='dataset')
    bptime_ms2_df = btime_ms2_df.merge(ptime_ms2_df, on='dataset')

    # display
    logger.info(f'\n[TEST] Point peformance (RMSE):\n{test_point_ms2_df}')
    logger.info(f'\n[TEST] Probabilistic peformance ({args.scoring.upper()}):\n{test_prob_ms2_df}')
    logger.info(f'\n[TEST] Probabilistic peformance ({args.scoring.upper()}, w/ delta):\n{test_prob_delta_ms2_df}')
    logger.info(f'\n[VAL] Probabilistic peformance ({args.scoring.upper()}):\n{val_prob_ms2_df}')
    logger.info(f'\n[VAL] Probabilistic peformance ({args.scoring.upper()}, w/ delta):\n{val_prob_delta_ms2_df}')
    logger.info(f'\n[TEST] Build time:\n{btime_ms2_df}')
    logger.info(f'\n[TEST] Avg. predict time:\n{ptime_ms2_df}')
    logger.info(f'\nParams:\n{param_df}')

    # save
    logger.info(f'\nSaving results to {out_dir}...')

    test_point_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.point_metric}_str.csv'), index=None)
    test_prob_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric1}_str.csv'), index=None)
    test_prob2_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric2}_str.csv'), index=None)
    test_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric1}_delta_str.csv'), index=None)
    test_prob2_delta_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric2}_delta_str.csv'), index=None)
    val_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.prob_metric1}_str.csv'), index=None)
    val_prob2_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.prob_metric2}_str.csv'), index=None)
    val_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.prob_metric1}_delta_str.csv'), index=None)
    val_prob2_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.prob_metric2}_delta_str.csv'), index=None)
    btime_ms2_df.to_csv(os.path.join(out_dir, 'test_btime_str.csv'), index=None)
    ptime_ms2_df.to_csv(os.path.join(out_dir, 'test_ptime_str.csv'), index=None)

    test_prob_point_delta_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric1}_rmse_str.csv'), index=None)
    bptime_ms2_df.to_csv(os.path.join(out_dir, f'test_bptime_str.csv'), index=None)
    param_df.to_csv(os.path.join(out_dir, 'param.csv'), index=None)

    # delta comparison
    test_prob_dboth_df = test_prob_df.merge(test_prob_delta_df, on=['dataset', 'fold'], how='left')
    test_prob_mean_dboth_df = test_prob_mean_df.merge(test_prob_delta_mean_df, on='dataset', how='left')
    test_prob_sem_dboth_df = test_prob_sem_df.merge(test_prob_delta_sem_df, on='dataset', how='left')
    test_prob_dboth_ms_df = join_mean_sem(test_prob_mean_dboth_df, test_prob_sem_dboth_df, metric=args.prob_metric1)
    test_prob_dboth_ms2_df = append_head2head(data_df=test_prob_dboth_df, attach_df=test_prob_dboth_ms_df)
    test_prob_dboth_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric1}_dcomp.csv'), index=None)

    # delta comparison (scoring 2)
    test_prob2_dboth_df = test_prob2_df.merge(test_prob2_delta_df, on=['dataset', 'fold'], how='left')
    test_prob2_mean_dboth_df = test_prob2_mean_df.merge(test_prob2_delta_mean_df, on='dataset', how='left')
    test_prob2_sem_dboth_df = test_prob2_sem_df.merge(test_prob2_delta_sem_df, on='dataset', how='left')
    test_prob2_dboth_ms_df = join_mean_sem(test_prob2_mean_dboth_df, test_prob2_sem_dboth_df, metric=args.prob_metric2)
    test_prob2_dboth_ms2_df = append_head2head(data_df=test_prob2_dboth_df, attach_df=test_prob2_dboth_ms_df)
    test_prob2_dboth_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.prob_metric2}_dcomp.csv'), index=None)

    # runtime
    logger.info('\nRuntime...')
    ref_cols = plot_runtime(bdf=btime_mean_df, pdf=ptime_mean_df, out_dir=out_dir)

    # # boxplot
    # logger.info('\nPlotting boxplots...')
    # fig, axs = plt.subplots(5, 5, figsize=(4 * 5, 3 * 5))
    # axs = axs.flatten()
    # i = 0
    # for dataset, gf in test_prob_delta_df.groupby('dataset'):
    #     gf.boxplot([c for c in gf.columns if c not in ['dataset', 'fold']], ax=axs[i])
    #     axs[i].set_title(dataset)
    #     axs[i].set_xticks([])
    #     i += 1
    # fig.delaxes(axs[-1])
    # fig.delaxes(axs[-2])
    # fig.delaxes(axs[-3])
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'boxplot.pdf'), bbox_inches='tight')

    # # Wilcoxon ranked test
    # if ref_cols is not None:
    #     ibg_col, ngb_col, pgb_col = ref_cols
    #     logger.info('\nWilocoxon signed rank test (two-tailed):')
    #     for c in [ngb_col, pgb_col]:
    #         test_prob_delta_df = test_prob_delta_df.dropna()  # TEMP
    #         test_prob_delta_mean_df = test_prob_delta_mean_df.dropna()  # TEMP
    #         statistic, p_val = stats.wilcoxon(test_prob_delta_df[ibg_col], test_prob_delta_df[c])
    #         statistic_m, p_val_m = stats.wilcoxon(test_prob_delta_mean_df[ibg_col], test_prob_delta_mean_df[c])
    #         logger.info(f'[{ibg_col} & {c}] ALL FOLDS p-val: {p_val}, MEAN of FOLDS p-val: {p_val_m}')


def main(args):

    in_dir = os.path.join(args.in_dir, args.custom_in_dir)
    out_dir = os.path.join(args.out_dir, args.custom_out_dir, args.tuning_metric)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    exp_util.clear_dir(out_dir)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, in_dir, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/talapas/experiments/predict/')
    parser.add_argument('--out_dir', type=str, default='output/talapas/postprocess/predict/')
    parser.add_argument('--custom_in_dir', type=str, default='default')
    parser.add_argument('--custom_out_dir', type=str, default='default')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model_type', type=str, nargs='+',
        default=['knn', 'knn_fi', 'ngboost', 'pgbm', 'cbu', 'bart', 'ibug', 'cbu_ibug'])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--tuning_metric', type=str, default='crps')
    parser.add_argument('--acc_metric', type=str, default='rmse')
    parser.add_argument('--scoring_rule_metric', type=str, default='crps')
    parser.add_argument('--cal_metric', type=str, default='ma_cal')

    args = parser.parse_args()
    main(args)
