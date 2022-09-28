"""
Organize results.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy import stats

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
    'cbu_ibug_831ff4730ee884556c05a3751389d8fb': 'IBUG-LGB+CBU',
    'cbu_ibug_732b2e458f8d0c10e5d5f534f9acc2fb': 'IBUG-CB+CBU',
    'pgbm_5e13b5ea212546260ba205c54e1d9559': 'PGBM',
    'ngboost': 'NGBoost',
    'knn_8bf9fb55d7e448013b5150404e77a943': 'KNN',
    'knn_001a4a2d54c0e64a8368b4daf1c08aff': 'KNN_n',
    'knn_f4cf753c299e5105a821a0e29f3a2403': 'KNN-LGB',
    'knn_f6d49086cea3bad421507f7084c8c516': 'KNN-LGB_n',
    'knn_5ba9d95f9045b3ea4a0248804c44dbb3': 'KNN-CB',
    'knn_d11684036f81bfaa8f34a3c0a1a46fb9': 'KNN-CB_n',
    'ibug_831ff4730ee884556c05a3751389d8fb': 'IBUG-LGB',
    'ibug_5b54b77b1323676a6ed208e39cf10155': 'IBUG-LGB_n',
    'ibug_732b2e458f8d0c10e5d5f534f9acc2fb': 'IBUG-CB',
    'ibug_897d3256175ee64ce73e931ccf70f4f4': 'IBUG-CB_n',
    'ibug_e01ca97af6a59a305a51f8eaad593980': 'IBUG-XGB',
    'ibug_fc813e5057b0b17774f983502e59797b': 'IBUG-SKRF',
    'bart': 'BART',
    'dataset': 'Dataset',
}

def adjust_sigfigs(v, n=None):
    """
    Adjust number of significant figures.
        - If          v < 1e-3, use scientific notation w/ 0 sigfigs.
        - If 1e-3  <= v < 1, use 3 sigfigs.
        - If 1     <= v < 10, use 3 sigfigs.
        - If 10    <= v < 100, use 1 sigfigs.
        - If 100   <= v < 1e5, use 0 sigfigs.
        - If 1e5   <= v < 1e6, use scientific notation w/ 1 sigfig.
        - If          v >= 1e6, use scientific notation w/ 0 sigfigs.

    Input
        v: float, Input value.
        n: int, Number of significant figures.

    Return
        str, with appropriate sigfigs.
    """
    if type(v) != float:
        v = float(v)
    assert type(v) == float
    if n is not None:
        res = f'{v:.{n}f}'
    else:
        if v < 1e-3:
            res = f'{v:.0e}'
        elif v < 1:
            res = f'{v:.3f}'
        elif v < 10:
            res = f'{v:.3f}'
        elif v < 100:
            res = f'{v:.1f}'
        elif v < 1e5:
            res = f'{v:.0f}'
        elif v < 1e6:
            res = f'{v:.1e}'
        else:
            res = f'{v:.0e}'
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

def plot_runtime_boxplots(btime_df, ptime_df, rotate_labels=False, shorten_names=True,
    out_dir=None, logger=None):
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
    if shorten_names:
        col_names = [c.replace('-LGB', '').replace('-CB', '') for c in col_names]

    _, axs = plt.subplots(1, 2, figsize=(4 * 2, 2.75 * 1))
    axs = axs.flatten()

    ax = axs[0]
    btime_df.boxplot(cols, ax=ax, return_type='dict')
    ax.set_yscale('log')
    ax.set_xticklabels(col_names)
    ax.set_ylabel('Total Train Time (s)')
    ax.grid(b=False, which='major', axis='x')
    ax.set_axisbelow(True)
    if rotate_labels:
        ax.set_xticklabels(col_names, ha='right', rotation=45)

    ax = axs[1]
    ptime_df.boxplot(cols, ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(col_names)
    ax.set_ylabel('Avg. Predict Time (ms)')
    ax.grid(b=False, which='major', axis='x')
    ax.set_axisbelow(True)
    if rotate_labels:
        ax.set_xticklabels(col_names, ha='right', rotation=45)

    if out_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'runtime.pdf'), bbox_inches='tight')


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
    float_opt='t', caption='', label='', is_center=True,
    align_str=None, header_formatting=True):
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
    if not header_formatting:
        s += '\n' + ' & '.join([c for c in df.columns]) + ' \\\\'
    else:
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
    p_val_threshold=0.05, return_significance=False):
    """
    Attach head-to-head wins, ties, and losses to the given dataframe.

    Input
        data_df: pd.DataFrame, Dataframe used to compute head-to-head scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.
        skip_cols: list, List of columns to ignore when computing scores.
        include_ties: bool, If True, include ties with scores.
        p_val_threshold: float, p-value threshold to denote statistical significance.
        return_significance: bool, If True, return significance dictionary.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    assert 'dataset' in data_df.columns
    assert 'fold' in data_df.columns
    cols = [c for c in data_df.columns if c not in ['dataset', 'fold']]

    sig = {}  # stores the wins-ties-losses for each comparison

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
            
            temp_df = data_df[['dataset', c1, c2]].dropna()

            sig_key = f'{c1} {c2}'
            sig[sig_key] = {'wins': [], 'ties': [], 'losses': []}

            n_wins, n_ties, n_losses = 0, 0, 0
            for dataset, df in temp_df.groupby('dataset'):
                t_stat, p_val = ttest_rel(df[c1], df[c2], nan_policy='omit')

                if t_stat < 0 and p_val < p_val_threshold:
                    n_wins += 1
                    sig[sig_key]['wins'].append(dataset)
                elif t_stat > 0 and p_val < p_val_threshold:
                    n_losses += 1
                    sig[sig_key]['losses'].append(dataset)
                else:
                    if np.isnan(p_val):  # no difference in method values
                        print('NAN P_val', dataset, c1, c2)
                    n_ties += 1
                    sig[sig_key]['ties'].append(dataset)

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

    return sig if return_significance else result_df


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
            param_list.append(md)
            param_list.append(result['model_params']['base_model_params_'][min_leaf_name])
            param_names += ['$d$', r'$n_{\ell_0}$']
            param_types += [int, int]

        param_list.append(int(result['model_params']['k']))
        param_list.append(result['model_params']['min_scale'])
        param_names += ['$k$', r'$\rho$']
        param_types += [int, float]
    
    elif 'cbu' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_names += ['$T$', r'$\eta$']
        param_types += [int, float]

        tree_type = result['train_args']['tree_type']
        md = result['model_params']['max_depth']
        md = -1 if md is None else md
        param_list.append(md)
        param_list.append(result['model_params']['min_data_in_leaf'])
        param_names += ['$d$', r'$n_{\ell_0}$']
        param_types += [int, int]


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

    elif 'knn' in name:
        if result['train_args']['tree_type'] in ['lgb', 'cb']:
            param_list.append(result['model_params']['max_feat'])
            param_list.append(result['model_params']['k'])
            param_list.append(result['model_params']['min_scale'])
            param_names += [r'$\upsilon$', '$k_2$', r'$\rho$']
            param_types += [int, int, float]
        else:
            param_list.append(result['model_params']['max_feat'])
            param_list.append(result['model_params']['base_model_params_']['n_neighbors'])
            param_list.append(result['model_params']['k'])
            param_list.append(result['model_params']['min_scale'])
            param_names += [r'$\upsilon$', '$k_1$', '$k_2$', r'$\rho$']
            param_types += [int, int, int, float]

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
                        param_str = adjust_sigfigs(param_val)

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
                test_prob[name] = res['metrics']['test']['scoring_rule'][args.sr_metric]
                test_prob2[name] = res['metrics']['test']['avg_calibration'][args.cal_metric]
                test_prob3[name] = res['metrics']['test']['sharpness']['sharp']
                test_prob_delta[name] = res['metrics']['test_delta']['scoring_rule'][args.sr_metric]
                test_prob2_delta[name] = res['metrics']['test_delta']['avg_calibration'][args.cal_metric]
                test_prob3_delta[name] = res['metrics']['test_delta']['sharpness']['sharp']

                val_prob[name] = res['metrics']['val']['scoring_rule'][args.sr_metric]
                val_prob2[name] = res['metrics']['val']['avg_calibration'][args.cal_metric]
                val_prob3[name] = res['metrics']['val']['sharpness']['sharp']
                val_prob_delta[name] = res['metrics']['val_delta']['scoring_rule'][args.sr_metric]
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
    mean_h2h = {key: append_head2head_old(data_df=raw[key], attach_df=mean[key]) for key in mean.keys()}
    sem_h2h = {key: append_head2head(df) for key, df in sem.items()}
    std_h2h = {key: append_head2head(df) for key, df in std.items()}

    logger.info('\n\nNO VARIANCE CALIBRATION')
    logger.info(f"\n{metric_names[args.acc_metric]} (Accuracy):\n{mean_h2h['test_acc_df']}")
    logger.info(f"\n{metric_names[args.sr_metric]} (Proper Scoring Rule):\n{mean_h2h['test_sr_df']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{mean_h2h['test_cal_df']}")
    logger.info(f"\nSharpness:\n{mean_h2h['test_sharp_df']}")

    logger.info('\n\nWITH VARIANCE CALIBRATION')
    logger.info(f"\n{metric_names[args.acc_metric]} (Accuracy):\n{mean_h2h['test_acc_df']}")
    logger.info(f"\n{metric_names[args.sr_metric]} (Proper Scoring Rule):\n{mean_h2h['test_sr_df_delta']}")
    logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{mean_h2h['test_cal_df_delta']}")
    logger.info(f"\nSharpness:\n{mean_h2h['test_sharp_df_delta']}")

    # delta comparison
    keys = [k for k in mean.keys() if 'delta' not in k and k not in ['test_acc_df', 'btime_df', 'ptime_df']]
    # delta = {key: mean[key].merge(mean[key+'_delta'], on='dataset', how='left') for key in keys}
    # delta_h2h = {key: append_head2head(df) for key, df in delta.items()}

    delta_raw = {key: raw[key].merge(raw[key+'_delta'], on=['dataset', 'fold'], how='left') for key in keys}
    delta_mean = {key: mean[key].merge(mean[key+'_delta'], on='dataset', how='left') for key in keys}
    delta_h2h = {key: append_head2head_old(data_df=df, attach_df=delta_mean[key]) for key, df in delta_raw.items()}

    # logger.info('\n\nW/O VS. WITH VARIANCE CALIBRATION\n')
    # logger.info(f"\n{metric_names[args.sr_metric]} (Proper Scoring Rule):\n{delta_h2h['test_sr_df']}")
    # logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{delta_h2h['test_cal_df']}")

    # val-test comparison
    # keys = [k for k in mean.keys() if not any(x in k for x in ['test', 'test_acc_df', 'btime_df', 'ptime_df'])]
    # vtest_h2h = {k.replace('val', 'vtest'): valtest_h2h(val_df=mean[k], test_df=mean[k.replace('val', 'test')]) for k in keys}

    # val-test comparison (scoring rule only)
    logger.info('\n\nVALIDATION AND TEST')
    val_sig = append_head2head_old(data_df=raw['val_sr_df_delta'], return_significance=True)
    test_sig = append_head2head_old(data_df=raw['test_sr_df_delta'], return_significance=True)
    logger.info('\nW/ Calibration (validation)')
    for k, v in val_sig.items():
        logger.info(f'{k}: {v["wins"]}')
    logger.info('\nW/ Calibration (test)')
    for k, v in test_sig.items():
        logger.info(f'{k}: {v["wins"]}')
    # logger.info('\nW/O Calibration')
    # logger.info(f"\n{metric_names[args.sr_metric]} (Proper Scoring Rule):\n{vtest_h2h['vtest_sr_df']}")
    # logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{vtest_h2h['vtest_cal_df']}")
    # logger.info(f"\nSharpness:\n{vtest_h2h['vtest_sharp_df']}")

    # logger.info('\nW/ Calibration')
    # logger.info(f"\n{metric_names[args.sr_metric]} (Proper Scoring Rule):\n{vtest_h2h['vtest_sr_df_delta']}")
    # logger.info(f"\n{metric_names[args.cal_metric]} (Calibration):\n{vtest_h2h['vtest_cal_df_delta']}")
    # logger.info(f"\nSharpness:\n{vtest_h2h['vtest_sharp_df_delta']}")

    # save
    for key, df in mean_h2h.items():
        df.to_csv(f'{out_dir}/mean_{key}.csv', index=False)

    for key, df in delta_h2h.items():
        df.to_csv(f'{out_dir}/dcomp_{key}.csv', index=False)

    param_df.to_csv(f'{out_dir}/param_df.csv', index=False)

    logger.info(f'\nSaving results to {out_dir}...')

    if 'cbu' in mean['test_cal_df']:
        logger.info('\n\nCBU Calibration Improvements')
        cbu_cal_ratio = mean['test_cal_df']['cbu'] / mean['test_cal_df_delta']['cbu']
        cbu_sharp_ratio = mean['test_sharp_df']['cbu'] / mean['test_sharp_df_delta']['cbu']
        logger.info(f'Median CBU calibration error ratio w/ and w/o delta: {cbu_cal_ratio.median():.3f}')
        logger.info(f'Median CBU sharpness ratio w/ and w/o delta: {cbu_sharp_ratio.median():.3f}')

    # print markdown tables
    logger.info('\n\nMARKDOWN TABLES')
    markdown_table(df=mean_h2h['test_acc_df'], logger=logger, suffix=args.acc_metric.upper())
    markdown_table(df=mean_h2h['test_sr_df_delta'], logger=logger, suffix=args.sr_metric.upper())
    markdown_table2(df1=mean_h2h['test_cal_df_delta'], df2=mean_h2h['test_sharp_df_delta'], logger=logger, suffix='MACE/Sharpness')

    # print latex tables
    logger.info('\n\nLATEX TABLES')
    latex_table_sem(df=mean_h2h['test_acc_df'], sem_df=sem_h2h['test_acc_df'], logger=logger, suffix=args.acc_metric.upper())
    latex_table_sem(df=mean_h2h['test_sr_df'], sem_df=sem_h2h['test_sr_df'], logger=logger,
        suffix=args.sr_metric.upper())
    latex_table_sem(df=mean_h2h['test_sr_df_delta'], sem_df=sem_h2h['test_sr_df_delta'], logger=logger,
        suffix=args.sr_metric.upper() + '+Delta')
    latex_table2(df1=mean_h2h['test_cal_df_delta'], df2=mean_h2h['test_sharp_df_delta'], logger=logger, suffix='MACE/Sharpness')
    latex_table(df=param_df, logger=logger, suffix='Hyperparameters', header_formatting=False)

    # runtime
    logger.info('\n\nRUNTIME')
    logger.info(f'\n Train time (s):\n{append_gmean(mean["btime_df"])}')
    logger.info(f'\n Avg. predict time per test example (s):\n{append_gmean(mean["ptime_df"])}')
    latex_table_sem(df=mean_h2h['btime_df'], sem_df=std_h2h['btime_df'], logger=logger, suffix='Train Time (s)')
    latex_table_sem(df=mean_h2h['ptime_df'], sem_df=std_h2h['ptime_df'], logger=logger, suffix='Avg. Predict Time (ms)')
    plot_runtime_boxplots(btime_df=mean_h2h['btime_df'], ptime_df=mean_h2h['ptime_df'], rotate_labels=args.rotate_labels,
        shorten_names=args.shorten_names, out_dir=out_dir, logger=logger)


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
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--model_type', type=str, nargs='+',
        default=['knn', 'ngboost', 'pgbm', 'cbu', 'bart', 'ibug', 'cbu_ibug'])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--cond_mean_type', type=str, nargs='+', default=['base'])

    parser.add_argument('--tuning_metric', type=str, default='crps')
    parser.add_argument('--acc_metric', type=str, default='rmse')
    parser.add_argument('--sr_metric', type=str, default='crps')
    parser.add_argument('--cal_metric', type=str, default='ma_cal')

    # plot settings
    parser.add_argument('--rotate_labels', action='store_true')
    parser.add_argument('--shorten_names', action='store_true')

    args = parser.parse_args()
    main(args)
