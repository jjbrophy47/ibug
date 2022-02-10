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


n_dict = {}
n_dict['ames'] = 2930
n_dict['bike'] = 17379
n_dict['california'] = 20640
n_dict['communities'] = 1994
n_dict['concrete'] = 1030
n_dict['energy'] = 768
n_dict['facebook'] = 40949
n_dict['kin8nm'] = 8192
n_dict['life'] = 2928
n_dict['meps'] = 15656
n_dict['msd'] = 515345
n_dict['naval'] = 11934
n_dict['news'] = 39644
n_dict['obesity'] = 48346
n_dict['power'] = 9568
n_dict['protein'] = 45730
n_dict['star'] = 2161
n_dict['superconductor'] = 21263
n_dict['synthetic'] = 10000
n_dict['wave'] = 287999
n_dict['wine'] = 6497
n_dict['yacht'] = 308

p_dict = {}
p_dict['ames'] = 358
p_dict['bike'] = 37
p_dict['california'] = 100
p_dict['communities'] = 100
p_dict['concrete'] = 8
p_dict['energy'] = 16
p_dict['facebook'] = 133
p_dict['kin8nm'] = 8
p_dict['life'] = 204
p_dict['meps'] = 139
p_dict['msd'] = 90
p_dict['naval'] = 17
p_dict['news'] = 58
p_dict['obesity'] = 100
p_dict['power'] = 4
p_dict['protein'] = 9
p_dict['star'] = 95
p_dict['superconductor'] = 82
p_dict['synthetic'] = 100
p_dict['wave'] = 48
p_dict['wine'] = 11
p_dict['yacht'] = 6

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


def plot_runtime(bdf, pdf, out_dir):
    """
    Plot train time and avg. predict time per test example scatter plots.

    Input
        bdf: pd.DataFrame, Average build time dataframe.
        pdf: pd.DataFrame, Average predict time dataframe.
        out_dir: str, Output directory.
    """
    util.plot_settings(fontsize=15, libertine=True)

    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))
    s = 75

    ax = axs[0]
    x = bdf['KGBM-LDG']
    y = bdf['NGBoost-D']
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e0, 1e5)
    ax.set_ylim(1e0, 1e5)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel('IBUG')
    ax.set_ylabel('NGBoost')
    ax.legend()

    ax = axs[1]
    x = bdf['KGBM-LDG']
    y = bdf['PGBM-DG']
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e0, 1e5)
    ax.set_ylim(1e0, 1e5)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel('IBUG')
    ax.set_ylabel('PGBM')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'runtime_train.pdf'))
    plt.close('all')

    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))

    ax = axs[0]
    x = pdf['KGBM-LDG']
    y = pdf['NGBoost-D']
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-5, 1e0)
    ax.set_ylim(1e-5, 1e0)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.tick_params(axis='both', which='minor')
    ax.set_xlabel('IBUG')
    ax.set_ylabel('NGBoost')

    ax = axs[1]
    x = pdf['KGBM-LDG']
    y = pdf['PGBM-DG']
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-5, 1e0)
    ax.set_ylim(1e-5, 1e0)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.tick_params(axis='x', which='minor')
    ax.set_xlabel('IBUG')
    ax.set_ylabel('PGBM')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'runtime_predict.pdf'))


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


def format_cols(df, format_dataset_names=False,
                dmap={'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}):
    """
    Format columns.

    Input
        df: pd.DataFrame, Dataframe rename columns.

    Return
        Dataframe with the newly formatted columns.
    """
    if format_dataset_names:
        df['dataset'] = df['dataset'].apply(lambda x: dmap[x] if x in dmap else x.capitalize())
    return df


def join_mean_sem(mean_df, sem_df, metric, mask_cols=['dataset'],
                  exclude_sem=False):
    """
    Joins floats from two dataframes into one dataframe of strings.

    Input
        mean_df: pd.DataFrame, dataframe with mean values.
        sem_df: pd.DataFrame, dataframe with SEM values.
        metric: str, metric to use for deciding formatting.
        mask_cols: list, Columns to not join but included in the result.
        exclude_sem: bool, If True, do not concatenate dataframes.

    Return
        pd.DataFrame object with all string columns.
    """
    float_cols = [c for c in mean_df.columns if c not in mask_cols]
    for float_col in float_cols:
        assert float_col in mean_df and float_col in sem_df

    init_dataset_list = mean_df['dataset'].tolist()

    # get min-value indices
    min_val_dict = {}  # schema={[dataset]: [list of min. indicies; e.g., [1, 3]]}

    mean_vals = mean_df[float_cols].values  # shape=(n_dataset, n_method)
    sem_vals = sem_df[float_cols].values  # shape=(n_dataset, n_method)
    mean_sem_vals = mean_vals - sem_vals  # shape=(n_dataset, n_method)

    min_vals = np.min(mean_vals, axis=1)  # shape=(n_dataset,)

    for dataset, mean_sem_row, min_val in list(zip(init_dataset_list, mean_sem_vals, min_vals)):
        min_idxs = np.where(mean_sem_row <= min_val)[0]
        min_val_dict[dataset] = min_idxs

    if metric == 'crps':
        formats = [(['ames', 'news', 'wave'], '{:.0f}'.format),
                   (['star'], '{:.2f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'facebook', 'kin8nm', 'life', 'meps', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'synthetic', 'wine', 'yacht'], '{:.3f}'.format)]
    elif metric == 'nll':
        formats = [(['ames', 'news', 'wave'], '{:.2f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'facebook', 'kin8nm', 'life', 'meps', 'msd',
                     'naval', 'obesity', 'power', 'protein', 'star',
                     'superconductor', 'synthetic', 'wine', 'yacht'], '{:.3f}'.format)]
    elif metric == 'rmse':
        formats = [(['ames', 'news', 'wave'], '{:.0f}'.format),
                   (['facebook', 'meps', 'synthetic'], '{:.2f}'.format),
                   (['star'], '{:.1f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'kin8nm', 'life', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'wine', 'yacht'], '{:.3f}'.format)]

    elif metric == 'btime':
        formats = [(['ames', 'news', 'wave'], '{:.0f}'.format),
                   (['facebook', 'meps', 'synthetic'], '{:.0f}'.format),
                   (['star'], '{:.0f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'kin8nm', 'life', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'wine', 'yacht'], '{:.0f}'.format)]

    elif metric == 'ptime':
        formats = [(['ames', 'news', 'wave'], '{:.1e}'.format),
                   (['facebook', 'meps', 'synthetic'], '{:.1e}'.format),
                   (['star'], '{:.1e}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'kin8nm', 'life', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'wine', 'yacht'], '{:.1e}'.format)]

    else:
        raise ValueError(f'Unknown metric {metric}')

    # format rows differently
    m_list = []
    s_list = []
    dataset_list = []
    min_idxs_list = []

    for datasets, fmt in formats:
        datasets = [d for d in datasets if d in init_dataset_list]

        if len(datasets) == 0:
            continue

        idxs = mean_df[mean_df['dataset'].isin(datasets)].index

        m_df = mean_df.loc[idxs][float_cols].applymap(fmt).astype(str)
        m_list.append(m_df)

        s_df = sem_df.loc[idxs][float_cols].applymap(fmt).astype(str)
        s_list.append(s_df)

        for d in datasets:
            min_idxs_list.append(min_val_dict[d])
        dataset_list += datasets

    temp1_df = pd.concat(m_list)
    temp2_df = pd.concat(s_list)

    if exclude_sem:
        result_df = temp1_df
    else:
        result_df = temp1_df + '$_{(' + temp2_df + ')}$'

    for i, min_idxs in enumerate(min_idxs_list):  # wrap values with the min. val in bf series
        for j in min_idxs:
            result_df.iloc[i, j] = '{\\bfseries ' + result_df.iloc[i, j] + '}'
    result_df.insert(loc=0, column='dataset', value=dataset_list)
    result_df = result_df.sort_values('dataset')

    return result_df


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
    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    res_list = []
    for dataset, df in param_df.groupby(['dataset']):
        dataset_name = dataset.capitalize() if dataset not in dataset_map else dataset_map[dataset]
        res = {'dataset': dataset_name}

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


def append_gmean(data_df, attach_df=None, fmt='int'):
    """
    Attach geometric mean to the given dataframe.

    Input
        data_df: pd.DataFrame, Dataframe used to compute head-to-head scores.
        attach_df: pd.DataFrame, Dataframe to attach results to.
        fmt: str, Format for the resulting gmean.

    Return
        Dataframe attached to `attach_df` if not None, otherwise scores
            are appended to `data_df`.
    """
    res = {'dataset': 'Geo. mean'}

    for c in data_df:
        if c == 'dataset':
            continue
        res[c] = stats.gmean(data_df[c])
        if fmt == 'int':
            res[c] = f'{int(res[c])}'
        elif fmt == 'sci':
            res[c] = f'{res[c]:.1e}'
        else:
            res[c] = f'{res[c]:.3f}'
    gmean_df = pd.DataFrame([res])

    if attach_df is not None:
        result_df = pd.concat([attach_df, gmean_df])
    else:
        result_df = pd.concat([data_df, gmean_df])

    return result_df


def append_head2head(data_df, attach_df=None, include_ties=True,
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
                    # if 'kgbm' in c1.lower() and 'ngboost' in c2.lower():
                    #     print(c1, c2, dataset, p_val)
                    n_losses += 1
                else:
                    if np.isnan(p_val):
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
        name: str, processed method name.
        res: dict, result dictionary.

    Return
        - List showing selected hyperparameter values.
        - List showing selected hyperparameter names.
    """
    param_list = []
    param_names = []
    param_types = []

    if 'KGBM' in name:
        param_list.append(result['tree_params']['n_estimators'])
        param_list.append(result['tree_params']['learning_rate'])
        param_names += ['$T$', r'$\eta$']
        param_types += [int, float]

        if result['tree_type'] == 'lgb':
            param_list.append(result['tree_params']['num_leaves'])
            param_names += ['$h$']
            param_types += [int]

        elif result['tree_type'] in ['xgb', 'cb']:
            md = result['tree_params']['max_depth']
            md = -1 if md is None else md
            param_list.append(md)
            param_names += ['$d$']
            param_types += [int]

        param_list.append(int(result['model_params']['k_']))
        param_list.append(result['model_params']['min_scale_'])
        param_names += ['$k$', r'$\rho$']
        param_types += [int, float]

    elif 'Constant' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_names += ['$T$', r'$\eta$']
        param_types += [int, float]

        if result['tree_type'] == 'lgb':
            param_list.append(result['model_params']['num_leaves'])
            param_names += ['$h$']
            param_types += [int]

        elif result['max_depth'] in ['xgb', 'cb']:
            md = result['model_params']['max_depth']
            md = -1 if md is None else md
            param_list.append(md)
            param_names += ['$d$']
            param_types += [int]

    elif 'KNN' in name:
        param_list.append(result['knn_params']['n_neighbors'])
        param_list.append(result['model_params']['k'])
        param_list.append(result['model_params']['min_scale'])
        param_names += ['$k_1$', '$k_2$', r'$\rho$']
        param_types += [int, int, float]

    elif 'NGBoost' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_names += ['$T$']
        param_types += [int]

    elif 'PGBM' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_list.append(result['model_params']['max_leaves'])
        param_names += ['$T$', r'$\eta$', '$h$']
        param_types += [int, float, int]

    if len(name.split('-')) > 1 and 'D' in name.split('-')[1]:
        d_val = result['best_delta']
        d_op = '+' if result['best_delta_op'] == 'add' else '*'
        param_list.append(f'{d_val:.0e}({d_op})')
        param_names += [r'$\Delta$']
        param_types += [str]

    return param_list, param_names, param_types


def process(args, out_dir, logger):
    color, ls, label = util.get_plot_dicts()

    # get results
    crps_list, nll_list, rmse_list = [], [], []
    btime_list, ptime_list = [], []
    param_list = []
    param_names = {}
    param_types = {}
    if args.val:
        val_nll_list = []

    for dataset in args.dataset:
        start = time.time()

        if dataset in args.skip:
            continue

        n = n_dict[dataset]
        n_test = int(n * args.test_frac)
        n_train = n - n_test
        n_feature = p_dict[dataset]

        for fold in args.fold:
            exp_dir = os.path.join(args.in_dir, dataset, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)

            crps = {'dataset': dataset, 'fold': fold}
            nll, rmse = crps.copy(), crps.copy()
            btime = {'dataset': dataset, 'fold': fold, 'n_train': n_train, 'n_features': n_feature}
            ptime = {'dataset': dataset, 'fold': fold, 'n_test': n_test, 'n_features': n_feature}
            param = crps.copy()
            if args.val:
                val_nll = crps.copy()

            for method, res in results:
                name = label[method]
                crps[name] = res['crps']
                nll[name] = res['nll']
                rmse[name] = res['rmse']
                btime[name] = res['total_build_time']
                ptime[name] = res['total_predict_time'] / n_test
                param[name], param_names[name], param_types[name] = get_param_list(name, res)
                if args.val:
                    val_nll[name] = res['val_res']['nll']

            crps_list.append(crps)
            nll_list.append(nll)
            rmse_list.append(rmse)
            btime_list.append(btime)
            ptime_list.append(ptime)
            param_list.append(param)
            if args.val:
                val_nll_list.append(val_nll)

        logger.info(f'{dataset}...{time.time() - start:.3f}s')

    # compile results
    crps_df = pd.DataFrame(crps_list)
    nll_df = pd.DataFrame(nll_list)
    rmse_df = pd.DataFrame(rmse_list)
    btime_df = pd.DataFrame(btime_list)
    ptime_df = pd.DataFrame(ptime_list)
    param_df = pd.DataFrame(param_list)
    if args.val:
        val_nll_df = pd.DataFrame(val_nll_list)

    # aggregate hyperparameters
    param_df = aggregate_params(param_df, param_names, param_types)

    # compute mean and std. error of the mean
    group_cols = ['dataset']

    crps_mean_df = crps_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    nll_mean_df = nll_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    rmse_mean_df = rmse_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    btime_mean_df = btime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    ptime_mean_df = ptime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    if args.val:
        val_nll_mean_df = val_nll_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])

    crps_sem_df = crps_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    nll_sem_df = nll_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    rmse_sem_df = rmse_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    btime_sem_df = btime_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    ptime_sem_df = ptime_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    if args.val:
        val_nll_sem_df = val_nll_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])

    # combine mean and sem into one dataframe
    crps_ms_df = join_mean_sem(crps_mean_df, crps_sem_df, metric='crps')
    nll_ms_df = join_mean_sem(nll_mean_df, nll_sem_df, metric='nll')
    rmse_ms_df = join_mean_sem(rmse_mean_df, rmse_sem_df, metric='rmse')
    btime_ms_df = join_mean_sem(btime_mean_df, btime_sem_df, metric='btime', exclude_sem=True)
    ptime_ms_df = join_mean_sem(ptime_mean_df, ptime_sem_df, metric='ptime', exclude_sem=True)
    if args.val:
        val_nll_ms_df = join_mean_sem(val_nll_mean_df, val_nll_sem_df, metric='nll')

    # attach head-to-head scores
    crps_ms_df = append_head2head(data_df=crps_df, attach_df=crps_ms_df, include_ties=True)
    nll_ms_df = append_head2head(data_df=nll_df, attach_df=nll_ms_df, include_ties=True)
    rmse_ms_df = append_head2head(data_df=rmse_df, attach_df=rmse_ms_df, include_ties=True)
    if args.val:
        val_nll_ms_df = append_head2head(data_df=val_nll_df, attach_df=val_nll_ms_df, include_ties=True)

    # format columns
    crps_ms_df = format_cols(crps_ms_df)
    nll_ms_df = format_cols(nll_ms_df)
    rmse_ms_df = format_cols(rmse_ms_df)
    btime_ms_df = format_cols(btime_ms_df)
    ptime_ms_df = format_cols(ptime_ms_df)
    if args.val:
        val_nll_ms_df = format_cols(val_nll_ms_df)

    # merge specific dataframes
    crps_rmse_df = crps_ms_df.merge(rmse_ms_df, on='dataset')
    nll_rmse_df = nll_ms_df.merge(rmse_ms_df, on='dataset')

    # display
    logger.info(f'\nCRPS (mean):\n{crps_mean_df}')
    logger.info(f'\nNLL (mean):\n{nll_mean_df}')
    logger.info(f'\nRMSE (mean):\n{rmse_mean_df}')
    logger.info(f'\nBuild time (mean):\n{btime_mean_df}')
    logger.info(f'\nParams:\n{param_df}')

    # save
    logger.info(f'\nSaving results to {out_dir}...')

    crps_mean_df.to_csv(os.path.join(out_dir, 'crps_mean.csv'), index=None)
    nll_mean_df.to_csv(os.path.join(out_dir, 'nll_mean.csv'), index=None)
    rmse_mean_df.to_csv(os.path.join(out_dir, 'rmse_mean.csv'), index=None)
    btime_mean_df.to_csv(os.path.join(out_dir, 'btime_mean.csv'), index=None)
    ptime_mean_df.to_csv(os.path.join(out_dir, 'ptime_mean.csv'), index=None)

    crps_sem_df.to_csv(os.path.join(out_dir, 'crps_sem.csv'), index=None)
    nll_sem_df.to_csv(os.path.join(out_dir, 'nll_sem.csv'), index=None)
    rmse_sem_df.to_csv(os.path.join(out_dir, 'rmse_sem.csv'), index=None)
    btime_sem_df.to_csv(os.path.join(out_dir, 'btime_sem.csv'), index=None)
    ptime_sem_df.to_csv(os.path.join(out_dir, 'ptime_sem.csv'), index=None)

    crps_ms_df.to_csv(os.path.join(out_dir, 'crps_str.csv'), index=None)
    nll_ms_df.to_csv(os.path.join(out_dir, 'nll_str.csv'), index=None)
    rmse_ms_df.to_csv(os.path.join(out_dir, 'rmse_str.csv'), index=None)
    btime_ms_df.to_csv(os.path.join(out_dir, 'btime_str.csv'), index=None)
    ptime_ms_df.to_csv(os.path.join(out_dir, 'ptime_str.csv'), index=None)
    if args.val:
        val_nll_ms_df.to_csv(os.path.join(out_dir, 'val_nll_str.csv'), index=None)

    crps_rmse_df.to_csv(os.path.join(out_dir, 'crps_rmse_str.csv'), index=None)
    nll_rmse_df.to_csv(os.path.join(out_dir, 'nll_rmse_str.csv'), index=None)

    param_df.to_csv(os.path.join(out_dir, 'param.csv'), index=None)

    # runtime
    logger.info('\nRuntime...')
    if 'KGBM-LDG' in btime_mean_df and 'NGBoost-D' in btime_mean_df and 'PGBM-DG' in btime_mean_df:
        plot_runtime(bdf=btime_mean_df, pdf=ptime_mean_df, out_dir=out_dir)

        skip_cols = ['dataset', 'n_test', 'n_train', 'n_features']
        ref_col = 'KGBM-LDG'
        drop_cols = [c for c in btime_mean_df.columns if 'knn' in c.lower()]

        bm_df = btime_mean_df.drop(columns=drop_cols)
        pm_df = ptime_mean_df.drop(columns=drop_cols)
        bs_df = btime_sem_df.drop(columns=drop_cols)
        ps_df = ptime_sem_df.drop(columns=drop_cols)

        b_ms_df = join_mean_sem(bm_df, ps_df, metric='btime', exclude_sem=True, mask_cols=skip_cols)
        p_ms_df = join_mean_sem(pm_df, ps_df, metric='ptime', exclude_sem=True, mask_cols=skip_cols)

        int_df = compute_time_intersect(bdf=bm_df, pdf=pm_df, ref_col=ref_col, skip_cols=skip_cols)
        b_ms_df = append_gmean(data_df=bm_df, attach_df=b_ms_df, fmt='int')
        p_ms_df = append_gmean(data_df=pm_df, attach_df=p_ms_df, fmt='sci')

        bp_ms_df = b_ms_df.merge(p_ms_df, on='dataset')
        bp_ms_df = bp_ms_df.merge(int_df, on='dataset', how='left').fillna(-1)
        bp_ms_df = bp_ms_df.drop(columns=['n_train', 'n_test', 'n_features_x', 'n_features_y'])

        n_col = bp_ms_df.pop('n')
        p_col = bp_ms_df.pop('p')
        bp_ms_df.insert(1, 'n', n_col)
        bp_ms_df.insert(2, 'p', p_col)
        bp_ms_df['n'] = bp_ms_df['n'].apply(lambda x: f'{int(x):,}')
        bp_ms_df['p'] = bp_ms_df['p'].apply(lambda x: f'{int(x):,}')

        bp_ms_df = format_cols(bp_ms_df, format_dataset_names=True)
        bp_ms_df['dataset'] = bp_ms_df['dataset'].apply(lambda x: c_dict[x.lower()] if x.lower() in c_dict else x)
        bp_ms_df.to_csv(os.path.join(out_dir, 'bp_time_str.csv'), index=None)

    # boxplot
    logger.info('\nPlotting boxplots...')
    fig, axs = plt.subplots(5, 5, figsize=(4 * 5, 3 * 5))
    axs = axs.flatten()
    i = 0
    for dataset, gf in nll_df.groupby('dataset'):
        gf.boxplot([c for c in gf.columns if c not in ['dataset', 'fold']], ax=axs[i])
        axs[i].set_title(dataset)
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boxplot.pdf'), bbox_inches='tight')

    # Wilcoxon ranked test
    logger.info('\nWilocoxon signed rank test (two-tailed):')
    ref = 'KGBM-LDG'
    for c in ['KNN-D', 'NGBoost-D', 'PGBM-DG']:
        if c not in nll_df:
            continue
        statistic, p_val = stats.wilcoxon(nll_df[ref], nll_df[c])
        statistic_m, p_val_m = stats.wilcoxon(nll_mean_df[ref], nll_mean_df[c])
        logger.info(f'[{ref} & {c}] ALL FOLDS p-val: {p_val}, MEAN of FOLDS p-val: {p_val_m}')


def main(args):

    out_dir = os.path.join(args.out_dir, args.custom_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='temp_output2/prediction/')
    parser.add_argument('--out_dir', type=str, default='output2/postprocess/prediction/')
    parser.add_argument('--custom_dir', type=str, default='results')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'heart', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model', type=str, nargs='+', default=['knn', 'ngboost', 'pgbm', 'kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])
    parser.add_argument('--val', type=int, default=0)
    parser.add_argument('--test_frac', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
