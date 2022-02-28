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

    # find IBUG, NGBoost, and PGBM columns
    igb_col = None
    ngb_col = None
    pgb_col = None

    for c in bdf.columns:
        if 'ibug' in c:
            ibg_col = c
            assert ibg_col in pdf
        elif 'ngboost' in c:
            ngb_col = c
            assert ngb_col in pdf
        elif 'pgbm' in c:
            pgb_col = c
            assert pgb_col in pdf

    if ibg_col is None and ngb_col is None and pgb_col is None:
        return

    util.plot_settings(fontsize=15, libertine=True)

    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))
    s = 75

    ax = axs[0]
    x = bdf[ibg_col]
    y = bdf[ngb_col]
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
    x = bdf[ibg_col]
    y = bdf[pgb_col]
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e1, 1e6)
    ax.set_ylim(1e1, 1e6)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel('IBUG')
    ax.set_ylabel('PGBM')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'runtime_train.pdf'))
    plt.close('all')

    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 3))

    ax = axs[0]
    x = pdf[ibg_col]
    y = pdf[ngb_col]
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-2, 1e3)
    lims = get_ax_lims(ax)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.tick_params(axis='both', which='minor')
    ax.set_xlabel('IBUG')
    ax.set_ylabel('NGBoost')

    ax = axs[1]
    x = pdf[ibg_col]
    y = pdf[pgb_col]
    ax.scatter(x, y, marker='1', s=s)
    ax.scatter(stats.gmean(x), stats.gmean(y), marker='X', color='red', label='Geo. mean', s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-2, 1e3)
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


def format_dataset_names(df, dmap={'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}):
    """
    Capitalize dataset names.

    Input
        df: pd.DataFrame, Dataframe rename columns.
        dmap: dict, Dict of special cases.

    Return
        Dataframe with the newly formatted columns.
    """
    assert 'dataset' in df
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
        formats = [(['ames', 'news', 'wave'], '{:.1f}'.format),
                   (['facebook', 'meps', 'synthetic'], '{:.1f}'.format),
                   (['star'], '{:.1f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'kin8nm', 'life', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'wine', 'yacht'], '{:.1f}'.format)]

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

        if fmt == 'int':
            res[c] = f'{int(res[c])}'
        elif fmt == 'sci':
            res[c] = f'{res[c]:.1e}'
        else:
            res[c] = f'{res[c]:.1f}'
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
    color, ls, label = util.get_plot_dicts()

    # get results
    test_point_list = []
    test_prob_list = []
    test_prob_delta_list = []
    val_prob_list = []
    val_prob_delta_list = []
    btime_list = []
    ptime_list = []
    param_list = []
    param_names = {}
    param_types = {}

    for dataset in args.dataset:
        start = time.time()

        for fold in args.fold:
            exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)

            test_point = {'dataset': dataset, 'fold': fold}
            test_prob = test_point.copy()
            test_prob_delta = test_point.copy()
            val_prob = test_point.copy()
            val_prob_delta = test_point.copy()
            param = test_point.copy()
            btime = test_point.copy()
            ptime = test_point.copy()

            for method, res in results:
                name = method
                test_point[name] = res['test_performance']['rmse']
                test_prob[name] = res['test_performance'][args.scoring]
                test_prob_delta[name] = res['test_performance'][f'{args.scoring}_delta']
                val_prob[name] = res['val_performance'][args.scoring]
                val_prob_delta[name] = res['val_performance'][f'{args.scoring}_delta']
                btime[name] = res['timing']['tune_train']
                ptime[name] = res['timing']['test_pred_time'] / res['data']['n_test'] * 1000  # milliseconds
                param[name], param_names[name], param_types[name] = get_param_list(name, res)

            test_point_list.append(test_point)
            test_prob_list.append(test_prob)
            test_prob_delta_list.append(test_prob_delta)
            val_prob_list.append(val_prob)
            val_prob_delta_list.append(val_prob_delta)
            btime_list.append(btime)
            ptime_list.append(ptime)
            param_list.append(param)

        logger.info(f'{dataset}...{time.time() - start:.3f}s')

    # compile results
    test_point_df = pd.DataFrame(test_point_list)
    test_prob_df = pd.DataFrame(test_prob_list)
    test_prob_delta_df = pd.DataFrame(test_prob_delta_list)
    val_prob_df = pd.DataFrame(val_prob_list)
    val_prob_delta_df = pd.DataFrame(val_prob_delta_list)
    btime_df = pd.DataFrame(btime_list)
    ptime_df = pd.DataFrame(ptime_list)
    param_df = aggregate_params(pd.DataFrame(param_list), param_names, param_types)  # aggregate hyperparameters

    # compute mean and std. error of the mean
    group_cols = ['dataset']

    test_point_mean_df = test_point_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    test_prob_mean_df = test_prob_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    test_prob_delta_mean_df = test_prob_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    val_prob_mean_df = val_prob_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    val_prob_delta_mean_df = val_prob_delta_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    btime_mean_df = btime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    ptime_mean_df = ptime_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])

    test_point_sem_df = test_point_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    test_prob_sem_df = test_prob_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    test_prob_delta_sem_df = test_prob_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    val_prob_sem_df = val_prob_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    val_prob_delta_sem_df = val_prob_delta_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    btime_std_df = btime_df.groupby(group_cols).std().reset_index().drop(columns=['fold'])
    ptime_std_df = ptime_df.groupby(group_cols).std().reset_index().drop(columns=['fold'])

    # combine mean and sem into one dataframe
    test_point_ms_df = join_mean_sem(test_point_mean_df, test_point_sem_df, metric='rmse')
    test_prob_ms_df = join_mean_sem(test_prob_mean_df, test_prob_sem_df, metric=args.scoring)
    test_prob_delta_ms_df = join_mean_sem(test_prob_delta_mean_df, test_prob_delta_sem_df, metric=args.scoring)
    val_prob_ms_df = join_mean_sem(val_prob_mean_df, val_prob_sem_df, metric=args.scoring)
    val_prob_delta_ms_df = join_mean_sem(val_prob_delta_mean_df, val_prob_delta_sem_df, metric=args.scoring)
    btime_ms_df = join_mean_sem(btime_mean_df, btime_std_df, metric='btime', exclude_sem=True)
    ptime_ms_df = join_mean_sem(ptime_mean_df, ptime_std_df, metric='ptime', exclude_sem=True)

    # attach head-to-head scores
    test_point_ms2_df = append_head2head(data_df=test_point_df, attach_df=test_point_ms_df)
    test_prob_ms2_df = append_head2head(data_df=test_prob_df, attach_df=test_prob_ms_df)
    test_prob_delta_ms2_df = append_head2head(data_df=test_prob_delta_df, attach_df=test_prob_delta_ms_df)
    val_prob_ms2_df = append_head2head(data_df=val_prob_df, attach_df=val_prob_ms_df)
    val_prob_delta_ms2_df = append_head2head(data_df=val_prob_delta_df, attach_df=val_prob_delta_ms_df)

    # attach g. mean scores
    btime_ms2_df = append_gmean(data_df=btime_df, attach_df=btime_ms_df, fmt='int')
    ptime_ms2_df = append_gmean(data_df=ptime_df, attach_df=ptime_ms_df, fmt='float')

    # format columns
    test_point_ms2_df = format_dataset_names(test_point_ms2_df)
    test_prob_ms2_df = format_dataset_names(test_prob_ms2_df)
    test_prob_delta_ms2_df = format_dataset_names(test_prob_delta_ms2_df)
    val_prob_ms2_df = format_dataset_names(val_prob_ms2_df)
    val_prob_delta_ms2_df = format_dataset_names(val_prob_delta_ms2_df)
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

    test_point_ms2_df.to_csv(os.path.join(out_dir, 'test_rmse_str.csv'), index=None)
    test_prob_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.scoring}_str.csv'), index=None)
    test_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.scoring}_delta_str.csv'), index=None)
    val_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.scoring}_str.csv'), index=None)
    val_prob_delta_ms2_df.to_csv(os.path.join(out_dir, f'val_{args.scoring}_delta_str.csv'), index=None)
    btime_ms2_df.to_csv(os.path.join(out_dir, 'test_btime_str.csv'), index=None)
    ptime_ms2_df.to_csv(os.path.join(out_dir, 'test_ptime_str.csv'), index=None)

    test_prob_point_delta_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.scoring}_rmse_str.csv'), index=None)
    bptime_ms2_df.to_csv(os.path.join(out_dir, f'test_bptime_str.csv'), index=None)
    param_df.to_csv(os.path.join(out_dir, 'param.csv'), index=None)

    # delta comparison
    test_prob_dboth_df = test_prob_df.merge(test_prob_delta_df, on=['dataset', 'fold'], how='left')
    test_prob_mean_dboth_df = test_prob_mean_df.merge(test_prob_delta_mean_df, on='dataset', how='left')
    test_prob_sem_dboth_df = test_prob_sem_df.merge(test_prob_delta_sem_df, on='dataset', how='left')
    test_prob_dboth_ms_df = join_mean_sem(test_prob_mean_dboth_df, test_prob_sem_dboth_df, metric=args.scoring)
    test_prob_dboth_ms2_df = append_head2head(data_df=test_prob_dboth_df, attach_df=test_prob_dboth_ms_df)
    test_prob_dboth_ms2_df.to_csv(os.path.join(out_dir, f'test_{args.scoring}_dcomp.csv'), index=None)

    # runtime
    logger.info('\nRuntime...')
    plot_runtime(bdf=btime_mean_df, pdf=ptime_mean_df, out_dir=out_dir)

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
    for c in ['KNN-D', 'NGBoost-D', 'PGBM-DG']:
        if c not in nll_df:
            continue
        statistic, p_val = stats.wilcoxon(nll_df[ref_col], nll_df[c])
        statistic_m, p_val_m = stats.wilcoxon(nll_mean_df[ref_col], nll_mean_df[c])
        logger.info(f'[{ref_col} & {c}] ALL FOLDS p-val: {p_val}, MEAN of FOLDS p-val: {p_val_m}')


def main(args):

    in_dir = os.path.join(args.in_dir, args.custom_in_dir)
    out_dir = os.path.join(args.out_dir, args.custom_out_dir)

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
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='results/predict/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/predict/')
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
    parser.add_argument('--model_type', type=str, nargs='+', default=['knn', 'ngboost', 'pgbm', 'ibug'])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--scoring', type=str, default='nll')

    args = parser.parse_args()
    main(args)
