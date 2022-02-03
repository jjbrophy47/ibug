"""
Organize results.
"""
import os
import sys
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


def format_cols(df, dataset_map={'Meps': 'MEPS', 'Msd': 'MSD', 'Star': 'STAR'}):
    """
    Format columns.

    Input
        df: pd.DataFrame, Dataframe rename columns.

    Return
        Dataframe with the newly formatted columns.
    """
    new_cols = []
    for c in df.columns:
        if '-' in c:
            items = c.split('-')
            c1 = items[0]
            c2 = items[1].replace('D', '').replace('G', '')
            if len(c2) == 0:
                c = c1
            else:
                c = f'{c1}-{c2}'
        else:
            c = c.capitalize()
        new_cols.append(c)
    df.columns = new_cols

    return df


def join_mean_sem(mean_df, sem_df, metric, mask_cols=['dataset']):
    """
    Joins floats from two dataframes into one dataframe of strings.

    Input
        mean_df: pd.DataFrame, dataframe with mean values.
        sem_df: pd.DataFrame, dataframe with SEM values.
        mask_cols: list, Columns to not join but included in the result.
        fmt: object, Format of the float columns.
        join_str: str, String used to connect the two floats.

    Return
        pd.DataFrame object with all string columns.
    """
    assert np.all(mean_df.columns == sem_df.columns)
    float_cols = [c for c in mean_df.columns if c not in mask_cols]

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
    else:
        formats = [(['ames', 'news', 'wave'], '{:.0f}'.format),
                   (['facebook', 'meps', 'synthetic'], '{:.2f}'.format),
                   (['star'], '{:.1f}'.format),
                   (['bike', 'california', 'communities', 'concrete', 'energy',
                     'kin8nm', 'life', 'msd',
                     'naval', 'obesity', 'power', 'protein',
                     'superconductor', 'wine', 'yacht'], '{:.3f}'.format)]

    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

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
        dataset_list += [dataset_map[d] if d in dataset_map else d.capitalize() for d in datasets]

    temp1_df = pd.concat(m_list)
    temp2_df = pd.concat(s_list)

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
        res = {'Dataset': dataset_name}

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
                if c1_mean < c2_mean and p_val < p_val_threshold:
                    n_wins += 1
                elif c2_mean < c1_mean and p_val < p_val_threshold:
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
        param_list.append(result['model_params']['n_neighbors'])
        param_names += ['$k$']
        param_types += [int]

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
    crps_list, nll_list, rmse_list, time_list = [], [], [], []
    param_list = []
    param_names = {}
    param_types = {}

    for dataset in args.dataset:
        if dataset in args.skip:
            continue

        for fold in args.fold:
            exp_dir = os.path.join(args.in_dir, dataset, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)

            crps = {'dataset': dataset, 'fold': fold}
            nll, rmse, tot_time = crps.copy(), crps.copy(), crps.copy()
            param = crps.copy()
            for method, res in results:
                name = label[method]
                crps[name] = res['crps']
                nll[name] = res['nll']
                rmse[name] = res['rmse']
                tot_time[name] = res['total_build_time'] + res['total_predict_time']
                param[name], param_names[name], param_types[name] = get_param_list(name, res)

            crps_list.append(crps)
            nll_list.append(nll)
            rmse_list.append(rmse)
            time_list.append(tot_time)
            param_list.append(param)

    # compile results
    crps_df = pd.DataFrame(crps_list)
    nll_df = pd.DataFrame(nll_list)
    rmse_df = pd.DataFrame(rmse_list)
    time_df = pd.DataFrame(time_list)
    param_df = pd.DataFrame(param_list)

    # aggregate hyperparameters
    param_df = aggregate_params(param_df, param_names, param_types)

    # compute mean and std. error of the mean
    group_cols = ['dataset']

    crps_mean_df = crps_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    nll_mean_df = nll_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    rmse_mean_df = rmse_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])
    time_mean_df = time_df.groupby(group_cols).mean().reset_index().drop(columns=['fold'])

    crps_sem_df = crps_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    nll_sem_df = nll_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    rmse_sem_df = rmse_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])
    time_sem_df = time_df.groupby(group_cols).sem().reset_index().drop(columns=['fold'])

    # combine mean and sem into one dataframe
    crps_ms_df = join_mean_sem(crps_mean_df, crps_sem_df, metric='crps')
    nll_ms_df = join_mean_sem(nll_mean_df, nll_sem_df, metric='nll')
    rmse_ms_df = join_mean_sem(rmse_mean_df, rmse_sem_df, metric='rmse')
    time_ms_df = join_mean_sem(time_mean_df, time_sem_df, metric='time')

    # attach head-to-head scores
    crps_ms_df = append_head2head(data_df=crps_df, attach_df=crps_ms_df, include_ties=True)
    nll_ms_df = append_head2head(data_df=nll_df, attach_df=nll_ms_df, include_ties=True)
    rmse_ms_df = append_head2head(data_df=rmse_df, attach_df=rmse_ms_df, include_ties=True)
    time_ms_df = append_head2head(data_df=time_df, attach_df=time_ms_df, include_ties=True)

    # format columns
    crps_ms_df = format_cols(crps_ms_df)
    nll_ms_df = format_cols(nll_ms_df)
    rmse_ms_df = format_cols(rmse_ms_df)
    time_ms_df = format_cols(time_ms_df)

    # merge specific dataframes
    crps_rmse_df = crps_ms_df.merge(rmse_ms_df, on='Dataset')
    nll_rmse_df = nll_ms_df.merge(rmse_ms_df, on='Dataset')

    # display
    logger.info(f'\nCRPS (mean):\n{crps_mean_df}')
    logger.info(f'\nNLL (mean):\n{nll_mean_df}')
    logger.info(f'\nRMSE (mean):\n{rmse_mean_df}')
    logger.info(f'\nTotal time (mean):\n{time_mean_df}')
    logger.info(f'\nParams:\n{param_df}')

    # save
    logger.info(f'\nSaving results to {out_dir}...')

    crps_mean_df.to_csv(os.path.join(out_dir, 'crps_mean.csv'), index=None)
    nll_mean_df.to_csv(os.path.join(out_dir, 'nll_mean.csv'), index=None)
    rmse_mean_df.to_csv(os.path.join(out_dir, 'rmse_mean.csv'), index=None)
    time_mean_df.to_csv(os.path.join(out_dir, 'time_mean.csv'), index=None)

    crps_sem_df.to_csv(os.path.join(out_dir, 'crps_sem.csv'), index=None)
    nll_sem_df.to_csv(os.path.join(out_dir, 'nll_sem.csv'), index=None)
    rmse_sem_df.to_csv(os.path.join(out_dir, 'rmse_sem.csv'), index=None)
    time_sem_df.to_csv(os.path.join(out_dir, 'time_sem.csv'), index=None)

    crps_ms_df.to_csv(os.path.join(out_dir, 'crps_str.csv'), index=None)
    nll_ms_df.to_csv(os.path.join(out_dir, 'nll_str.csv'), index=None)
    rmse_ms_df.to_csv(os.path.join(out_dir, 'rmse_str.csv'), index=None)
    time_ms_df.to_csv(os.path.join(out_dir, 'time_str.csv'), index=None)

    crps_rmse_df.to_csv(os.path.join(out_dir, 'crps_rmse_str.csv'), index=None)
    nll_rmse_df.to_csv(os.path.join(out_dir, 'nll_rmse_str.csv'), index=None)

    param_df.to_csv(os.path.join(out_dir, 'param.csv'), index=None)


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
    parser.add_argument('--in_dir', type=str, default='temp_prediction/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/prediction/')
    parser.add_argument('--custom_dir', type=str, default='results')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'heart', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--model', type=str, nargs='+', default=['knn', 'ngboost', 'pgbm', 'kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])

    args = parser.parse_args()
    main(args)
