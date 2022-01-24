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
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util


def join_mean_sem(mean_df, sem_df, mask_cols=['dataset'],
                  fmt='{:.3f}'.format, join_str=' +/- '):
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

    temp1_df = mean_df[float_cols].applymap(fmt).astype(str)
    temp2_df = sem_df[float_cols].applymap(fmt).astype(str)
    result_df = temp1_df + join_str + temp2_df

    for c in mask_cols:
        result_df.insert(loc=0, column=c, value=mean_df[c])

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

    for dataset, df in param_df.groupby(['dataset']):
        dataset_dict = {'dataset': dataset}

        for name in df.columns:
            if name in ['dataset', 'fold']:
                continue

            values_list = [x for x in df[name].values if isinstance(x, list)]

            print(df)

            # aggregate each list of values into one string
            param_str = ''
            if len(values_list) > 0:
                values = np.vstack(values_list)

                for i in range(values.shape[1]):  # per parameter
                    param_vals = []

                    print(i, name, values[:, i], param_names[name][i])

                    # format value
                    for x in values[:, i]:
                        if param_types[name][i] == float:
                            param_val = f'{param_types[name][i](x):.3f}'.rstrip('0').rstrip('.')
                        else:
                            param_val = str(param_types[name][i](x))
                        param_vals.append(param_val)

                    param_vals_str = ', '.join(param_vals)
                    param_str += f'{param_names[name][i]}: {param_vals_str}\n'

            dataset_dict[name] = param_str.strip()
        param_list.append(dataset_dict)
    param_df = pd.DataFrame(param_list)

    return param_df


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

    print(name)

    if 'KGBM' in name:
        param_list.append(int(result['model_params']['k_']))
        param_list.append(result['model_params']['min_scale_'])
        param_list.append(result['tree_params']['n_estimators'])
        param_list.append(result['tree_params']['learning_rate'])
        param_list.append(result['tree_params']['max_bin'])
        param_names += ['k', 'min_scale', 'n_tree', 'lr', 'max_bin']
        param_types += [int, float, int, float, int]

        # if result['tree_type'] == 'lgb':
        if 'num_leaves' in result['tree_params']:  # TEMP
            param_list.append(result['tree_params']['num_leaves'])
            param_names += ['num_leaves']
            param_types += [int]

        # elif result['tree_type'] in ['xgb', 'cb']:
        if 'max_depth' in result['tree_params']:  # TEMP
            md = result['tree_params']['max_depth']
            md = -1 if md is None else md
            param_list.append(md)
            param_names += ['max_depth']
            param_types += [int]

    elif 'Constant' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['learning_rate'])
        param_list.append(result['model_params']['max_bin'])
        param_names += ['n_tree', 'lr', 'max_bin']
        param_types += [int, float, int]

        # if result['tree_type'] == 'lgb':
        if 'num_leaves' in result['model_params']:  # TEMP
            param_list.append(result['model_params']['num_leaves'])
            param_names += ['num_leaves']
            param_types += [int]

        # elif result['max_depth'] in ['xgb', 'cb']:
        if 'max_depth' in result['model_params']:  # TEMP
            md = result['model_params']['max_depth']
            md = -1 if md is None else md
            param_list.append(md)
            param_names += ['max_depth']
            param_types += [int]

    elif 'KNN' in name:
        param_list.append(result['model_params']['n_neighbors'])
        param_names += ['k']
        param_types += [int]

    elif 'NGBoost' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_names += ['n_tree']
        param_types += [int]

    elif 'PGBM' in name:
        param_list.append(result['model_params']['n_estimators'])
        param_list.append(result['model_params']['max_leaves'])
        param_list.append(result['model_params']['learning_rate'])
        param_list.append(result['model_params']['max_bin'])
        param_names += ['n_tree', 'max_leaves', 'lr', 'max_bin']
        param_types += [int, int, float, int]

    if len(name.split('-')) > 1 and 'D' in name.split('-')[1]:
        param_list.append(result['best_delta'])
        param_list.append(result['best_delta_op'])
        param_names += ['delta', 'delta_op']
        param_types += [float, str]

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
    crps_ms_df = join_mean_sem(crps_mean_df, crps_sem_df)
    nll_ms_df = join_mean_sem(nll_mean_df, nll_sem_df)
    rmse_ms_df = join_mean_sem(rmse_mean_df, rmse_sem_df)
    time_ms_df = join_mean_sem(time_mean_df, time_sem_df)

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
    parser.add_argument('--model', type=str, nargs='+', default=['constant', 'kgbm', 'knn', 'ngboost', 'pgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted', 'weighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[0, 1])

    args = parser.parse_args()
    main(args)
