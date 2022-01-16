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


def process(args, out_dir, logger):
    color, ls, label = util.get_plot_dicts()

    # get results
    crps_list, nll_list, rmse_list, time_list = [], [], [], []
    param_list = []
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

                if 'KGBM' in name:
                    param['kgbm_k'] = res['model_params']['k_']
                elif 'KNN' in name:
                    param['knn_k'] = res['model_params']['n_neighbors']
                elif 'NGBoost' in name:
                    param['ngb_iter'] = res['model_params']['n_estimators']
                elif 'PGBM' in name:
                    pass

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
    crps_ms_df = crps_mean_df.copy()
    for c in [x for x in crps_ms_df.columns if x not in group_cols]:
        crps_ms_df[c] = crps_mean_df[c].astype(str) + ' +/- ' + crps_sem_df[c].astype(str)

    print(crps_ms_df)

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

    param_df.to_csv(os.path.join(out_dir, 'param.csv'), index=None)


def main(args):

    delta_dir = 'no_delta' if not args.delta else 'delta'
    out_dir = os.path.join(args.out_dir, delta_dir)

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

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames_housing', 'cal_housing', 'concrete', 'energy', 'heart',
                                 'kin8nm', 'life', 'msd', 'naval', 'obesity', 'online_news', 'power', 'protein',
                                 'synth_regression', 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--model', type=str, nargs='+', default=['constant', 'kgbm', 'knn', 'ngboost', 'pgbm'])
    parser.add_argument('--delta', type=int, default=0)
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.001])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted', 'weighted'])

    args = parser.parse_args()
    main(args)
