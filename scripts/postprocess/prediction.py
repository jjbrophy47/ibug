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
    for dataset in args.dataset:
        exp_dir = os.path.join(args.in_dir, dataset)
        results = util.get_results(args, exp_dir, logger, progress_bar=False)

        crps = {'dataset': dataset}
        nll, rmse, tot_time = crps.copy(), crps.copy(), crps.copy()
        for method, res in results:
            name = label[method]
            crps[name] = res['crps']
            nll[name] = res['nll']
            rmse[name] = res['rmse']
            tot_time[name] = res['total_build_time'] + res['total_predict_time']

        crps_list.append(crps)
        nll_list.append(nll)
        rmse_list.append(rmse)
        time_list.append(tot_time)

    # compile results
    crps_df = pd.DataFrame(crps_list)
    nll_df = pd.DataFrame(nll_list)
    rmse_df = pd.DataFrame(rmse_list)
    time_df = pd.DataFrame(time_list)

    # display
    logger.info(f'\nCRPS:\n{crps_df}')
    logger.info(f'\nNLL:\n{nll_df}')
    logger.info(f'\nRMSE:\n{rmse_df}')
    logger.info(f'\nTotal time:\n{time_df}')

    # save
    logger.info(f'\nSaving results to {out_dir}...')
    crps_df.to_csv(os.path.join(out_dir, 'crps.csv'), index=None)
    nll_df.to_csv(os.path.join(out_dir, 'nll.csv'), index=None)
    rmse_df.to_csv(os.path.join(out_dir, 'rmse.csv'), index=None)
    time_df.to_csv(os.path.join(out_dir, 'time.csv'), index=None)


def main(args):

    out_dir = args.out_dir

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
    parser.add_argument('--in_dir', type=str, default='prediction/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/prediction/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames_housing', 'cal_housing', 'concrete', 'energy', 'heart',
                                 'kin8nm', 'life', 'msd', 'naval', 'obesity', 'online_news', 'power', 'protein',
                                 'synth_regression', 'wine', 'yacht'])
    parser.add_argument('--model', type=str, nargs='+', default=['constant', 'kgbm', 'knn', 'ngboost', 'pgbm'])
    parser.add_argument('--scale_bias', type=str, nargs='+', default=[None, 'add', 'mult'])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted', 'uniform'])

    args = parser.parse_args()
    main(args)
