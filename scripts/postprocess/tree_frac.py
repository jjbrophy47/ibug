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
from scipy import stats
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util


def process(args, out_dir, logger):

    color, ls, label = util.get_plot_dicts()

    for dataset in args.dataset:
        if dataset in args.skip:
            continue

        # get results
        crps = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
                'pgbm': np.zeros(len(args.fold)),
                'ngboost': np.zeros(len(args.fold))}
        nll = crps.copy()
        runtime = crps.copy()

        for fold in args.fold:
            exp_dir = os.path.join(args.in_dir, dataset, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)
            for method, res in results:
                if 'kgbm' in method:
                    idx = args.tree_frac.index(res['tree_frac'])
                    crps['kgbm'][fold - 1][idx] = res['crps']
                    nll['kgbm'][fold - 1][idx] = res['nll']
                    runtime['kgbm'][fold - 1][idx] = res['total_build_time'] + res['total_predict_time']
                elif 'pgbm' in method:
                    crps['pgbm'][fold - 1] = res['crps']
                    nll['pgbm'][fold - 1] = res['nll']
                    runtime['pgbm'][fold - 1] = res['total_build_time'] + res['total_predict_time']
                elif 'ngboost' in method:
                    crps['ngboost'][fold - 1] = res['crps']
                    nll['ngboost'][fold - 1] = res['nll']
                    runtime['ngboost'][fold - 1] = res['total_build_time'] + res['total_predict_time']

        print(crps)

    # save
    logger.info(f'\nSaving results to {out_dir}...')


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
    parser.add_argument('--in_dir', type=str, default='temp_tree_frac/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/tree_frac/')
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
    parser.add_argument('--model', type=str, nargs='+', default=['ngboost', 'pgbm', 'kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted', 'weighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])

    args = parser.parse_args()
    main(args)
