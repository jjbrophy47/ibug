"""
Organize results.
"""
import os
import sys
import copy
import time
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


def process(args, in_dir, in_dir2, out_dir, logger):

    util.plot_settings(fontsize=21, libertine=True)

    if args.combine:
        n_row = 2
        n_col = len(args.dataset)
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 3))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=False)

    for i, dataset in enumerate(args.dataset):
        logger.info(f'{dataset}...')
        start = time.time()

        # result containers
        prob = {'ibug': np.full((len(args.fold), len(args.tree_subsample_frac)), np.nan, dtype=np.float32),
                'pgbm': np.full(len(args.fold), np.nan, dtype=np.float32),
                'ngboost': np.full(len(args.fold), np.nan, dtype=np.float32)}

        ptime = {'ibug': np.full((len(args.fold), len(args.tree_subsample_frac)), np.nan, dtype=np.float32),
                 'pgbm': np.full(len(args.fold), np.nan, dtype=np.float32),
                 'ngboost': np.full(len(args.fold), np.nan, dtype=np.float32)}

        # get results
        for fold in args.fold:

            # get IBUG results for different tree fractions
            args.model_type = ['ibug']
            exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, remove_neighbors=True)

            for method, res in results:
                assert 'ibug' in method
                idx = args.tree_subsample_frac.index(res['predict_args']['tree_subsample_frac'])
                prob['ibug'][fold - 1][idx] = res['test_performance'][f'{args.scoring}_delta']
                ptime['ibug'][fold - 1][idx] = res['timing']['test_pred_time'] / res['data']['n_test'] * 1000  # ms

            # get NGBoost and PGBM results
            args.model_type = ['ngboost', 'pgbm']
            exp_dir2 = os.path.join(args.in_dir2, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir2, logger, remove_neighbors=True)

            for method, res in results:

                if 'pgbm' in method:
                    prob['pgbm'][fold - 1] = res['test_performance'][f'{args.scoring}_delta']
                    ptime['pgbm'][fold - 1] = res['timing']['test_pred_time'] / res['data']['n_test'] * 1000  # ms

                elif 'ngboost' in method:
                    prob['ngboost'][fold - 1] = res['test_performance'][f'{args.scoring}_delta']
                    ptime['ngboost'][fold - 1] = res['timing']['test_pred_time'] / res['data']['n_test'] * 1000  # ms

        # aggregate results
        ibug_dict = {'prob_mean': np.nanmean(prob['ibug'], axis=0),
                     'prob_sem': sem(prob['ibug'], axis=0, nan_policy='omit'),
                     'ptime_mean': np.nanmean(ptime['ibug'], axis=0),
                     'ptime_std': np.nanstd(ptime['ibug'], axis=0)}

        pgbm_dict = {'prob_mean': np.nanmean(prob['pgbm']),
                     'prob_sem': sem(prob['pgbm'], nan_policy='omit'),
                     'ptime_mean': np.nanmean(ptime['pgbm']),
                     'ptime_std': np.nanstd(ptime['pgbm'])}

        ngboost_dict = {'prob_mean': np.nanmean(prob['ngboost']),
                        'prob_sem': sem(prob['ngboost'], nan_policy='omit'),
                        'ptime_mean': np.nanmean(ptime['ngboost']),
                        'ptime_std': np.nanstd(ptime['ngboost'])}

        # plot prob. performance
        dataset_dict = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

        x = np.array(args.tree_subsample_frac) * 100
        y = ibug_dict['prob_mean']
        yerr = ibug_dict['prob_sem']

        y_pgbm = pgbm_dict['prob_mean']
        yerr_pgbm = pgbm_dict['prob_sem']

        y_ngb = ngboost_dict['prob_mean']
        yerr_ngb = ngboost_dict['prob_sem']

        ax = axs[0][i] if args.combine else axs[0]
        ax.plot(x, y, color='red', label='IBUG', ls='-')
        ax.fill_between(x, y + yerr, y - yerr, color='red', alpha=0.1)

        ax.plot([0, 100], [y_pgbm] * 2, color='orange', label='PGBM', ls='-.')
        ax.fill_between([0, 100], [y_pgbm + yerr_pgbm] * 2, [y_pgbm - yerr_pgbm] * 2, color='orange', alpha=0.1)

        ax.plot([0, 100], [y_ngb] * 2, color='purple', label='NGBoost', ls='--')
        ax.fill_between([0, 100], [y_ngb + yerr_ngb] * 2, [y_ngb - yerr_ngb] * 2, color='purple', alpha=0.1)

        xticks = [0, 25, 50, 75, 100]
        dataset_name = dataset_dict[dataset] if dataset in dataset_dict else dataset.capitalize()
        ax.set_title(dataset_name)
        ax.set_xticks(xticks)
        ax.set_xticklabels([])

        if i == 0:
            ax.set_ylabel(f'Test {args.scoring.upper()}')
        elif i == 1:
            ax.legend(fontsize=15)

        # runtime
        y = ibug_dict['ptime_mean']
        yerr = ibug_dict['ptime_std']

        y_pgbm = pgbm_dict['ptime_mean']
        yerr_pgbm = pgbm_dict['ptime_std']

        y_ngb = ngboost_dict['ptime_mean']
        yerr_ngb = ngboost_dict['ptime_std']

        ax = axs[1][i] if args.combine else axs[1]
        ax.plot(x, y, color='red', label='IBUG', ls='-')
        # ax.fill_between(x, y + yerr, y - yerr, color='red', alpha=0.1)

        ax.plot([0, 100], [y_pgbm] * 2, color='orange', label='PGBM', ls='-.')
        # ax.fill_between([0, 100], [y_pgbm + yerr_pgbm] * 2, [y_pgbm - yerr_pgbm] * 2, color='orange', alpha=0.1)

        ax.plot([0, 100], [y_ngb] * 2, color='purple', label='NGBoost', ls='--')
        # ax.fill_between([0, 100], [y_ngb + yerr_ngb] * 2, [y_ngb - yerr_ngb] * 2, color='purple', alpha=0.1)

        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{c:.0f}' for c in xticks])
        ax.set_xlabel('% Trees sampled')
        ax.set_yscale('log')

        if i == 0:
            ax.set_ylabel('Avg. pred. time (ms)')

        if not args.combine:
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{dataset}.png'))
            axs[0].clear()
            axs[1].clear()

    if args.combine:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(os.path.join(out_dir, f'aggregate.pdf'))

    # save
    logger.info(f'Saving results to {out_dir}...')


def main(args):

    assert len(args.tree_subsample_order)

    in_dir = os.path.join(args.in_dir, f'tree_subsample_{args.tree_subsample_order[0]}')
    in_dir2 = os.path.join(args.in_dir2)
    out_dir = os.path.join(args.out_dir, args.tree_subsample_order[0], args.scoring)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, in_dir, in_dir2, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='results/experiments/predict/')
    parser.add_argument('--in_dir2', type=str, default='results/experiments/predict/default/')
    parser.add_argument('--out_dir', type=str, default='results/postprocess/tree_sampling/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model_type', type=str, nargs='+', default=['ibug'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--scoring', type=str, default='nll')
    parser.add_argument('--random_state', type=int, default=1)

    # Additional settings
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
