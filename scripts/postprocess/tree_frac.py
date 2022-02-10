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
from scipy import stats
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util


def process(args, out_dir, logger):

    util.plot_settings(fontsize=21, libertine=True)
    color, ls, label = util.get_plot_dicts()

    metric_dict = {'crps': 'CRPS', 'nll': 'NLL'}
    metric2_dict = {'build': 'Build', 'pred': 'Avg. pred. time (s)', 'bpred': 'Total time (s)'}
    dataset_dict = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    if args.combine:
        n_row = 2
        n_col = len(args.dataset)
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 3))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=False)

    for i, dataset in enumerate(args.dataset):
        start = time.time()

        if dataset in args.skip:
            continue

        # get results
        crps = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
                'pgbm': np.zeros(len(args.fold)),
                'ngboost': np.zeros(len(args.fold))}
        nll = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
               'pgbm': np.zeros(len(args.fold)),
               'ngboost': np.zeros(len(args.fold))}
        build = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
                 'pgbm': np.zeros(len(args.fold)),
                 'ngboost': np.zeros(len(args.fold))}
        pred = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
                'pgbm': np.zeros(len(args.fold)),
                'ngboost': np.zeros(len(args.fold))}
        bpred = {'kgbm': np.zeros((len(args.fold), len(args.tree_frac))),
                 'pgbm': np.zeros(len(args.fold)),
                 'ngboost': np.zeros(len(args.fold))}

        for fold in args.fold:

            n_test = None

            # get KGBM results for different tree fractions
            exp_dir = os.path.join(args.in_dir, dataset, f'fold{fold}')
            args.model = ['kgbm']
            results = util.get_results(args, exp_dir, logger, remove_neighbors=True)
            for method, res in results:
                assert 'kgbm' in method
                n_test = res['n_test']
                idx = args.tree_frac.index(res['tree_frac'])
                crps['kgbm'][fold - 1][idx] = res['crps']
                nll['kgbm'][fold - 1][idx] = res['nll']
                build['kgbm'][fold - 1][idx] = res['total_build_time']
                pred['kgbm'][fold - 1][idx] = res['total_predict_time'] / n_test
                bpred['kgbm'][fold - 1][idx] = res['total_build_time'] + res['total_predict_time']

            # get NGBoost and PGBM results
            exp_dir2 = os.path.join(args.in_dir2, dataset, f'fold{fold}')
            args.model = ['ngboost', 'pgbm']
            results = util.get_results(args, exp_dir2, logger, remove_neighbors=True)
            for method, res in results:
                if 'pgbm' in method:
                    crps['pgbm'][fold - 1] = res['crps']
                    nll['pgbm'][fold - 1] = res['nll']
                    build['pgbm'][fold - 1] = res['total_build_time']
                    pred['pgbm'][fold - 1] = res['total_predict_time'] / n_test
                    bpred['pgbm'][fold - 1] = res['total_build_time'] + res['total_predict_time']
                elif 'ngboost' in method:
                    crps['ngboost'][fold - 1] = res['crps']
                    nll['ngboost'][fold - 1] = res['nll']
                    build['ngboost'][fold - 1] = res['total_build_time']
                    build['ngboost'][fold - 1] = res['total_build_time'] + res['total_predict_time']
                    pred['ngboost'][fold - 1] = res['total_predict_time'] / n_test
                    bpred['ngboost'][fold - 1] = res['total_build_time'] + res['total_predict_time']

        kgbm_dict = {'crps_mean': np.mean(crps['kgbm'], axis=0), 'crps_sem': sem(crps['kgbm'], axis=0),
                     'nll_mean': np.mean(nll['kgbm'], axis=0), 'nll_sem': sem(nll['kgbm'], axis=0),
                     'build_mean': np.mean(build['kgbm'], axis=0), 'build_std': np.std(build['kgbm'], axis=0),
                     'pred_mean': np.mean(pred['kgbm'], axis=0), 'pred_std': np.std(pred['kgbm'], axis=0),
                     'bpred_mean': np.mean(bpred['kgbm'], axis=0), 'bpred_std': np.std(bpred['kgbm'], axis=0)}

        pgbm_dict = {'crps_mean': np.mean(crps['pgbm']), 'crps_sem': sem(crps['pgbm']),
                     'nll_mean': np.mean(nll['pgbm']), 'nll_sem': sem(nll['pgbm']),
                     'build_mean': np.mean(build['pgbm']), 'build_std': np.std(build['pgbm']),
                     'pred_mean': np.mean(pred['pgbm']), 'pred_std': np.std(pred['pgbm']),
                     'bpred_mean': np.mean(bpred['pgbm']), 'bpred_std': np.std(bpred['pgbm'])}

        ngboost_dict = {'crps_mean': np.mean(crps['ngboost']), 'crps_sem': sem(crps['ngboost']),
                        'nll_mean': np.mean(nll['ngboost']), 'nll_sem': sem(nll['ngboost']),
                        'build_mean': np.mean(build['ngboost']), 'build_std': np.std(build['ngboost']),
                        'pred_mean': np.mean(pred['ngboost']), 'pred_std': np.std(pred['ngboost']),
                        'bpred_mean': np.mean(bpred['ngboost']), 'bpred_std': np.std(bpred['ngboost'])}

        # prob. performance
        x = np.array(args.tree_frac) * 100
        y = kgbm_dict[f'{args.metric}_mean']
        yerr = kgbm_dict[f'{args.metric}_sem']

        y_pgbm = pgbm_dict[f'{args.metric}_mean']
        yerr_pgbm = pgbm_dict[f'{args.metric}_sem']
        ypgbm1 = [y_pgbm + yerr_pgbm] * 2
        ypgbm2 = [y_pgbm - yerr_pgbm] * 2

        y_ngboost = ngboost_dict[f'{args.metric}_mean']
        yerr_ngboost = ngboost_dict[f'{args.metric}_sem']
        yngboost1 = [y_ngboost + yerr_ngboost] * 2
        yngboost2 = [y_ngboost - yerr_ngboost] * 2

        ax = axs[0][i] if args.combine else axs[0]

        ax.plot(x, y, color='red', label='KGBM', ls='-')
        ax.fill_between(x, y + yerr, y - yerr, color='red', alpha=0.1)

        ax.plot([0, 100], [y_pgbm] * 2, color='orange', label='PGBM', ls='-.')
        ax.fill_between([0, 100], ypgbm1, ypgbm2, color='orange', alpha=0.1)

        ax.plot([0, 100], [y_ngboost] * 2, color='purple', label='NGBoost', ls='--')
        ax.fill_between([0, 100], yngboost1, yngboost2, color='purple', alpha=0.1)

        xticks = [0, 25, 50, 75, 100]
        dataset_name = dataset_dict[dataset] if dataset in dataset_dict else dataset.capitalize()
        ax.set_title(dataset_name)
        ax.set_xticks(xticks)
        ax.set_xticklabels([])

        if i == 0:
            ax.set_ylabel(f'Test {metric_dict[args.metric]}')
        elif i == 1:
            ax.legend(fontsize=15)

        # runtime
        y = kgbm_dict[f'{args.metric2}_mean']
        yerr = kgbm_dict[f'{args.metric2}_std']

        y_pgbm = pgbm_dict[f'{args.metric2}_mean']
        yerr_pgbm = pgbm_dict[f'{args.metric2}_std']
        ypgbm1 = [y_pgbm + yerr_pgbm] * 2
        ypgbm2 = [y_pgbm - yerr_pgbm] * 2

        y_ngboost = ngboost_dict[f'{args.metric2}_mean']
        yerr_ngboost = ngboost_dict[f'{args.metric2}_std']
        yngboost1 = [y_ngboost + yerr_ngboost] * 2
        yngboost2 = [y_ngboost - yerr_ngboost] * 2

        ax = axs[1][i] if args.combine else axs[1]
        ax.plot(x, y, color='red', label='KGBM', ls='-')
        ax.fill_between(x, y + yerr, y - yerr, color='red', alpha=0.1)

        ax.plot([0, 100], [y_pgbm] * 2, color='orange', label='PGBM', ls='-.')
        ax.fill_between([0, 100], ypgbm1, ypgbm2, color='orange', alpha=0.1)

        ax.plot([0, 100], [y_ngboost] * 2, color='purple', label='NGBoost', ls='--')
        ax.fill_between([0, 100], yngboost1, yngboost2, color='purple', alpha=0.1)

        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{c:.0f}' for c in xticks])
        ax.set_xlabel('% Trees sampled')
        ax.set_yscale('log')

        if i == 0:
            ax.set_ylabel(f'{metric2_dict[args.metric2]}')

        if not args.combine:
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{dataset}.png'))
            axs[0].clear()
            axs[1].clear()

        logger.info(f'{dataset}...{time.time() - start:.3f}s')

    if args.combine:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(os.path.join(out_dir, f'aggregate.pdf'))

    # save
    logger.info(f'Saving results to {out_dir}...')


def main(args):

    out_dir = os.path.join(args.out_dir, f'{args.metric}_{args.metric2}')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='/Volumes/30/kgbm/temp_tree_frac/')
    parser.add_argument('--in_dir2', type=str, default='temp_prediction')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/tree_frac/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'heart', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model', type=str, nargs='+', default=['kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted', 'weighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])
    parser.add_argument('--metric', type=str, default='nll')
    parser.add_argument('--metric2', type=str, default='bpred')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
