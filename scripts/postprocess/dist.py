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


def process(args, in_dir, out_dir, logger):

    util.plot_settings(fontsize=21, libertine=True)
    rng = np.random.default_rng(args.random_state)

    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    # setup plots
    if args.combine:
        n_row = len(args.dataset)
        n_col = 6
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 3 + 1))
    else:
        fig, axs = plt.subplots(1, 6, figsize=(24, 3))

    # get results for each dataset
    for i, dataset in enumerate(args.dataset):
        logger.info(f'{dataset}...')

        val_list = []
        val_norm_list = []
        test_list = []
        test_norm_list = []

        success = True
        for fold in args.fold:
            val_dict = {'dataset': dataset, 'fold': fold}
            test_dict = {'dataset': dataset, 'fold': fold}

            exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)
            if len(results) != 1:
                success = False
                break

            method, res = results[0]
            assert 'ibug' in method
            val_dict.update(res['val_posterior'][args.scoring])
            test_dict.update(res['test_posterior'][args.scoring])
            val_list.append(val_dict)
            test_list.append(test_dict)

            val_norm_list.append(res['val_performance'][f'{args.scoring}_delta'])
            test_norm_list.append(res['test_performance'][f'{args.scoring}_delta'])

            if fold == 1:
                neighbor_idxs = res['neighbors']['idxs']
                neighbor_vals = res['neighbors']['y_vals']
                k = res['model_params']['k']

        if not success:
            continue

        val_df = pd.DataFrame(val_list).replace(np.inf, np.nan).dropna(axis=1)
        test_df = pd.DataFrame(test_list).replace(np.inf, np.nan).dropna(axis=1)
        dist_cols = [c for c in val_df.columns if c not in ['dataset', 'fold']]

        val_vals = val_df[dist_cols].values  # shape=(n_fold, n_distribution)
        test_vals = test_df[dist_cols].values  # shape=(n_fold, n_distribution)

        val_mean = np.mean(val_vals, axis=0)  # shape=(n_distribution,)
        val_sem = sem(val_vals, axis=0)
        test_mean = np.mean(test_vals, axis=0)  # shape=(n_distribution,)
        test_sem = sem(test_vals, axis=0)

        val_min_idx = np.argmin(val_mean)
        val_norm_mean = np.mean(val_norm_list)
        val_norm_sem = sem(val_norm_list)

        test_min_idx = np.argmin(test_mean)
        test_norm_mean = np.mean(test_norm_list)
        test_norm_sem = sem(test_norm_list)

        val_dist = dist_cols[val_min_idx].capitalize()
        val_dist = 'KDE' if val_dist == 'Kde' else val_dist

        test_dist = dist_cols[test_min_idx].capitalize()
        test_dist = 'KDE' if test_dist == 'Kde' else test_dist

        logger.info(f'\t->val/test distribution: {val_dist}/{test_dist}')

        test_means = [test_norm_mean, test_mean[test_min_idx]]
        test_sems = [test_norm_sem, test_sem[test_min_idx]]
        test_names = ['Normal', test_dist]

        # plot neighbor distributions
        assert neighbor_idxs is not None
        assert neighbor_vals is not None
        test_idxs = rng.choice(neighbor_idxs.shape[0], size=5, replace=False)

        dataset_name = dataset_map[dataset] if dataset in dataset_map else dataset.capitalize()
        stat = 'count'

        for j, test_idx in enumerate(test_idxs):
            ax = axs[i][j] if args.combine else axs[j]
            sns.kdeplot(neighbor_vals[test_idx], ax=ax)

            if dataset == 'meps':
                ax.set_xlim(0, None)

            ax.set_title(f'Test Index {test_idx}')

            if i == len(args.dataset) - 1:
                ax.set_xlabel('Output value')

            if j == 0:
                ax.set_ylabel(f'({dataset_name}, ' r'$k$=' f'{k})' '\n' r'$k$-train density')

            else:
                ax.set_ylabel('')

        ax = axs[i][-1] if args.combine else axs[-1]
        ax.bar(test_names, test_means, yerr=test_sems)
        ax.axhline(0, color='k', ls='-')
        ax.set_title('Avg. Performance')

        if i == len(args.dataset) - 1:
            ax.set_xlabel('Distribution')
        ax.set_ylabel(f'Test {args.scoring.upper()}')

        if not args.combine:
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{dataset}.png'))
            for i in range(6):
                axs[i].clear()

    if args.combine:
        x = 0.8375
        line = plt.Line2D([x, x], [0.05, 0.95], transform=fig.transFigure,
                          color='lightgray', linestyle='--', linewidth=3.5)
        fig.add_artist(line)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'aggregate.pdf'))

    # save
    logger.info(f'Saving results to {out_dir}...')


def main(args):

    in_dir = os.path.join(args.in_dir, args.custom_in_dir)
    out_dir = os.path.join(args.out_dir, args.custom_out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, in_dir, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='results/experiments/predict/')
    parser.add_argument('--out_dir', type=str, default='results/postprocess/')
    parser.add_argument('--custom_in_dir', type=str, default='dist')
    parser.add_argument('--custom_out_dir', type=str, default='dist')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model_type', type=str, nargs='+', default=['ibug'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--scoring', type=str, default='nll')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
