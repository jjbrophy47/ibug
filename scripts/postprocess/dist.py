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

    util.plot_settings(fontsize=21, libertine=True)

    rng = np.random.default_rng(args.random_state)
    color, ls, label = util.get_plot_dicts()
    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    if args.combine:
        n_row = len(args.dataset)
        n_col = 6
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 3 + 1))

    for i, dataset in enumerate(args.dataset):
        if dataset in args.skip:
            continue

        logger.info(f'{dataset}...')

        # get results
        res_list = []
        score_list = []

        success = True
        for fold in args.fold:
            res_dict = {'dataset': dataset, 'fold': fold}

            exp_dir = os.path.join(args.in_dir, dataset, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)
            if len(results) != 1:
                success = False
                break
            method, res = results[0]
            assert 'kgbm' in method

            res_dict.update(res['dist_res'][args.metric])
            res_list.append(res_dict)
            score_list.append(res[args.metric])

        if not success:
            continue

        df = pd.DataFrame(res_list)
        dist_cols = [c for c in df.columns if c not in ['dataset', 'fold']]
        dist_vals = df[dist_cols].values  # shape=(no. folds, no. distributions)
        dist_mean_df = df[dist_cols].mean(axis=0)
        dist_names = dist_mean_df.index
        dist_mean = dist_mean_df.values
        dist_sem = df[dist_cols].sem(axis=0).values
        min_idx = np.argmin(dist_mean)
        normal_mean = np.mean(score_list)
        normal_sem = sem(score_list)

        dist_name = dist_names[min_idx].capitalize()
        dist_name = 'KDE' if dist_name == 'Kde' else dist_name

        data_list = [score_list, dist_vals[:, min_idx]]
        means = [normal_mean, dist_mean[min_idx]]
        sems = [normal_sem, dist_sem[min_idx]]
        names = ['Normal', dist_name]

        # plot neighbor distributions
        neighbor_idxs = res['neighbor_idxs']
        neighbor_vals = res['neighbor_vals']
        test_idxs = rng.choice(neighbor_idxs.shape[0], size=5, replace=False)

        dataset_name = dataset_map[dataset] if dataset in dataset_map else dataset.capitalize()
        stat = 'count'

        if not args.combine:
            fig, axs = plt.subplots(1, 6, figsize=(24, 3))

        for j, test_idx in enumerate(test_idxs):
            ax = axs[i][j] if args.combine else axs[j]
            sns.kdeplot(neighbor_vals[test_idx], ax=ax)
            if dataset == 'meps':
                ax.set_xlim(0, None)
            ax.set_title(f'Test Index {test_idx}')
            if i == len(args.dataset) - 1:
                ax.set_xlabel('Output value')
            if j == 0:
                ax.set_ylabel(f'({dataset_name})\nNo. nearest train')
                ax.set_ylabel(f'({dataset_name})\n'r'$k$-train density')
            else:
                ax.set_ylabel('')

        print(means, sems)
        ax = axs[i][-1] if args.combine else axs[-1]
        ax.bar(names, means, yerr=sems)
        ax.axhline(0, color='k', ls='-')
        ax.set_title('Avg. Performance')
        if i == len(args.dataset) - 1:
            ax.set_xlabel('Distribution')
        ax.set_ylabel(args.metric.upper())

        if not args.combine:
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{dataset}.png'))

    if args.combine:
        line = plt.Line2D([0.8325, 0.8325], [0.05, 0.95], transform=fig.transFigure,
                          color='lightgray', linestyle='--', linewidth=3.5)
        fig.add_artist(line)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'aggregate.pdf'))

    # save
    logger.info(f'Saving results to {out_dir}...')


def main(args):

    out_dir = os.path.join(args.out_dir, args.custom_dir, args.metric)

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
    parser.add_argument('--in_dir', type=str, default='temp_dist/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/')
    parser.add_argument('--custom_dir', type=str, default='dist')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'heart', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--model', type=str, nargs='+', default=['kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])
    parser.add_argument('--metric', type=str, default='nll')
    parser.add_argument('--random_state', type=int, default=2)
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
