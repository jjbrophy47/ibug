"""
Process affinity and leaf densities.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util

# constants
tree_map = {'cb': 'CatBoost', 'lgb': 'LightGBM', 'xgb': 'XGBoost'}


def process(args, in_dir, out_dir, logger):

    util.plot_settings(fontsize=args.font_size, libertine=True) # 20 for agg.pdf, 15 for affinity.pdf

    res_list = []
    affinity_list = []
    max_density_list = []

    n_agg_datasets = len(args.agg_datasets)
    _, axs = plt.subplots(1, n_agg_datasets, figsize=(4*n_agg_datasets, 4), sharey=True)

    j = 0
    for dataset in args.dataset:
        logger.info(f'{dataset}...')
        fails = 0

        for fold in args.fold:
            res_dict = {'dataset': dataset, 'fold': fold}

            exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)
            if len(results) == 1:
                method, res = results[0]
                assert 'ibug' in method
                assert 'affinity_count' in res
                assert 'leaf_density' in res

                affinity = res['affinity_count']['mean']  # shape=(n_boost,)
                max_density = res['leaf_density']['max']  # shape=(n_boost,)
                n_boost = len(affinity)

                frac_list = np.linspace(0, 1, 100)
                affinity_arr = np.zeros(len(frac_list))
                max_density_arr = np.zeros(len(frac_list))
                for i, frac in enumerate(frac_list):
                    idx = min(int(frac * n_boost), n_boost - 1)
                    affinity_arr[i] = affinity[idx]
                    max_density_arr[i] = max_density[idx]

                affinity_list.append(affinity_arr)
                max_density_list.append(max_density_arr)

                # average density and affinity over all trees
                res_dict['max_density'] = np.mean(max_density)
                res_dict['affinity'] = np.mean(affinity)
                res_dict['n_train'] = res['data']['n_train']
                res_list.append(res_dict)
            else:
                fails += 1
                continue

        if fails == len(args.fold):
            continue

        # plot affinity and leaf density for each dataset
        affinity = np.vstack(affinity_list)
        max_density = np.vstack(max_density_list)

        # _, ax = plt.subplots(figsize=(4, 3), sharey=True)
        # _, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

        if dataset in args.agg_datasets:
            ax = axs[j]
            x = frac_list * 100
            y, yerr = np.mean(affinity, axis=0) * 100, np.std(affinity, axis=0) * 100
            ax.plot(x, y)
            ax.fill_between(x, y + yerr, y - yerr, alpha=0.1)
            # ax.set_title('Avg. Affinity per Test Instance')
            ax.set_title(f'{dataset.capitalize()}')
            ax.set_xlabel('Boosting iteration (%)')
            ax.set_ylim(0, 100)
            if j == 0:
                ax.set_ylabel(f'({tree_map[args.tree_type[0]]})\n% train visited')
            j += 1

        # ax = axs[1]
        # x = frac_list * 100
        # y, yerr = np.mean(max_density, axis=0) * 100, np.std(max_density, axis=0) * 100
        # ax.plot(x, y)
        # ax.fill_between(x, y + yerr, y - yerr, alpha=0.1)
        # ax.set_title('Max. Leaf Density')
        # ax.set_xlabel('Boosting iteration (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'agg.pdf'), bbox_inches='tight')
    plt.close('all')

    # plot averge affinity and leaf denstity over all trees
    df = pd.DataFrame(res_list)
    mean_df = df.groupby('dataset').mean().reset_index().drop(columns=['fold'])
    std_df = df.groupby('dataset').std().reset_index().drop(columns=['fold'])

    logger.info(f'Saving results to {out_dir}...')
    mean_df.to_csv(os.path.join(out_dir, 'affinity.csv'))

    _, ax = plt.subplots(figsize=(4, 3), sharey=True)
    # _, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    # ax = axs[0]
    ax.errorbar(x=mean_df['n_train'], y=mean_df['affinity'] * 100, yerr=std_df['affinity'] * 100,
                fmt='o', ecolor='k', color='blue', lw=1, capsize=2)
    # ax.set_title('Avg. Affinity per Test Instance')
    ax.set_xlabel('No. train')
    ax.set_ylabel('% train visited / tree')
    ax.set_xscale('log')
    ax.set_ylim(0, 100)

    # ax = axs[1]
    # ax.errorbar(x=mean_df['n_train'], y=mean_df['max_density'] * 100, yerr=std_df['max_density'] * 100,
    #             fmt='o', ecolor='k', color='blue', lw=1, capsize=2)
    # ax.set_title('Max. Leaf Density')
    # ax.set_xlabel('No. training examples')
    # ax.set_ylabel('Max. % train / tree')
    # ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'affinity.pdf'), bbox_inches='tight')


def main(args):

    assert len(args.tree_type) == 1

    in_dir = os.path.join(args.in_dir, args.custom_in_dir)
    out_dir = os.path.join(args.out_dir, args.custom_out_dir, args.tree_type[0])

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
    parser.add_argument('--in_dir', type=str, default='output/talapas/experiments/predict/')
    parser.add_argument('--out_dir', type=str, default='output/talapas/postprocess/')
    parser.add_argument('--custom_in_dir', type=str, default='default')
    parser.add_argument('--custom_out_dir', type=str, default='affinity')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
        default=['ames', 'bike', 'california', 'communities', 'concrete',
        'energy', 'facebook', 'kin8nm', 'life', 'meps',
        'msd', 'naval', 'obesity', 'news', 'power', 'protein',
        'star', 'superconductor', 'synthetic', 'wave',
        'wine', 'yacht'])
    parser.add_argument('--agg_datasets', type=str, nargs='+',
        default=['concrete', 'kin8nm', 'synthetic', 'wine'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--model_type', type=str, nargs='+', default=['ibug'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['cb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--cond_mean_type', type=str, nargs='+', default=['base'])
    parser.add_argument('--scoring', type=str, default='crps')
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=1)

    parser.add_argument('--font_size', type=int, default=15)

    args = parser.parse_args()
    main(args)
