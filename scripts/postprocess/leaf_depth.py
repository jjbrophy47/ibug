"""
Organize posterior results.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import sem

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util


def process(args, in_dir, out_dir, logger):

    util.plot_settings(fontsize=21, libertine=True)
    rng = np.random.default_rng(args.random_state)

    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}
    datasets = [d for d in args.dataset if d not in args.exclude_dataset]

    # get results for each dataset
    for i, dataset in enumerate(datasets):
        logger.info(f'{dataset}...')

        test_list = []

        for fold in args.fold:
            test_dict = {'dataset': dataset, 'fold': fold}

            exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
            results = util.get_results(args, exp_dir, logger, progress_bar=False)
            if len(results) != 1:
                success = False
                break

            method, res = results[0]
            assert 'ibug' in method

            all_depths = res['leaf_depth']['all_depths']  # shape=(total_n_leaves,)
            leaf_counts = res['leaf_depth']['leaf_counts']  # shape=(n_boost,)
            pred_depth = res['leaf_depth']['pred_depth']  # shape=(n_train, n_boost)

            test_dict.update(res['test_posterior'][args.scoring])
            test_list.append(test_dict)

            test_norm_list.append(res['metrics']['test_delta']['scoring_rule'][f'{args.scoring}'])

        if not success:
            continue

        test_df = pd.DataFrame(test_list).replace(np.inf, np.nan).dropna(axis=1)
        dist_cols = [c for c in test_df.columns if c not in ['dataset', 'fold']]

        test_vals = test_df[dist_cols].values  # shape=(n_fold, n_distribution)

        test_mean = np.mean(test_vals, axis=0)  # shape=(n_distribution,)
        test_sem = sem(test_vals, axis=0)

        test_min_idx = np.argmin(test_mean)
        test_norm_mean = np.mean(test_norm_list)
        test_norm_sem = sem(test_norm_list)

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

    # save
    logger.info(f'Saving results to {out_dir}...')


def main(args):

    in_dir = os.path.join(args.in_dir, args.custom_in_dir)
    out_dir = os.path.join(args.out_dir, args.custom_out_dir, args.scoring)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, in_dir, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/talapas/experiments/predict/')
    parser.add_argument('--out_dir', type=str, default='output/talapas/postprocess/')
    parser.add_argument('--custom_in_dir', type=str, default='leaf_depth')
    parser.add_argument('--custom_out_dir', type=str, default='leaf_depth')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--exclude_dataset', type=str, nargs='+', default=['msd', 'wave'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--model_type', type=str, nargs='+', default=['ibug'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['cb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--cond_mean_type', type=str, nargs='+', default=['base'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--scoring', type=str, default='nll')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
