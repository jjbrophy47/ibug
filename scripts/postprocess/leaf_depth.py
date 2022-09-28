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
from predict import method_map


def process(args, in_dir, out_dir, logger):

    util.plot_settings(fontsize=21, libertine=True)
    rng = np.random.default_rng(args.random_state)

    # dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}
    datasets = [d for d in args.dataset if d not in args.exclude_dataset]

    leaf_depth_list = []
    pred_depth_list = []

    # get results for each dataset
    for i, dataset in enumerate(datasets):
        logger.info(f'{dataset}...')

        for fold in args.fold:
            leaf_depth_dict = {'dataset': dataset, 'fold': fold}
            pred_depth_dict = {'dataset': dataset, 'fold': fold}

            try:
                exp_dir = os.path.join(in_dir, dataset, args.scoring, f'fold{fold}')
                results = util.get_results(args, exp_dir, logger, progress_bar=False)
            except:
                continue

            for method, res in results:
                assert 'ibug' in method

                all_depths = res['leaf_depth']['all_depths']  # shape=(total_n_leaves,)
                leaf_counts = res['leaf_depth']['leaf_counts']  # shape=(n_boost,)
                pred_depth = res['leaf_depth']['pred_depth']  # shape=(n_test, n_boost)

                # compute average median leaf_depth per tree
                cum_count = 0
                med_leaf_depths = []
                for leaf_count in leaf_counts:
                    med_leaf_depths.append(np.median(all_depths[cum_count: cum_count+leaf_count]))
                    cum_count += leaf_count
                med_leaf_depths = np.array(med_leaf_depths, dtype=np.float32)  # shape=(n_boost,)
                leaf_depth_dict[method] = np.mean(med_leaf_depths)  # scalar

                # compute average median prediction depth from all trees over all test instances
                med_pred_depth = np.median(pred_depth, axis=1)  # shape=(n_test,)
                pred_depth_dict[method] = np.mean(med_pred_depth)  # scalar

            leaf_depth_list.append(leaf_depth_dict)
            pred_depth_list.append(pred_depth_dict)
        
    # aggregate results
    leaf_df = pd.DataFrame(leaf_depth_list).dropna()
    pred_df = pd.DataFrame(pred_depth_list).dropna()

    # average over folds
    mean_leaf_df = leaf_df.groupby('dataset').mean()
    mean_leaf_df.columns = [method_map[c] if c in method_map else c for c in mean_leaf_df.columns]

    mean_pred_df = pred_df.groupby('dataset').mean()
    mean_pred_df.columns = [method_map[c] if c in method_map else c for c in mean_pred_df.columns]

    # display
    logger.info(f'\nMedian leaf depth (averaged over all trees -> folds):\n{mean_leaf_df}')
    logger.info(f'\nMedian pred. depth (averaged over all trees -> test instances -> folds):\n{mean_pred_df}')

    # save
    logger.info(f'\nSaving results to {out_dir}...')


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
    parser.add_argument('--tree_type', type=str, nargs='+', default=['skrf', 'cb', 'lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--cond_mean_type', type=str, nargs='+', default=['base'])
    parser.add_argument('--gridsearch', type=int, default=1)
    parser.add_argument('--scoring', type=str, default='crps')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--combine', type=int, default=0)

    args = parser.parse_args()
    main(args)
