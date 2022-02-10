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
from scipy.stats import spearmanr
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util


def process(args, out_dir, logger):

    util.plot_settings(fontsize=15, libertine=True)

    rng = np.random.default_rng(args.random_state)
    color, ls, label = util.get_plot_dicts()
    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    res_list = []
    for dataset in args.dataset:
        logger.info(f'{dataset}...')

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
            assert 'affinity_dict' in res
            for k, v in res['affinity_dict'].items():
                if 'sem' in list(v.keys()):
                    success = False
                    break
                res_dict[f'{k}_mean'] = v['mean']
                res_dict[f'{k}_std'] = v['std']
            if not success:
                break
            res_dict['n_train'] = res['n_train']
            res_list.append(res_dict)

    df = pd.DataFrame(res_list)[['dataset', 'fold', 'cnt_tree_mean', 'cnt_tree_std', 'n_train']]
    df['cnt_tree_mean_frac'] = df['cnt_tree_mean'] / df['n_train']
    df['cnt_tree_std_frac'] = df['cnt_tree_std'] / df['n_train']
    df = df.sort_values('n_train')

    spearman = spearmanr(df['n_train'], df['cnt_tree_mean_frac'])[0]

    fig, ax = plt.subplots()
    ax.errorbar(df['n_train'], df['cnt_tree_mean_frac'] * 100,
                yerr=df['cnt_tree_std_frac'] * 100, fmt='o',
                label=f'Spearman={spearman:.3f}')
    ax.set_ylim(0, 100)
    ax.set_xscale('log')
    ax.set_ylabel('% train visited / tree')
    ax.set_xlabel('No. train')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot.pdf'), bbox_inches='tight')

    # save
    logger.info(f'Saving results to {out_dir}...')
    df.to_csv(os.path.join(out_dir, 'result.csv'))


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
    parser.add_argument('--in_dir', type=str, default='temp_output2/affinity_stats/')
    parser.add_argument('--out_dir', type=str, default='output2/postprocess/')
    parser.add_argument('--custom_dir', type=str, default='affinity_stats')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'heart', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--skip', type=str, nargs='+', default=['heart'])
    parser.add_argument('--fold', type=int, nargs='+', default=[1])
    parser.add_argument('--model', type=str, nargs='+', default=['kgbm'])
    parser.add_argument('--tree_frac', type=str, nargs='+', default=[1.0])
    parser.add_argument('--min_scale_pct', type=float, nargs='+', default=[0.0])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb'])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--delta', type=int, nargs='+', default=[1])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])
    parser.add_argument('--random_state', type=int, default=1)

    args = parser.parse_args()
    main(args)
