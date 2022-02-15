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

ld_mean = {}
ld_mean['ames'] = 68.0322580645161
ld_mean['bike'] = 205.11475409836
ld_mean['california'] = 479.354838709677
ld_mean['communities'] = 46.2903225806451
ld_mean['concrete'] = 49.4
ld_mean['energy'] = 36.8073614722944
ld_mean['facebook'] = 323.989010989011
ld_mean['kin8nm'] = 190.225806451612
ld_mean['life'] = 140.533333333333
ld_mean['meps'] = 751.466666666666
ld_mean['msd'] = 407.736263736263
ld_mean['naval'] = 277.16129032258
ld_mean['news'] = 1902.86666666666
ld_mean['obesity'] = 2320.53333333333
ld_mean['power'] = 222.193548387096
ld_mean['protein'] = 361.813186813186
ld_mean['star'] = 50.16129032258065
ld_mean['superconductor'] = 1020.53333333333
ld_mean['synthetic'] = 480
ld_mean['wave'] = 1382.3333333333333
ld_mean['wine'] = 51.3956043956043
ld_mean['yacht'] = 27.2167487684729

ld_std = {}
ld_std['ames'] = 118.95464108994
ld_std['bike'] = 1141.32608016189
ld_std['california'] = 1879.54437039486
ld_std['communities'] = 68.6011142335858
ld_std['concrete'] = 64.5097160640679
ld_std['energy'] = 18.1715952114117
ld_std['facebook'] = 2441.57204291341
ld_std['kin8nm'] = 716.586938591427
ld_std['life'] = 350.999606394207
ld_std['meps'] = 2223.15002302788
ld_std['msd'] = 1163.22170221396
ld_std['naval'] = 1163.22170221396
ld_std['news'] = 3960.14240697926
ld_std['obesity'] = 7102.92576115567
ld_std['power'] = 895.875225663918
ld_std['protein'] = 2597.94260121134
ld_std['star'] = 72.45865863537838
ld_std['superconductor'] = 1601.17294708875
ld_std['synthetic'] = 786.718625176752
ld_std['wave'] = 3208.4166814939867
ld_std['wine'] = 151.25203390665
ld_std['yacht'] = 5.16833054873821


def process(args, out_dir, logger):

    util.plot_settings(fontsize=11, libertine=True)

    rng = np.random.default_rng(args.random_state)
    color, ls, label = util.get_plot_dicts()
    dataset_map = {'meps': 'MEPS', 'msd': 'MSD', 'star': 'STAR'}

    res_list = []
    for dataset in args.dataset:
        logger.info(f'{dataset}...')

        success = True
        for fold in args.fold:
            res_dict = {'dataset': dataset, 'fold': fold,
                        'ld_mean': ld_mean[dataset], 'ld_std': ld_std[dataset]}
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

    cols = ['dataset', 'fold', 'cnt_tree_mean', 'cnt_tree_std', 'n_train', 'ld_mean', 'ld_std']
    df = pd.DataFrame(res_list)[cols]
    df['cnt_tree_mean_frac'] = df['cnt_tree_mean'] / df['n_train']
    df['cnt_tree_std_frac'] = df['cnt_tree_std'] / df['n_train']
    # df['ld_mean_frac'] = df['ld_mean'] / df['n_train']
    # df['ld_std_frac'] = df['ld_std'] / df['n_train']
    df = df.sort_values('n_train')

    spearman = spearmanr(df['n_train'], df['cnt_tree_mean_frac'])[0]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(df['n_train'] * 100, df['cnt_tree_mean_frac'] * 100,
                yerr=df['cnt_tree_std_frac'] * 100, fmt='o',
                label=f'Spearman={spearman:.3f}', ecolor='k',
                color='blue', lw=1, capsize=2)
    ax.set_ylim(0, 100)
    ax.set_xscale('log')
    ax.set_ylabel('% train visited / tree')
    ax.set_xlabel('No. train instances')
    # ax.legend(loc='upper left')

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
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
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