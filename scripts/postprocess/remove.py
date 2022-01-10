"""
Plot results for a single dataset.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from config import post_args


def process(args, exp_hash, out_dir, logger):

    color, line, label, marker = util.get_plot_dicts(markers=True)

    n_test = None

    # get dataset
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

    # get results
    exp_dir = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}')

    results = util.get_results(args, exp_dir, logger)
    results = util.filter_results(results, args.skip)

    results_dict = {label[method]: (method, res) for method, res in results}

    order = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'LeafRefit',
             'LeafInfluence', 'SubSample', 'LOO', 'Random', 'RandomSL']

    util.plot_settings(fontsize=23)

    fig, ax = plt.subplots()

    for i, key in enumerate(order):

        if key not in results_dict:
            continue

        method, res = results_dict[key]

        # sanity check
        if i == 0:
            n_test = res['loss'].shape[0]

        else:
            temp = res['loss'].shape[0]
            assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

        # plot loss
        x = res['remove_frac'] * 100
        y = res['loss'].mean(axis=0)
        y_err = sem(res['loss'], axis=0)
        y_err = y_err if args.std_err else None

        ax.errorbar(x, y, yerr=y_err, label=label[method], color=color[method],
                    linestyle=line[method], marker=marker[method], alpha=0.75)
        ax.set_xlabel('Train data removed (%)')
        ax.set_ylabel(f'Average test loss')

        if args.legend:
            ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.pdf'), bbox_inches='tight')

    logger.info(f'\nSaving results to {out_dir}/...')


def main(args):

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir, args.tree_type, f'exp_{exp_hash}', 'postprocess')
    log_dir = os.path.join(out_dir, 'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_remove_args().parse_args())
