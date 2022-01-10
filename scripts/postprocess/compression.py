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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from config import post_args


def process(args, out_dir, logger):

    # get results
    exp_dir = os.path.join(args.in_dir,
                           args.tree_type,
                           args.dataset)

    results = util.get_results(args, exp_dir, logger)

    # plot
    color, ls, label = util.get_plot_dicts()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for method, res in results:

        ax = axs[0]
        ax.plot(res['pct'], res['loss'], label=label[method], color=color[method], ls=ls[method])
        ax.set_xlabel('Train data removed (%)')
        ax.set_ylabel('Test loss')

        ax = axs[1]
        ax.plot(res['pct'], res['acc'], label=label[method], color=color[method], ls=ls[method])
        ax.set_xlabel('Train data removed (%)')
        ax.set_ylabel('Test accuracy')

        ax = axs[2]
        ax.plot(res['pct'], res['auc'], label=label[method], color=color[method], ls=ls[method])
        ax.set_xlabel('Train data removed (%)')
        ax.set_ylabel('Test AUC')

    axs[0].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')

    logger.info(f'\nSaving results to {out_dir}/...')


def main(args):

    out_dir = os.path.join(args.out_dir, args.tree_type)
    log_dir = os.path.join(out_dir, 'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_compression_args().parse_args())
