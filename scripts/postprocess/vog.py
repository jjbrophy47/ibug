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
from scipy.stats import spearmanr
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from config import post_args


def process(args, out_dir, logger):

    # load data
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

    # get results
    exp_dir = os.path.join(args.in_dir,
                           args.tree_type,
                           args.dataset)

    fp = os.path.join(exp_dir, 'results.npy')
    assert os.path.exists(os.path.join(exp_dir, 'results.npy'))

    res = np.load(os.path.join(exp_dir, 'results.npy'), allow_pickle=True)[()]
    predictions = res['predictions']  # shape=(no. train, no. boost, no. class)
    gradients = res['gradients']  # shape=(no. train, no. boost, no. class)
    losses = res['losses']  # shape=(no. train, no. boost)

    # compute statistics
    confidence = np.mean(predictions, axis=1)  # shape=(no. train, no. class)
    variability = np.std(predictions, axis=1)  # shape=(no. train, no. class)
    vog = np.var(gradients, axis=1)  # shape=(no. train, no. class)
    mean_loss = np.mean(losses, axis=1)  # shape=(no. train,)

    # compute statistics for 2nd half of training
    n_boost = predictions.shape[1]
    n_start = int(n_boost / 2)

    confidence_b = np.mean(predictions[:, n_start:], axis=1)  # shape=(no. train, no. class)
    variability_b = np.std(predictions[:, n_start:], axis=1)  # shape=(no. train, no. class)
    vog_b = np.var(gradients[:, n_start:], axis=1)  # shape=(no. train, no. class)
    mean_loss_b = np.mean(losses[:, n_start:], axis=1)  # shape=(no. train,)

    # reshape confidence and variability for binary tasks
    if objective == 'binary':

        confidence = np.vstack([1 - confidence[:, 0], confidence[:, 0]]).T  # shape=(no. train, 2)
        variability = np.vstack([variability[:, 0], variability[:, 0]]).T  # shape=(no. train, 2)
        vog = np.vstack([vog[:, 0], vog[:, 0]]).T  # shape=(no. train, 2)

        confidence_b = np.vstack([1 - confidence_b[:, 0], confidence_b[:, 0]]).T  # shape=(no. train, 2)
        variability_b = np.vstack([variability_b[:, 0], variability_b[:, 0]]).T  # shape=(no. train, 2)
        vog_b = np.vstack([vog_b[:, 0], vog_b[:, 0]]).T  # shape=(no. train, 2)

    # plot settings
    colors = ['g', 'r', 'b', 'green', 'yellow', 'orange', 'brown']
    markers = ['.', 'x', 'd', '*', '+', 'o', '<']

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # confidence vs. variability
    ax = axs[0][0]

    if objective in ['binary', 'multiclass']:

        for i, label in enumerate(np.unique(y_train)):
            idxs = np.where(y_train == label)[0]

            ax.scatter(variability[idxs, label], confidence[idxs, label], s=1,
                       label=f'class {label}: {len(idxs):,}', color=colors[i], marker=markers[i])

        ax.set_title('Confidence vs. Variability\n(all boosting iterations)')
        ax.set_ylabel('Confidence (avg. prob.)')
        ax.legend()

    else:
        spearman = spearmanr(variability[:, 0], mean_loss)[0]
        ax.scatter(variability[:, 0], mean_loss, s=1, label=f'spearman: {spearman:.3f}')
        ax.set_title('Error vs. Variability\n(all boosting iterations)')
        ax.set_ylabel('Error (avg. squared error)')
        ax.legend()

    ax.set_xlabel('Variability (s.d. prob.)')

    # confidence vs. variability (second 1/2 training)
    ax = axs[0][1]

    if objective in ['binary', 'multiclass']:

        for i, label in enumerate(np.unique(y_train)):
            idxs = np.where(y_train == label)[0]

            ax.scatter(variability_b[idxs, label], confidence_b[idxs, label], s=1,
                       label=f'class {label}: {len(idxs):,}', color=colors[i], marker=markers[i])

        ax.set_title('Confidence vs. Variability\n(second half of training)')
        ax.set_ylabel('Confidence (avg. prob.)')

    else:
        spearman = spearmanr(variability_b[:, 0], mean_loss_b)[0]
        ax.scatter(variability_b[:, 0], mean_loss_b, s=1, label=f'spearman: {spearman:.3f}')
        ax.set_title('Error vs. Variability\n(second half of training)')
        ax.set_ylabel('Error (avg. squared error)')
        ax.legend()

    ax.set_xlabel('Variability (s.d. prob.)')

    # VOG vs. variability
    ax = axs[0][2]
    if objective in ['binary', 'multiclass']:

        for i, label in enumerate(np.unique(y_train)):
            idxs = np.where(y_train == label)[0]

            spearman = spearmanr(variability[idxs, label], vog[idxs, label])[0]
            ax.scatter(variability[idxs, label], vog[idxs, label], s=1,
                       label=f'class {label}: {spearman:.3f}', color=colors[i], marker=markers[i])

            ax.legend(title='Spearman')
    else:
        spearman = spearmanr(variability[:, 0], vog[:, 0])[0]
        ax.scatter(variability[:, 0], vog[:, 0], s=1, label=f'spearman: {spearman:.3f}')
        ax.legend()

    ax.set_title('VOG vs. Variability')
    ax.set_xlabel('Variability (s.d. prob.)')
    ax.set_ylabel('VOG (variance of gradients)')

    # confidence vs. VOG
    ax = axs[1][0]

    if objective in ['binary', 'multiclass']:

        for i, label in enumerate(np.unique(y_train)):
            idxs = np.where(y_train == label)[0]

            ax.scatter(vog[idxs, label], confidence[idxs, label], s=1,
                       label=f'class {label}: {len(idxs):,}', color=colors[i], marker=markers[i])

        ax.set_title('Confidence vs. VOG\n(all boosting iterations)')
        ax.set_ylabel('Confidence (avg. prob.)')

    else:
        spearman = spearmanr(vog[:, 0], mean_loss)[0]
        ax.scatter(vog[:, 0], mean_loss, s=1, label=f'spearman: {spearman:.3f}')
        ax.set_title('Error vs. VOG\n(all boosting iterations)')
        ax.set_ylabel('Error (avg. squared error)')
        ax.legend()

    ax.set_xlabel('VOG (var. gradients)')

    # confidence vs. VOG (second 1/2 training)
    ax = axs[1][1]

    if objective in ['binary', 'multiclass']:

        for i, label in enumerate(np.unique(y_train)):
            idxs = np.where(y_train == label)[0]

            ax.scatter(vog_b[idxs, label], confidence_b[idxs, label], s=1,
                       label=f'class {label}: {len(idxs):,}', color=colors[i], marker=markers[i])

        ax.set_title('Confidence vs. VOG\n(second half of training)')
        ax.set_ylabel('Confidence (avg. prob.)')

    else:
        spearman = spearmanr(vog_b[:, 0], mean_loss_b)[0]
        ax.scatter(vog_b[:, 0], mean_loss_b, s=1, label=f'spearman: {spearman:.3f}')
        ax.set_title('Error vs. VOG\n(second half of training)')
        ax.set_ylabel('Error (avg. squared error)')
        ax.legend()

    ax.set_xlabel('VOG (var. gradients)')

    fig.delaxes(ax=axs[1][2])

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
    main(post_args.get_vog_args().parse_args())
