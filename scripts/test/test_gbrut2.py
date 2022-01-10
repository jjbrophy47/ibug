"""
Compute correlation between uncertainties and error.
"""
import os
import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy.stats import spearmanr

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import test_util
from scripts.experiments import util
from intent.estimators import GBRUT
from test_gbrut1 import estimate_uncertainty


def main(args):
    print(f'\n***** test_GBRUT2 *****')

    rng = np.random.default_rng()

    X_train, X_test, y_train, y_test, objective = util.get_data('data', args.dataset)

    # hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    # gbrt = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    # gbrt.set_params(**hp)

    gbrt = test_util._get_model(args)
    gbrt = gbrt.fit(X_train, y_train)

    # uncertainty estimation
    start = time.time()
    pred1, ci1, lw1 = estimate_uncertainty('gbrut', X_test, gbrt, X_train, y_train, rng, include_lw=True)
    print(f'GBRUT...{time.time() - start:.3f}s')

    start = time.time()
    pred2, ci2 = estimate_uncertainty('ensemble', X_test, gbrt, X_train, y_train, rng, tau=args.tau)
    print(f'Ensemble...{time.time() - start:.3f}s')

    error1 = (pred1 - y_test) ** 2
    error2 = (pred2 - y_test) ** 2

    spearman1 = spearmanr(ci1, error1)[0]
    spearman2 = spearmanr(ci2, error2)[0]
    spearman3 = spearmanr(ci1, ci2)[0]
    spearman4 = spearmanr(ci1, lw1)[0]

    # plot
    n_col = 3
    fig, axs = plt.subplots(1, n_col, figsize=(4 * n_col, 4))

    ax = axs[0]
    ax.scatter(ci1, error1, label=f'spearman: {spearman1:.3f}', s=2)
    ax.set_title(f'GBRUT vs. Error')
    ax.set_xlabel(f'GBRUT CI')
    ax.set_ylabel(f'Squared error')
    ax.legend()

    ax = axs[1]
    ax.scatter(ci2, error2, label=f'spearman: {spearman2:.3f}', s=2)
    ax.set_title(f'Ensemble vs. Error')
    ax.set_xlabel(f'Ensemble CI')
    ax.legend()

    ax = axs[2]
    ax.scatter(ci1, ci2, label=f'spearman: {spearman3:.3f}', s=2)
    ax.set_title(f'GBRUT vs. Ensemble')
    ax.set_xlabel(f'GBRUT CI')
    ax.set_ylabel(f'Ensemble CI')
    ax.legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('RdYlGn_r')
    sc = ax.scatter(ci1, lw1, label=f'spearman: {spearman4:.3f}', s=7.5, c=error1, cmap=cm, alpha=0.75)
    ax.set_ylabel('Leaf weight (avg. no. examples / leaf)')
    ax.set_xlabel('GBRUT CI')
    ax.set_title('GBRUT vs. Leaf Weight')
    ax.legend()
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--n_class', type=int, default=-1)
    parser.add_argument('--n_feat', type=int, default=10)

    # tree-ensemble settings
    parser.add_argument('--n_tree', type=int, default=100)
    parser.add_argument('--n_leaf', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--model_type', type=str, default='regressor')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--rs', type=int, default=1)

    # uncertainty-estimation settings
    parser.add_argument('--method', type=str, default='gbrut')
    parser.add_argument('--tau', type=int, default=10)
    parser.add_argument('--m', type=float, default=0.7)

    args = parser.parse_args()

    main(args)
