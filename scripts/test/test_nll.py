import os
import sys
import time
import argparse

import numpy as np
import properscoring as ps
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
from scipy import stats
from scipy.stats import spearmanr
from ngboost import NGBRegressor
from pgbm import PGBMRegressor

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import test_util
from scripts.experiments import util
from intent.estimators import FlexPGBM
from intent.estimators import GBRUT


def compute_nll(method, X_test, y_test, gbrt, X_train, y_train, rng, tau=0.75, m=100):
    """
    Return predictions and 95% CIs for each x in X_test.

    Input
        method: str, uncertainty-estimation method.
        X_test: 2d test data.
        gbrt: Fitted regression model.
        X_train: 2d train data.
        y_train: 1d train labels.
        tau: int, no. models to train ("ensemble" method only).
        m: float, fraction of data to use to train each model ("ensemble" method only).

    Return
        1d array of predictions and 1d array of CIs.
    """

    if method == 'flex_pgbm':
        fpgbm = FlexPGBM().fit(gbrt, X_train, y_train)
        kdes = fpgbm.pred_dist(X_test, tau=0.65, m=100, kde=True)
        nlls = [kde.score_samples(y_test[i].reshape(-1, 1))[0] for i, kde in enumerate(kdes)]
        print(nlls)
        nll = -np.sum(nlls)
        print(nll)
        exit(0)

    elif method == 'ngboost':
        ngb = NGBRegressor().fit(X_train, y_train)
        y_dist = ngb.pred_dist(X_test)
        pred, ci = y_dist.params['loc'], y_dist.params['scale']

    elif method == 'pgbm':
        pgb = PGBMRegressor(max_leaves=31).fit(X_train, y_train)
        y_dist = pgb.predict_dist(X_test, n_forecasts=1000)
        pred, ci = np.mean(y_dist, axis=0), np.std(y_dist, axis=0)
        pred = pgb.predict(X_test)

    result = (pred, ci)

    if include_lw:
        result += (lw,)

    return result


def linear_dataset1(rng, n=100):
    """
    https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b
    """
    X_train = rng.uniform(-3, -2, n)
    y_train = X_train + rng.normal(size=(X_train.shape)) * 0.5

    X_train = np.concatenate([X_train, rng.uniform(2, 3, n)])
    y_train = np.concatenate([y_train, X_train[n:] + rng.normal(size=(X_train[n:].shape)) * 0.1])

    # test
    m = 10

    # X_test = rng.uniform(-5, -3, m)
    # y_test = X_test + rng.normal(size=(X_test.shape))

    X_test = rng.uniform(-3, -2, m)
    y_test = X_test + rng.normal(size=(X_test.shape)) * 0.5

    # X_test = np.concatenate([X_test, rng.uniform(-3, -2, m)])
    # y_test = np.concatenate([y_test, X_test[-m:] + rng.normal(size=(X_test[-m:].shape)) * 0.5])

    # X_test = np.concatenate([X_test, rng.uniform(-2, 2, m)])
    # y_test = np.concatenate([y_test, X_test[-m:] + rng.normal(size=(X_test[-m:].shape))])

    X_test = np.concatenate([X_test, rng.uniform(2, 3, m)])
    y_test = np.concatenate([y_test, X_test[-m:] + rng.normal(size=(X_test[-m:].shape)) * 0.1])

    # X_test = np.concatenate([X_test, rng.uniform(3, 5, m)])
    # y_test = np.concatenate([y_test, X_test[-m:] + rng.normal(size=(X_test[-m:].shape))])s

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def poly_dataset1(rng, n=100, m=10, include_ood=False):
    """
    https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b
    """
    y_func_poly = lambda x: np.sin(x)

    x_train_poly = rng.uniform(-3, -2, n)
    y_train_poly = y_func_poly(x_train_poly) + rng.normal(scale=.2, size=x_train_poly.shape)

    x_train_poly = np.concatenate([x_train_poly, rng.uniform(0, 1, n)])
    y_train_poly = np.concatenate([y_train_poly, y_func_poly(x_train_poly[n:]) +
                                  rng.normal(scale=.1, size=x_train_poly[n:].shape)])

    x_train_poly = np.concatenate([x_train_poly, rng.uniform(2, 3, n)])
    y_train_poly = np.concatenate([y_train_poly, y_func_poly(x_train_poly[-n:]) +
                                   rng.normal(scale=.05, size=x_train_poly[-n:].shape)])

    # test
    if include_ood:
        x_test_poly = np.linspace(-5, 5, m * 10)

    else:
        x_test_poly = np.linspace(-3, -2, m)
        x_test_poly = np.concatenate([x_test_poly, np.linspace(0.1, 1, m)])
        x_test_poly = np.concatenate([x_test_poly, np.linspace(2, 3, m)])

    y_test_poly = y_func_poly(x_test_poly)

    X_train = x_train_poly.reshape(-1, 1)
    X_test = x_test_poly.reshape(-1, 1)

    y_train = y_train_poly
    y_test = y_test_poly

    return X_train, X_test, y_train, y_test


def custom_dataset1(rng, n=1000):
    """
    1d dataset with differing amounts of uncertainty.

    Input
        rng: numpy random number generator.

    Return
        X_train, X_test, y_train.
    """
    y_list = []
    X_list = []

    y_list_test = []
    X_list_test = []

    y_list.append(rng.uniform(0, 50, size=n))  # 1
    y_list.append(rng.uniform(10, 40, size=n))  # 2
    y_list.append(rng.uniform(20, 30, size=n))  # 3
    y_list.append(rng.uniform(25, 25, size=n))  # 4
    y_list.append(rng.uniform(25, 25, size=n))  # 5
    y_list.append(rng.uniform(25, 25, size=n))  # 6
    y_list.append(rng.uniform(20, 50, size=n))  # 7
    y_list.append(rng.uniform(10, 70, size=int(n / 50)))  # 8
    y_list.append(rng.uniform(0, 90, size=n))  # 9
    y_list.append(rng.uniform(0, 100, size=n))  # 10

    X_list.append(rng.uniform(0.5, 1.5, size=n))  # 1
    X_list.append(rng.uniform(1.5, 2.5, size=n))  # 2
    X_list.append(rng.uniform(2.5, 3.5, size=n))  # 3
    X_list.append(rng.uniform(3.5, 4.5, size=n))  # 4
    X_list.append(rng.uniform(4.5, 5.5, size=n))  # 5
    X_list.append(rng.uniform(5.5, 6.5, size=n))  # 6
    X_list.append(rng.uniform(6.5, 7.5, size=n))  # 7
    X_list.append(rng.uniform(7.5, 8.5, size=int(n / 50)))  # 8
    X_list.append(rng.uniform(8.5, 9.5, size=n))  # 9
    X_list.append(rng.uniform(9.5, 10.5, size=n))  # 10

    y_list_test.append(rng.uniform(0, 50, size=10))  # 1
    y_list_test.append(rng.uniform(10, 40, size=10))  # 2
    y_list_test.append(rng.uniform(20, 30, size=10))  # 3
    y_list_test.append(rng.uniform(25, 25, size=10))  # 4
    y_list_test.append(rng.uniform(25, 25, size=10))  # 5
    y_list_test.append(rng.uniform(25, 25, size=10))  # 6
    y_list_test.append(rng.uniform(20, 50, size=10))  # 7
    y_list_test.append(rng.uniform(10, 70, size=10))  # 8
    y_list_test.append(rng.uniform(0, 90, size=10))  # 9
    y_list_test.append(rng.uniform(0, 100, size=10))  # 10

    X_list_test.append(rng.uniform(0.5, 1.5, size=10))  # 1
    X_list_test.append(rng.uniform(1.5, 2.5, size=10))  # 2
    X_list_test.append(rng.uniform(2.5, 3.5, size=10))  # 3
    X_list_test.append(rng.uniform(3.5, 4.5, size=10))  # 4
    X_list_test.append(rng.uniform(4.5, 5.5, size=10))  # 5
    X_list_test.append(rng.uniform(5.5, 6.5, size=10))  # 6
    X_list_test.append(rng.uniform(6.5, 7.5, size=10))  # 7
    X_list_test.append(rng.uniform(7.5, 8.5, size=10))  # 8
    X_list_test.append(rng.uniform(8.5, 9.5, size=10))  # 9
    X_list_test.append(rng.uniform(9.5, 10.5, size=10))  # 10

    X_train = np.concatenate(X_list).reshape(-1, 1)
    y_train = np.concatenate(y_list)

    X_test = np.concatenate(X_list_test).reshape(-1, 1)
    y_test = np.concatenate(y_list_test)

    return X_train, X_test, y_train, y_test


def main(args):

    rng = np.random.default_rng(1)

    n_list = [100, 1000, 10000]
    fig, axs = plt.subplots(1, len(n_list), figsize=(4 * len(n_list), 4))

    for i, n in enumerate(n_list):
        start = time.time()

        # select dataset
        if args.dataset == 'custom1':
            X_train, X_test, y_train, y_test = custom_dataset1(rng, n=n)
        elif args.dataset == 'linear1':
            X_train, X_test, y_train, y_test = linear_dataset1(rng, n=n)
        elif args.dataset == 'poly1':
            X_train, X_test, y_train, y_test = poly_dataset1(rng, n=n, include_ood=args.ood)
        else:
            X_train, X_test, y_train, y_test, objective = util.get_data('data', args.dataset)

        gbrt = test_util._get_model(args)
        gbrt = gbrt.fit(X_train, y_train)
        print('done fitting GBRT...')

        # uncertainty estimation
        pred, ci = compute_nll(args.method, X_test, y_test, gbrt, X_train, y_train, rng,
                               tau=args.tau, m=args.m)

        # print(pred, ci)

        # TEMP: avoid 0 ci
        ci = ci + 1e-8

        crps = np.array([ps.crps_gaussian(y_test[i], mu=pred[i], sig=ci[i]) for i in range(len(X_test))]).mean()
        nll = np.sum([-stats.norm.logpdf(y_test[i], loc=pred[i], scale=ci[i])])
        print(f'CRPS: {crps:.3f}, NLL: {nll:.3f}')

        idxs = np.argsort(X_test.flatten())

        ax = axs[i]
        ax.scatter(X_train, y_train, alpha=0.25, color='gray', label='train', s=1)
        ax.errorbar(X_test.flatten()[idxs], pred[idxs], yerr=ci[idxs], capsize=2, fmt='.',
                    lw=1, color='k', label=f'test, CRPS: {crps: .3f}, NLL: {nll:.3f}')
        # ax.scatter(X_test.flatten()[idxs], y_test[idxs], color='green', label='y_test', s=1)
        # ax.scatter(X_test.flatten()[idxs], pred[idxs] + ci[idxs], color='k', label=r'$\hat{\sigma}$', marker='+', s=1)
        # ax.scatter(X_test.flatten()[idxs], pred[idxs] - ci[idxs], color='k', marker='+', s=1)
        # ax.scatter(X_test.flatten()[idxs], pred[idxs], alpha=0, label=f'CRPS: {crps:.3f}')
        ax.set_title(f'n={n:,} train samples per block')
        ax.set_xlabel(f'$x_1$')

        if args.method == 'dummy':
            ax2 = ax.twinx()
            ax2.plot(X_test.flatten()[idxs], lw[idxs], color='green', label='leaf weight', ls='--', alpha=0.75, lw=1)
            if i == 0:
                ax2.legend()
            if i == 2:
                ax2.set_ylabel('Leaf weight (avg. no. examples)')

        if i == 0:
            ax.set_ylabel(r'y / $\hat{y}$')
        ax.legend()

        print(f'n={n:,}...{time.time() - start:.3f}s')

    out_dir = os.path.join(args.out_dir, args.dataset)

    if args.ood:
        out_dir = os.path.join(args.out_dir, args.dataset, 'test_ood')

    os.makedirs(out_dir, exist_ok=True)

    print(f'\nsaving results to {out_dir}...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.method}.png'), bbox_inches='tight')

    if args.method == 'dummy':
        fig, ax = plt.subplots()
        spearman = spearmanr(ci, lw)[0]
        ax.scatter(ci, lw, label=f'spearman: {spearman:.3f}', s=7.5)
        ax.set_ylabel('Leaf weight (avg. no. examples / leaf)')
        ax.set_xlabel('GBRUT CI')
        ax.set_title('GBRUT vs. Leaf Weight')
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # i/o settings
    parser.add_argument('--out_dir', type=str, default='output/test/test_gbrut1/')

    # data settings
    parser.add_argument('--dataset', type=str, default='custom1')
    parser.add_argument('--ood', action='store_true', default=False)

    # tree-ensemble settings
    parser.add_argument('--n_tree', type=int, default=100)
    parser.add_argument('--n_leaf', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--model_type', type=str, default='regressor')
    parser.add_argument('--rs', type=int, default=1)

    # uncertainty-estimation settings
    parser.add_argument('--method', type=str, default='gbrut')
    parser.add_argument('--tau', type=int, default=0.75)
    parser.add_argument('--m', type=float, default=100)

    args = parser.parse_args()

    main(args)
