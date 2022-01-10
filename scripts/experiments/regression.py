"""
Remove train data and measure loss.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
import properscoring as ps
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from pgbm import PGBMRegressor
from scipy import stats

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # intent
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
from intent import FlexPGBM
import util


def compute_loc_scale(method, X_test, tree, X_train, y_train, k=50, loc_type='neighbors'):
    """
    Return predictions and 95% CIs for each x in X_test.

    Input
        method: str, uncertainty-estimation method.
        X_test: 2d test data.
        tree: Fitted regression model.
        X_train: 2d train data.
        y_train: 1d train labels.
        k: int, k-nearest neighbors ("flex_pgbm" method).

    Return
        1d array of predictions and 1d array of CIs.
    """
    if method == 'flex_pgbm':
        fpgbm = FlexPGBM(k=k, loc_type=loc_type).fit(tree, X_train, y_train)
        loc, scale = fpgbm.pred_dist(X_test)

    elif method == 'ngboost':
        ngb = NGBRegressor().fit(X_train, y_train)
        y_dist = ngb.pred_dist(X_test)
        loc, scale = y_dist.params['loc'], y_dist.params['scale']

    elif method == 'pgbm':
        pgb = PGBMRegressor(max_leaves=31).fit(X_train, y_train)
        y_dist = pgb.predict_dist(X_test, n_forecasts=1000)
        loc, scale = np.mean(y_dist, axis=0), np.std(y_dist, axis=0)

    return loc, scale


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)

    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # predict w/ uncertainty
    loc, scale = compute_loc_scale(args.method, X_test, tree, X_train, y_train, k=args.k, loc_type=args.loc_type)

    # evaluate
    crps = np.mean([ps.crps_gaussian(y_test[i], mu=loc[i], sig=scale[i]) for i in range(len(X_test))])
    nll = np.mean([-stats.norm.logpdf(y_test[i], loc=loc[i], scale=scale[i]) for i in range(len(X_test))])
    rmse = mean_squared_error(y_test, loc, squared=False)
    print(f'CRPS: {crps:.5f}, NLL: {nll:.5f}, RMSE: {rmse:.5f}')

    # plot predictions
    test_idxs = np.argsort(y_test)

    fig, ax = plt.subplots()
    x = np.arange(len(test_idxs))
    ax.plot(x, y_test[test_idxs], ls='--', color='orange', label='actual')
    ax.fill_between(x, loc[test_idxs] - scale[test_idxs], loc[test_idxs] + scale[test_idxs],
                    label='uncertainty', alpha=0.75)
    ax.set_title(f'CRPS: {crps:.3f}, NLL: {nll:.3f}, RMSE: {rmse:.3f}')
    ax.set_xlabel('Test index (sorted by output)')
    ax.set_ylabel('Output')
    ax.legend()

    # save
    method_name = f'{args.method}'

    if args.method == 'flex_pgbm':
        method_name += f'_{args.loc_type}_{args.k}'

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{method_name}.pdf'), bbox_inches='tight')

    logger.info(f'\nsaving results to {out_dir}...')


def main(args):

    out_dir = os.path.join(args.out_dir, args.dataset)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(out_dir, 'z_log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # i/o settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/regression/')

    # data settings
    parser.add_argument('--dataset', type=str, default='concrete')

    # tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--random_state', type=int, default=1)

    # uncertainty-estimation settings
    parser.add_argument('--method', type=str, default='flex_pgbm')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--loc_type', type=str, default='knn')

    args = parser.parse_args()

    main(args)
