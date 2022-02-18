"""
Model performance.
"""
import os
import sys
import time
import ngboost
import resource
import argparse
import warnings
import itertools
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import pandas as pd
import seaborn as sns
import properscoring as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.scores import CRPScore
from ngboost.scores import LogScore
from scipy.stats import gaussian_kde
from scipy.stats import lognorm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for ibug
import util
from ibug import IBUGWrapper
from ibug import KNNWrapper
from train import get_loc_scale


def eval_posterior(model, X, y, min_scale, loc, scale, distribution_list,
                   fixed_params, random_state=1, logger=None, prefix=''):
    """
    Model the posterior using different parametric and non-parametric
        distributions and evaluate their predictive performance.

    Input
        model: IBUG, Fitted model.
        X: np.ndarray, 2d array of input data.
        y: np.ndarray, 1d array of ground-truth targets.
        loc: np.ndarray, 1d array of location values.
        scale: np.ndarray, 1d array of scale values.
        min_scale: float, Minimum scale value.
        distribution_list: list, List of distributions to try.
        fixed_params: str, Selects which parameters of the distribution to keep fixed.
        random_state: int, Random seed.
        logger: logging object, Log updates.
        prefix: str, Used to identify updates.

    Return
        3-tuple including a dict of CRPS and NLL results, and 2 2d
            arrays of neighbor indices and target values, shape=(no. test, k).
    """
    assert 'ibug' in str(model).lower()
    assert X.ndim == 2 and y.ndim == 1 and loc.ndim == 1 and scale.ndim == 1
    assert X.shape[0] == len(y) == len(loc) == len(scale)

    if logger:
        logger.info(f'\n[{prefix}] Computing k-nearest neighbors...')

    start = time.time()
    loc, scale, neighbor_idxs, neighbor_vals = model.pred_dist(X, return_kneighbors=True)
    if logger:
        logger.info(f'time: {time.time() - start:.3f}s')
        logger.info(f'\n[{prefix}] Evaluating distributions...')

    start = time.time()
    dist_res = {'nll': {}, 'crps': {}}

    for dist in distribution_list:

        # select fixed parameters
        if fixed_params == 'fl_dist':
            loc_copy, scale_copy = loc.copy(), None

        elif fixed_params == 'fls_dist':
            loc_copy, scale_copy = loc.copy(), scale.copy()

        else:
            loc_copy, scale_copy = None, None

        nll, crps = util.eval_dist(y=y, samples=neighbor_vals.copy(), dist=dist, nll=True,
                                   crps=True, min_scale=min_scale, random_state=random_state,
                                   loc=loc_copy, scale=scale_copy)

        if logger:
            logger.info(f'[{prefix} - {dist}] CRPS: {crps:.5f}, NLL: {nll:.5f}')

        dist_res['nll'][dist] = nll
        dist_res['crps'][dist] = crps

    if logger:
        logger.info(f'[{prefix} time: {time.time() - start:.3f}s')

    # assemble output
    result = {'performance': dist_res, 'neighbors': {'idxs': neighbor_idxs, 'y_vals': neighbor_vals}}
    return result


def calibrate_variance(scale_arr, delta, op):
    """
    Add or multiply all values in `scale_arr` by `delta`.

    Input
        scale_arr: np.ndarray, 1d array of scale values.
        delta: float, Value to add or multiply.
        op: str, Operation to perform on `scale_arr`.

    Return
        1d array of calibrated scale values.
    """
    assert scale_arr.ndim == 1
    assert op in ['add', 'mult']
    result = np.array(scale_arr)
    if op == 'add':
        result = result + delta
    else:
        result = result * delta
    return result


def experiment(args, in_dir, out_dir, logger):
    """
    Main method comparing performance of tree ensembles and svm models.
    """
    begin = time.time()  # experiment timer
    rng = np.random.default_rng(args.random_state)  # pseduo-random number generator

    # load results and models from training
    result = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
    model_val = util.load_model(model_type=args.model_type, fp=result['saved_models']['model_val'])
    model_test = util.load_model(model_type=args.model_type, fp=result['saved_models']['model_test'])

    # additional settings
    if args.model_type == 'ibug':
        model_test.set_tree_subsampling(frac=args.tree_subsample_frac, order=args.tree_subsample_order)
        model_test.set_instance_subsampling(frac=args.instance_subsample_frac)

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset, args.fold)
    val_idxs = result['data']['val_idxs']
    tune_idxs = result['data']['tune_idxs']
    X_val, y_val = X_train[val_idxs].copy(), y_train[val_idxs].copy()
    X_tune, y_tune = X_train[tune_idxs].copy(), y_train[tune_idxs].copy()

    logger.info('\nno. train: {:,}'.format(X_train.shape[0]))
    logger.info('  -> no. val.: {:,}'.format(X_val.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))
    logger.info(f"\ndelta: {result['delta']['best_delta']}, 'op': {result['delta']['best_op']}")

    # validation: predict
    logger.info(f'\nVALIDATION SET\n')
    start = time.time()
    loc_val, scale_val = get_loc_scale(model=model_val, model_type=args.model_type, X=X_val, y_train=y_tune)
    val_pred_time = time.time() - start
    logger.info(f'predicting...{time.time() - start:.3f}s')

    # validation: evaluate point performance
    start = time.time()
    rmse_val, mae_val = util.eval_pred(y_val, model=model_val, X=X_val, logger=None, prefix='VAL')
    val_eval_point_time = time.time() - start
    logger.info(f'evaluating point performance...{time.time() - start:.3f}s')

    # validation: evaluate probabilistic performance (w/o delta)
    start = time.time()
    nll_val1, crps_val1 = util.eval_normal(y=y_val, loc=loc_val, scale=scale_val, nll=True, crps=True)
    val_eval_prob_time = time.time() - start
    logger.info(f'evaluating probabilistic performance...{time.time() - start:.3f}s')

    # validation: calibrate variance
    best_delta, best_op = result['delta']['best_delta'], result['delta']['best_op']
    scale_val_delta = calibrate_variance(scale_val, delta=best_delta, op=best_op)

    # validation: evaluate probabilistic performance (w/ delta)
    start = time.time()
    nll_val2, crps_val2 = util.eval_normal(y=y_val, loc=loc_val, scale=scale_val_delta, nll=True, crps=True)
    logger.info(f'evaluating probabilistic performance (w/ delta)...{time.time() - start:.3f}s')

    logger.info(f'\nRMSE: {rmse_val:.5f}, MAE: {mae_val:.5f}')
    logger.info(f'CRPS: {crps_val1:.5f}, NLL: {nll_val1:.5f}')
    logger.info(f'CRPS: {crps_val2:.5f}, NLL: {nll_val2:.5f} (w/ delta)')

    # test: predict
    logger.info(f'\nTEST SET\n')
    start = time.time()
    loc_test, scale_test = get_loc_scale(model=model_test, model_type=args.model_type, X=X_test, y_train=y_train)
    test_pred_time = time.time() - start
    logger.info(f'predicting...{time.time() - start:.3f}s')

    # test: evaluate point performance
    start = time.time()
    rmse_test, mae_test = util.eval_pred(y_test, model=model_test, X=X_test, logger=None, prefix='TEST')
    test_eval_point_time = time.time() - start
    logger.info(f'evaluating point performance...{time.time() - start:.3f}s')

    # test: evaluate probabilistic performance
    start = time.time()
    nll_test1, crps_test1 = util.eval_normal(y=y_test, loc=loc_test, scale=scale_test, nll=True, crps=True)
    test_eval_prob_time = time.time() - start
    logger.info(f'evaluating probabilistic performance...{time.time() - start:.3f}s')

    # test: calibrate variance
    scale_test_delta = calibrate_variance(scale_test, delta=best_delta, op=best_op)

    # test: evaluate probabilistic performance
    start = time.time()
    nll_test2, crps_test2 = util.eval_normal(y=y_test, loc=loc_test, scale=scale_test_delta, nll=True, crps=True)
    logger.info(f'evaluating probabilistic performance (w/ delta)...{time.time() - start:.3f}s')

    logger.info(f'\nRMSE: {rmse_test:.5f}, MAE: {mae_test:.5f}')
    logger.info(f'CRPS: {crps_test1:.5f}, NLL: {nll_test1:.5f}')
    logger.info(f'CRPS: {crps_test2:.5f}, NLL: {nll_test2:.5f} (w/ delta)')

    # plot predictions
    test_idxs = np.argsort(y_test)

    fig, ax = plt.subplots()
    x = np.arange(len(test_idxs))
    ax.plot(x, y_test[test_idxs], ls='--', color='orange', label='actual', zorder=1)
    ax.errorbar(x, loc_test[test_idxs], yerr=scale_test[test_idxs], fmt='.',
                color='green', label='prediction', lw=1, zorder=0)
    ax.set_title(f'[w/ Delta] CRPS: {crps_test2:.3f}, NLL: {nll_test2:.3f}, RMSE: {rmse_test:.3f}')
    ax.set_xlabel('Test index (sorted by output)')
    ax.set_ylabel('Output')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'test_pred.png'), bbox_inches='tight')

    # save results
    del result['data']['tune_idxs']
    del result['data']['val_idxs']
    result['predict_args'] = vars(args)
    result['timing'].update({'val_pred_time': val_pred_time,
                             'val_eval_point_time': val_eval_point_time,
                             'val_eval_prob_time': val_eval_prob_time,
                             'test_pred_time': test_pred_time,
                             'test_eval_point_time': test_eval_point_time,
                             'test_eval_prob_time': test_eval_prob_time})
    result['val_performance'] = {'rmse': rmse_val,
                                 'mae': mae_val,
                                 'nll': nll_val1,
                                 'crps': crps_val1,
                                 'nll_delta': nll_val2,
                                 'crps_delta': crps_val2}
    result['test_performance'] = {'rmse': rmse_test,
                                  'mae': mae_test,
                                  'nll': nll_test1,
                                  'crps': crps_test1,
                                  'nll_delta': nll_test2,
                                  'crps_delta': crps_test2}
    result['misc'] = {'max_RSS': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,  # MB if OSX, GB if Linux
                      'total_experiment_time': time.time() - begin}
    if args.model_type == 'ibug':

        # collecting statistics
        logger.info('\n\nLEAF STATISTICS')
        start = time.time()
        result['affinity_count'] = model_test.get_affinity_stats(X_test)
        result['leaf_density'] = model_test.get_leaf_stats()
        logger.info(f'time: {time.time() - start:.3f}s')

        # posterior modeling
        if args.custom_out_dir in ['dist', 'fl_dist', 'fls_dist']:
            logger.info('\n\nPOSTERIOR MODELING')
            val_res = eval_posterior(model=model_val, X=X_val, y=y_val, min_scale=model_val.min_scale_,
                                     loc=loc_val, scale=scale_val_delta, distribution_list=args.distribution,
                                     fixed_params=args.custom_out_dir, random_state=args.random_state,
                                     logger=logger, prefix='VAL')
            test_res = eval_posterior(model=model_test, X=X_test, y=y_test, min_scale=model_test.min_scale_,
                                      loc=loc_test, scale=scale_test_delta, distribution_list=args.distribution,
                                      fixed_params=args.custom_out_dir, random_state=args.random_state,
                                      logger=logger, prefix='TEST')
            result['val_posterior'] = val_res['performance']
            result['test_posterior'] = test_res['performance']
            if args.fold == 1:
                result['neighbors'] = test_res['neighbors']

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['misc']['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB): {result['misc']['max_RSS']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results and models to {os.path.join(out_dir, 'results.npy')}")

    # save results/models
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # define input directory
    train_args = vars(args).copy()
    train_args['tree_subsample_frac'] = 1.0
    train_args['tree_subsample_order'] = 'random'
    train_args['instance_subsample_frac'] = 1.0
    method_name = util.get_method_identifier(args.model_type, train_args)

    in_dir = os.path.join(args.in_dir,
                          args.custom_in_dir,
                          args.dataset,
                          args.scoring,
                          f'fold{args.fold}',
                          method_name)

    # define output directory
    method_name = util.get_method_identifier(args.model_type, vars(args))

    out_dir = os.path.join(args.out_dir,
                           args.custom_out_dir,
                           args.dataset,
                           args.scoring,
                           f'fold{args.fold}',
                           method_name)

    # create outut directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    experiment(args, in_dir, out_dir, logger)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='output/experiments/train/')
    parser.add_argument('--out_dir', type=str, default='output/experiments/predict')
    parser.add_argument('--custom_in_dir', type=str, default='default')
    parser.add_argument('--custom_out_dir', type=str, default='default')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='ibug')

    # Method settings
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, IBUG, PGBM
    parser.add_argument('--tree_type', type=str, default='lgb')  # IBUG, constant
    parser.add_argument('--tree_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--tree_subsample_order', type=str, default='random')  # IBUG
    parser.add_argument('--instance_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--affinity', type=str, default='unweighted')  # IBUG

    # Default settings
    parser.add_argument('--tune_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--val_frac', type=float, default=0.2)  # ALL
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--verbose', type=int, default=2)  # ALL
    parser.add_argument('--n_stopping_rounds', type=int, default=25)  # NGBoost, PGBM, IBUG, constant
    parser.add_argument('--scoring', type=str, default='nll')
    parser.add_argument('--distribution', type=str, nargs='+',
                        default=['normal', 'skewnormal', 'lognormal', 'laplace',
                                 'student_t', 'logistic', 'gumbel', 'weibull', 'kde'])

    args = parser.parse_args()
    main(args)
