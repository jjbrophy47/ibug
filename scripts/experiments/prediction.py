"""
Model performance.
"""
import os
import sys
import time
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
from scipy.stats import gaussian_kde
from scipy.stats import lognorm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import util
from kgbm import KGBMWrapper

# constants
EPSILON = 1e-15


def tune_delta(loc, scale, y, ops=['add', 'mult'],
               delta_vals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                           1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
               scoring='nll', verbose=0, logger=None):
    """
    Add or multiply detla to scale values.

    Input
        scale: 1d array of scale values.
        y: 1d array of target values (same shape as scale).
        op: str, operation to perform to scale array.

    Return
        1d array of updated scale values.
    """
    assert scoring == 'nll'

    results = []
    for op in ops:
        for delta in delta_vals:

            if op == 'mult' and delta == 0.0:
                continue

            if op == 'add':
                temp_scale = scale + delta
            else:
                temp_scale = scale * delta

            nll = util.eval_normal(y=y, loc=loc, scale=temp_scale, nll=True, crps=False)
            results.append({'delta': delta, 'op': op, 'score': nll})

    df = pd.DataFrame(results).sort_values('score', ascending=True)

    best_delta = df.iloc[0]['delta']
    best_op = df.iloc[0]['op']

    if verbose > 0:
        if logger:
            logger.info(f'\ndelta gridsearch:\n{df}')
        else:
            print(f'\ndelta gridsearch:\n{df}')

    return best_delta, best_op


def get_loc_scale(model, model_type, X, y=None, min_scale=None,
                  verbose=0, logger=None):
    """
    Predict location and scale for each x in X.

    Input
        model: object, uncertainty estimator.
        model_type: str, type of uncertainty estimator.
        X: 2d array of input data.
        y: 1d array of targets (constant and KNN methods only).

    Return
        Tuple, 2 1d arrays of locations and scales.
    """
    if model_type == 'constant':
        assert y is not None
        loc = model.predict(X)
        scale = np.full(len(X), np.std(y), dtype=np.float32)

    elif model_type == 'kgbm':
        loc, scale = model.pred_dist(X)

    elif model_type == 'knn':
        assert y is not None

        loc = np.zeros(len(X), dtype=np.float32)
        scale = np.zeros(len(X), dtype=np.float32)

        for i in range(len(X)):
            train_idxs = model.kneighbors(X[[i]], return_distance=False).flatten()  # shape=(len(X),)
            loc[i] = np.mean(y[train_idxs])
            scale[i] = np.std(y[train_idxs]) + EPSILON  # avoid 0 scale

            if min_scale is not None and scale[i] < min_scale:
                scale[i] = min_scale

            if (i + 1) % 100 == 0 and args.verbose > 0:
                if logger:
                    logger.info(f'[KNN - predict]: {i + 1:,} / {len(X):,}')
                else:
                    print(f'[KNN - predict]: {i + 1:,} / {len(X):,}')

    elif model_type == 'pgbm':
        _, loc, variance = model.learner_.predict_dist(X.astype(np.float32),
                                                       output_sample_statistics=True)
        scale = np.sqrt(variance)

    elif model_type == 'ngboost':
        y_dist = model.pred_dist(X)
        loc, scale = y_dist.params['loc'], y_dist.params['scale']

    return loc, scale


def product_dict(**kwargs):
    """
    Return cross-product iterable of dicts given
    a set of named lists.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_model(args, n_train):
    """
    Return the appropriate classifier.

    Input
        args: object, script arguments.
        n_train: int, no. training examples.
    """
    if args.model == 'ngboost':
        model = NGBRegressor(n_estimators=2000, verbose=args.verbose)
        params = {'n_estimators': [10, 25, 50, 100, 200, 500, 1000]}

    elif args.model == 'pgbm':
        from pgbm import PGBMRegressor  # dynamic import (some machines cannot install pgbm)
        model = PGBMRegressor(n_estimators=2000, learning_rate=0.1, max_leaves=16,
                              max_bin=64, verbose=args.verbose + 1)
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                  'max_leaves': [15, 31, 61, 91],
                  'learning_rate': [0.01, 0.1],
                  'max_bin': [255]}

    elif args.model == 'knn':
        model = KNeighborsRegressor(weights=args.weights)
        params = {'n_neighbors': [3, 5, 7, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701]}
        params['n_neighbors'] = [k for k in params['n_neighbors'] if k <= n_train]

    elif args.model in ['constant', 'kgbm']:

        if args.tree_type == 'lgb':
            model = LGBMRegressor(n_estimators=2000, learning_rate=0.1, max_depth=-1,
                                  num_leaves=16, max_bin=64, random_state=args.random_state)
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'num_leaves': [15, 31, 61, 91],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}

        elif args.tree_type == 'xgb':
            model = XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.1,
                                 max_bin=64, random_state=args.random_state)
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}

        elif args.tree_type == 'cb':
            model = CatBoostRegressor(n_estimators=2000, max_depth=4, max_bin=64,
                                      random_state=args.random_state,
                                      logging_level='Silent', learning_rate=0.1)
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}

        else:
            raise ValueError('tree_type unknown: {}'.format(args.tree_type))

    else:
        raise ValueError('model unknown: {}'.format(args.model))

    return model, params


def experiment(args, logger, out_dir):
    """
    Main method comparing performance of tree ensembles and svm models.
    """
    begin = time.time()  # experiment timer
    rng = np.random.default_rng(args.random_state)  # pseduo-random number generator

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset, args.fold)

    if args.tune_frac < 1.0:
        assert args.tune_frac > 0.0
        n_tune = int(len(X_train) * args.tune_frac)
        tune_idxs = rng.choice(np.arange(len(X_train)), size=n_tune, replace=False)
        X_tune = X_train[tune_idxs].copy()
        y_tune = y_train[tune_idxs].copy()
    else:
        X_tune = X_train.copy()
        y_tune = y_train.copy()

    X_tune, X_val, y_tune, y_val = train_test_split(X_tune, y_tune,
                                                    test_size=args.val_frac,
                                                    random_state=args.random_state)

    logger.info('no. train: {:,}'.format(X_train.shape[0]))
    logger.info('  -> no. tune: {:,}'.format(X_tune.shape[0]))
    logger.info('  -> no. val.: {:,}'.format(X_val.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # tune
    logger.info('\nTuning model...')
    start = time.time()

    model, param_grid = get_model(args, n_train=len(X_train))

    if args.model == 'knn' or (args.model in ['constant', 'kgbm', 'pgbm'] and args.gridsearch):
        logger.info('\nmodel: {}, param_grid: {}'.format(args.model, param_grid))

        cv_results = []
        param_dicts = list(product_dict(**param_grid))

        # gridsearch
        best_score = None
        best_model = None
        best_params = None

        for i, param_dict in enumerate(param_dicts):
            temp_model = clone(model).set_params(**param_dict).fit(X_tune, y_tune)
            y_val_hat = temp_model.predict(X_val)
            param_dict['score'] = mean_squared_error(y_val, y_val_hat)

            logger.info(f'[{i + 1:,}/{len(param_dicts):,}] {param_dict}'
                        f', cum. time: {time.time() - start:.3f}s')
            cv_results.append(param_dict)

            if best_score is None or param_dict['score'] < best_score:
                best_score = param_dict['score']
                best_model = temp_model
                best_params = param_dict

        # get best params
        df = pd.DataFrame(cv_results).sort_values('score', ascending=True)  # lower is better
        logger.info(f'\ngridsearch results:\n{df}')

        del best_params['score']
        logger.info(f'\nbest params: {best_params}')

        assert best_model is not None
        model_val = best_model

    elif args.model in ['constant', 'kgbm']:
        if args.tree_type == 'lgb':
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         eval_metric='mse', early_stopping_rounds=args.n_stopping_rounds)
            best_n_estimators = model_val.best_iteration_
        elif args.tree_type == 'xgb':
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         early_stopping_rounds=args.n_stopping_rounds)
            best_n_estimators = model_val.best_ntree_limit
        else:
            assert args.tree_type == 'cb'
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         early_stopping_rounds=args.n_stopping_rounds)
            best_n_estimators = model_val.tree_count_

        best_params = {'n_estimators': best_n_estimators}
        logger.info(f'\nbest params: {best_params}')

    elif args.model == 'pgbm':  # PGBM, only tune no. iterations
        model_val = clone(model).fit(X_tune, y_tune, eval_set=(X_val, y_val),
                                     early_stopping_rounds=args.n_stopping_rounds)
        best_n_estimators = model_val.learner_.best_iteration
        best_params = {'n_estimators': best_n_estimators}
        logger.info(f'\nbest params: {best_params}')

    else:  # NGBoost, only tune no. iterations
        assert args.model == 'ngboost'
        model_val = clone(model).fit(X_tune, y_tune, X_val=X_val, Y_val=y_val,
                                     early_stopping_rounds=args.n_stopping_rounds)
        if model_val.best_val_loss_itr is None:
            best_n_estimators = model_val.n_estimators
        else:
            best_n_estimators = model_val.best_val_loss_itr + 1
        best_params = {'n_estimators': best_n_estimators}
        logger.info(f'\nbest params: {best_params}')

    tune_time = time.time() - start
    logger.info('tune time: {:.3f}s'.format(tune_time))

    # KGBM ONLY: tune k
    if args.model == 'kgbm':
        tree_params = model_val.get_params()

        logger.info('\nTuning k (KGBM)...')
        start = time.time()

        model_val = KGBMWrapper(tree_frac=args.tree_frac, verbose=args.verbose,
                                logger=logger).fit(model_val, X_tune, y_tune, X_val=X_val, y_val=y_val)
        best_k = model_val.k_
        min_scale = model_val.min_scale_

        tune_time_kgbm = time.time() - start
        logger.info(f'\nbest k: {best_k}')
        logger.info(f'tune time (KGBM): {tune_time_kgbm:.3f}s')
    else:
        min_scale = None
        tune_time_kgbm = 0

    # KNN ONLY: tune min_scale
    if args.model == 'knn' and args.min_scale_pct > 0:
        logger.info('\nTuning KNN...')
        start = time.time()
        loc_val, scale_val = get_loc_scale(model_val, args.model, X=X_val, y=y_train,
                                           verbose=args.verbose, logger=logger)
        idx = int(len(scale_val) * args.min_scale_pct)
        min_scale = np.argsort(scale_val)[idx]
        tune_time_knn = time.time() - start

        logger.info(f'\nmin. scale: {min_scale}')
        logger.info(f'\ntune time (KNN): {tune_time_knn:.3f}s')
    else:
        tune_time_knn = 0

    # tune delta
    if args.delta:
        logger.info(f'\nTuning delta...')
        start = time.time()

        if args.model == 'kgbm':
            loc_val, scale_val = model_val.loc_val_, model_val.scale_val_

        elif args.model in ['constant', 'ngboost', 'pgbm'] or (args.model == 'knn' and args.min_scale_pct == 0):
            loc_val, scale_val = get_loc_scale(model_val, args.model, X=X_val, y=y_train,
                                               verbose=args.verbose, logger=logger)

        delta, delta_op = tune_delta(loc=loc_val, scale=scale_val, y=y_val,
                                     verbose=args.verbose, logger=logger)

        tune_time_delta = time.time() - start
        logger.info(f'\nbest delta: {delta}, op: {delta_op}')
        logger.info(f'tune time (delta): {tune_time_delta:.3f}s')
    else:
        tune_time_delta = 0

    # train: build using train+val data with best params
    logger.info('\nTraining...')
    start = time.time()

    model = clone(model).set_params(**best_params).fit(X_train, y_train)
    if args.model == 'kgbm':  # wrap model
        model = KGBMWrapper(k=best_k, tree_frac=args.tree_frac, min_scale=min_scale,
                            verbose=args.verbose, logger=logger).fit(model, X_train, y_train)

    train_time = time.time() - start
    logger.info(f'train time: {train_time:.3f}s')

    total_build_time = tune_time + tune_time_kgbm + tune_time_knn + tune_time_delta + train_time

    # predict: compute location and scale
    logger.info('\nPredicting...')
    start = time.time()
    loc, scale = get_loc_scale(model, args.model, X=X_test, y=y_train,
                               min_scale=min_scale, verbose=args.verbose, logger=logger)

    # add/multiply delta
    if args.delta:
        if delta_op == 'add':
            scale = scale + delta
        elif delta_op == 'mult':
            scale = scale * delta

    total_predict_time = time.time() - start
    logger.info(f'time: {total_predict_time:.3f}s')

    # display times
    logger.info(f'\ntotal build time: {total_build_time:.3f}s')
    logger.info(f'total predict time: {total_predict_time:.3f}s')

    # evaluate
    logger.info(f'\nEvaluating...')
    start = time.time()
    rmse, mae = util.eval_pred(y_test, model=model, X=X_test, logger=None, prefix=args.model)
    nll, crps = util.eval_normal(y=y_test, loc=loc, scale=scale, nll=True, crps=True)
    logger.info(f'CRPS: {crps:.5f}, NLL: {nll:.5f}, RMSE: {rmse:.5f}, MAE: {mae:.5f}')
    logger.info(f'time: {time.time() - start:.3f}s')

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
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'test_pred.png'), bbox_inches='tight')

    # save results
    result = vars(args)
    result['model_params'] = model.get_params()
    result['rmse'] = rmse
    result['mae'] = mae
    result['nll'] = nll
    result['crps'] = crps
    result['tune_time'] = tune_time
    result['train_time'] = train_time
    result['total_build_time'] = total_build_time
    result['total_predict_time'] = total_predict_time
    result['total_experiment_time'] = time.time() - begin
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    if args.delta:
        result['best_delta'] = delta
        result['best_delta_op'] = delta_op
        result['tune_time_delta'] = tune_time_delta
    if args.model == 'kgbm':
        result['tree_params'] = tree_params
        result['tune_time_kgbm'] = tune_time_kgbm
    if args.model == 'knn' and args.min_scale_pct > 0:
        result['best_min_scale'] = min_scale
        result['tune_time_knn'] = tune_time_knn
    if args.custom_dir == 'tree_frac':
        neighbor_idxs, neighbor_vals = model.pred_dist(X_test, return_kneighbors=True)
        result['neighbor_idxs'] = neighbor_idxs
        result['neighbor_vals'] = neighbor_vals
    elif args.custom_dir in ['dist', 'fl_dist', 'fls_dist']:
        logger.info('\nPredicting (distribution)...')
        start = time.time()
        neighbor_idxs, neighbor_vals = model.pred_dist(X_test, return_kneighbors=True)
        logger.info(f'time: {time.time() - start:.3f}s')

        # model output distribution using KDE
        logger.info('\nEvaluating (distributions)...')
        start = time.time()
        dist_res = {'nll': {}, 'crps': {}}
        for dist in args.distribution:
            if args.custom_dir == 'fl_dist':
                loc_test, scale_test = loc.copy(), None
            elif args.custom_dir == 'fls_dist':
                loc_test, scale_test = loc.copy(), scale.copy()
            else:
                loc_test, scale_test = None, None
            nll_d, crps_d = util.eval_dist(y=y_test, samples=neighbor_vals.copy(), dist=dist, nll=True,
                                           crps=True, min_scale=min_scale, random_state=args.random_state,
                                           loc=loc_test, scale=scale_test)
            logger.info(f'CRPS ({dist}): {crps_d:.5f}, NLL ({dist}): {nll_d:.5f}')
            dist_res['nll'][dist] = nll_d
            dist_res['crps'][dist] = crps_d
        logger.info(f'time: {time.time() - start:.3f}s')

        # plot output dist. of k nearest neighbors
        test_idxs = rng.choice(np.arange(X_test.shape[0]), size=16)

        fig, axs = plt.subplots(4, 4, figsize=(12, 8))
        axs = axs.flatten()

        for i, test_idx in enumerate(test_idxs):
            ax = axs[i]
            sns.histplot(neighbor_vals[test_idx], kde=True, ax=ax)
            ax.set_title(f'Test idx.: {test_idx}')
            if i >= 12:
                ax.set_xlabel('y')
            elif i in [0, 4, 8, 12]:
                ax.set_ylabel('Count')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'kdist.png'), bbox_inches='tight')

        result['neighbor_idxs'] = neighbor_idxs
        result['neighbor_vals'] = neighbor_vals
        result['dist_res'] = dist_res

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB): {result['max_rss_MB']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results to {os.path.join(out_dir, 'results.npy')}")

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # create method identifier
    method_name = util.get_method_identifier(args.model, vars(args))

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.custom_dir,
                           args.dataset,
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
    experiment(args, logger, out_dir)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--custom_dir', type=str, default='prediction')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model', type=str, default='kgbm')

    # Method settings
    parser.add_argument('--delta', type=int, default=1)  # affects ALL
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, KGBM, PGBM
    parser.add_argument('--tree_type', type=str, default='lgb')  # KGBM, constant
    parser.add_argument('--tree_frac', type=float, default=1.0)  # KGBM
    parser.add_argument('--affinity', type=str, default='unweighted')  # KGBM
    parser.add_argument('--min_scale_pct', type=float, default=0.0)  # KNN

    # Default settings
    parser.add_argument('--tune_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--val_frac', type=float, default=0.2)  # ALL
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--verbose', type=int, default=1)  # ALL
    parser.add_argument('--n_stopping_rounds', type=int, default=25)  # NGBoost, PGBM, KGBM, constant
    parser.add_argument('--weights', type=str, default='uniform')  # KNN
    parser.add_argument('--distribution', type=str, nargs='+',
                        default=['normal', 'skewnormal', 'lognormal', 'laplace',
                                 'student_t', 'logistic', 'gumbel', 'weibull', 'kde'])

    args = parser.parse_args()
    main(args)
