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
from ngboost.scores import CRPScore
from ngboost.scores import LogScore
from scipy.stats import gaussian_kde
from scipy.stats import lognorm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import util
from ibug import IBUGWrapper
from ibug import KNNWrapper


def tune_model(model_type, X_tune, y_tune, X_val, y_val, tree_type=None,
               gridsearch=True, n_stopping_rounds=25, logger=None):
    """
    Hyperparameter tuning.

    Input
        model_type: str, Model type.
        X_tune: 2d array of training data.
        y_tune: 1d array of training targets.
        X_val: 2d array of evaluation data.
        y_val: 1d array of evaluation targets.
        tree_type: str, GBRT type.
        gridsearch: bool, If True, do gridsearch tuning.
        n_stopping_rounds: int, No. iterations to run without improved validation scoring.
            Only used when gridsearch is False.
        logger: object, Object for logging.

    Return tuned model and dict of best hyperparameters.
    """
    start = time.time()

    # result objects
    model_val = None
    tune_dict = {}

    # get model and candidate parameters
    model = get_model(model_type=model_type, tree_type=tree_type)
    param_grid = get_params(model_type=model_type, tree_type=tree_type, n_train=len(X_tune))

    # gridsearch
    if model_type == 'knn' or (model_type in ['constant', 'ibug', 'pgbm'] and gridsearch):

        if logger:
            logger.info('\nmodel: {}, param_grid: {}'.format(args.model_type, param_grid))

        cv_results = []
        best_score = None
        best_model = None
        best_params = None

        param_dicts = list(product_dict(**param_grid))
        for i, param_dict in enumerate(param_dicts):
            temp_model = clone(model).set_params(**param_dict).fit(X_tune, y_tune)
            y_val_hat = temp_model.predict(X_val)
            param_dict['score'] = mean_squared_error(y_val, y_val_hat)
            cv_results.append(param_dict)

            if logger:
                logger.info(f'[{i + 1:,}/{len(param_dicts):,}] {param_dict}'
                            f', cum. time: {time.time() - start:.3f}s')

            if best_score is None or param_dict['score'] < best_score:
                best_score = param_dict['score']
                best_model = temp_model
                best_params = param_dict

        # get best params
        df = pd.DataFrame(cv_results).sort_values('score', ascending=True)  # lower is better
        del best_params['score']
        if logger:
            logger.info(f'\ngridsearch results:\n{df}')

        assert best_model is not None
        model_val = best_model

    # base model, only tune no. iterations
    elif model_type in ['constant', 'ibug']:
        assert tree_type in ['lgb', 'xgb', 'cb']

        if tree_type == 'lgb':
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         eval_metric='mse', early_stopping_rounds=n_stopping_rounds)
            best_n_estimators = model_val.best_iteration_
        elif tree_type == 'xgb':
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         early_stopping_rounds=n_stopping_rounds)
            best_n_estimators = model_val.best_ntree_limit
        else:
            assert tree_type == 'cb'
            model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                         early_stopping_rounds=n_stopping_rounds)
            best_n_estimators = model_val.tree_count_

        best_params = {'n_estimators': best_n_estimators}

    # PGBM, only tune no. iterations
    elif model_type == 'pgbm':
        model_val = clone(model).fit(X_tune, y_tune, eval_set=(X_val, y_val),
                                     early_stopping_rounds=args.n_stopping_rounds)
        best_n_estimators = model_val.learner_.best_iteration
        best_params = {'n_estimators': best_n_estimators}

    # NGBoost, only tune no. iterations
    else:
        assert model_type == 'ngboost'
        model_val = clone(model).fit(X_tune, y_tune, X_val=X_val, Y_val=y_val,
                                     early_stopping_rounds=args.n_stopping_rounds)
        if model_val.best_val_loss_itr is None:
            best_n_estimators = model_val.n_estimators
        else:
            best_n_estimators = model_val.best_val_loss_itr + 1
        best_params = {'n_estimators': best_n_estimators}

    tune_dict['model_val'] = model_val
    tune_dict['best_params'] = best_params
    tune_dict['tune_time_model'] = time.time() - start
    if logger:
        logger.info(f"\nbest params: {tune_dict['best_params']}")
        logger.info(f"tune time (model): {tune_dict['tune_time_model']:.3f}s")

    # IBUG and KNN ONLY: tune k (and min. scale)
    tune_time_extra = 0
    if model_type in ['ibug', 'knn']:
        best_params_wrapper = {}
        WrapperClass = IBUGWrapper if model_type == 'ibug' else KNNWrapper

        if logger:
            logger.info('\nTuning k and min. scale...')

        start = time.time()
        model_val_wrapper = WrapperClass(scoring=args.scoring, verbose=args.verbose,
                                         logger=logger).fit(model_val, X_tune, y_tune,
                                                            X_val=X_val, y_val=y_val)
        best_params_wrapper = {'k': model_val_wrapper.k_,
                               'min_scale': model_val_wrapper.min_scale_}
        tune_dict['model_val_wrapper'] = model_val_wrapper
        tune_dict['best_params_wrapper'] = best_params_wrapper
        tune_dict['WrapperClass'] = WrapperClass
        tune_dict['tune_time_extra'] = time.time() - start

        if logger:
            logger.info(f"best params (wrapper): {best_params_wrapper}")
            logger.info(f"tune time (extra): {tune_dict['tune_time_extra']:.3f}s")

    # get validation predictions
    if model_type in ['ibug', 'knn']:
        loc_val, scale_val = model_val_wrapper.loc_val_, model_val_wrapper.scale_val_
    elif model_type in ['constant', 'ngboost', 'pgbm']:
        loc_val, scale_val = get_loc_scale(model_val, model_type, X=X_val, y_train=y_train)
    tune_dict['loc_val'] = loc_val
    tune_dict['scale_val'] = scale_val

    return tune_dict


def tune_delta(loc, scale, y, ops=['add', 'mult'],
               delta_vals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                           1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
               scoring='nll', verbose=0, logger=None):
    """
    Add or multiply detla to scale values.

    Input
        loc: 1d array of location values.
        scale: 1d array of scale values.
        y: 1d array of target values (same shape as scale).
        op: list, List of operations to perform to scale array.
        delta_vals: list, List of candidate delta values.
        scoring: str, Evaluation metric.
        verbose: int, Verbosity level.
        logger: object, Object for logging.

    Return
        1d array of updated scale values.
    """
    assert scoring in ['nll', 'crps']
    assert ops == ['add', 'mult']
    assert loc.shape == scale.shape == y.shape

    results = []
    for op in ops:
        for delta in delta_vals:

            if op == 'mult' and delta == 0.0:
                continue

            if op == 'add':
                temp_scale = scale + delta
            else:
                temp_scale = scale * delta

            if scoring == 'nll':
                score = util.eval_normal(y=y, loc=loc, scale=temp_scale, nll=True, crps=False)
            else:
                score = util.eval_normal(y=y, loc=loc, scale=temp_scale, nll=False, crps=True)
            results.append({'delta': delta, 'op': op, 'score': score})

    df = pd.DataFrame(results).sort_values('score', ascending=True)

    best_delta = df.iloc[0]['delta']
    best_op = df.iloc[0]['op']

    if verbose > 0:
        if logger:
            logger.info(f'\ndelta gridsearch:\n{df}')
        else:
            print(f'\ndelta gridsearch:\n{df}')

    return best_delta, best_op


def get_loc_scale(model, model_type, X, y_train=None, delta=None, delta_op=None):
    """
    Predict location and scale for each x in X.

    Input
        model: object, Uncertainty estimator.
        model_type: str, Type of uncertainty estimator.
        X: 2d array of input data.
        y_train: 1d array of targets (constant method only).
        delta: float, Value to add or multiply to each scale value.
        delta_op: str, Operation to perform on each scalue value using delta.

    Return
        Tuple, 2 1d arrays of locations and scales.
    """
    if model_type == 'constant':
        assert y_train is not None
        loc = model.predict(X)
        scale = np.full(len(X), np.std(y_train), dtype=np.float32)

    elif model_type == 'ibug':
        loc, scale = model.pred_dist(X)

    elif model_type == 'knn':
        loc, scale = model.pred_dist(X)

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


def get_params(model_type, n_train, tree_type=None):
    """
    Return dict of parameters values to try for gridsearch.

    Input
        model_type: str, Probabilistic estimator.
        n_train: int, Number of train instances.
        tree_type: str, GBRT type.

    Return dict of gridsearch parameter values.
    """
    if model_type == 'ngboost':
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000]}

    elif model_type == 'pgbm':
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                  'max_leaves': [15, 31, 61, 91],
                  'learning_rate': [0.01, 0.1],
                  'max_bin': [255]}

    elif model_type == 'knn':
        k_list = [3, 5, 7, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701]
        params = {'n_neighbors': [k for k in k_list if k <= n_train]}

    elif model_type in ['constant', 'ibug']:
        assert tree_type is not None

        if tree_type == 'lgb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'num_leaves': [15, 31, 61, 91],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}

        elif tree_type == 'xgb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}

        elif tree_type == 'cb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'max_bin': [255]}
        else:
            raise ValueError('tree_type unknown: {}'.format(tree_type))
    else:
        raise ValueError('model_type unknown: {}'.format(model_type))

    return params


def get_model(model_type, tree_type, scoring='nll', n_estimators=2000, max_bin=64,
              lr=0.1, max_leaves=16, max_depth=4, verbose=2, random_state=1):
    """
    Return the appropriate classifier.

    Input
        model_type: str, Probabilistic estimator.
        tree_type: str, GBRT type
        scoring: str, Evaluation metric.
        n_estimators: int, Number of trees.
        max_bin: int, Maximum number of feature bins.
        lr: float, Learning rate.
        max_leaves: int, Maximum number of leaves.
        max_depth: int, Maximum depth of each tree.
        verbose: int, Verbosity level.
        random_state: int, Random seed.

    Return initialized models with default values.
    """
    if model_type == 'ngboost':
        assert scoring in ['nll', 'crps']
        score = ngboost.scores.CRPScore if scoring == 'crps' else ngboost.scores.LogScore
        model = ngboost.NGBRegressor(n_estimators=n_estimators, verbose=verbose, Score=score)

    elif model_type == 'pgbm':
        import pgbm  # dynamic import (some machines cannot install pgbm)
        model = pgbm.PGBMRegressor(n_estimators=n_estimators, learning_rate=lr, max_leaves=max_leaves,
                                   max_bin=max_bin, verbose=verbose)

    elif model_type == 'knn':
        model = KNeighborsRegressor(weights='uniform')

    elif model_type in ['constant', 'ibug']:
        assert tree_type is not None

        if tree_type == 'lgb':
            model = LGBMRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=-1,
                                  num_leaves=max_leaves, max_bin=max_bin, random_state=random_state)

        elif tree_type == 'xgb':
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                                 max_bin=max_bin, random_state=random_state)

        elif tree_type == 'cb':
            model = CatBoostRegressor(n_estimators=n_estimators, max_depth=max_depth, max_bin=max_bin,
                                      random_state=random_state,
                                      logging_level='Silent', learning_rate=lr)
        else:
            raise ValueError('tree_type unknown: {}'.format(tree_type))
    else:
        raise ValueError('model unknown: {}'.format(model))

    return model


def experiment(args, logger, out_dir):
    """
    Main method comparing performance of tree ensembles and svm models.
    """
    begin = time.time()  # experiment timer
    rng = np.random.default_rng(args.random_state)  # pseduo-random number generator

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset, args.fold)

    # use a fraction of the training data for tuning
    if args.tune_frac < 1.0:
        assert args.tune_frac > 0.0
        n_tune = int(len(X_train) * args.tune_frac)
        tune_idxs = rng.choice(np.arange(len(X_train)), size=n_tune, replace=False)
    else:
        tune_idxs = np.arange(len(X_train))

    # split total tuning set into a train/validation set
    tune_idxs, val_idxs = train_test_split(tune_idxs, test_size=args.val_frac,
                                           random_state=args.random_state)
    X_tune, y_tune = X_train[tune_idxs].copy(), y_train[tune_idxs].copy()
    X_val, y_val = X_train[val_idxs].copy(), y_train[val_idxs].copy()

    logger.info('no. train: {:,}'.format(X_train.shape[0]))
    logger.info('  -> no. tune: {:,}'.format(X_tune.shape[0]))
    logger.info('  -> no. val.: {:,}'.format(X_val.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # tune
    logger.info('\nTuning model...')
    start = time.time()
    tune_dict = tune_model(model_type=args.model_type,
                           X_tune=X_tune, y_tune=y_tune, X_val=X_val, y_val=y_val,
                           tree_type=args.tree_type, n_stopping_rounds=args.n_stopping_rounds,
                           gridsearch=args.gridsearch, logger=logger)
    tune_time_model = tune_dict['tune_time_model']
    tune_time_extra = tune_dict['tune_time_extra']
    logger.info(f'\ntune time (model+extra): {time.time() - start:.3f}s')

    # tune delta
    assert 'loc_val' in tune_dict and 'scale_val' in tune_dict
    logger.info(f'\nTuning delta...')
    start = time.time()
    delta, delta_op = tune_delta(loc=tune_dict['loc_val'], scale=tune_dict['scale_val'], y=y_val,
                                 scoring=args.scoring, verbose=args.verbose, logger=logger)
    tune_time_delta = time.time() - start
    logger.info(f'\nbest delta: {delta}, op: {delta_op}')
    logger.info(f'tune time (delta): {tune_time_delta:.3f}s')

    # train: build using train+val data with best params
    logger.info('\n[TEST] Training...')
    start = time.time()

    assert 'best_params' in tune_dict
    best_params = tune_dict['best_params']

    if args.model_type in ['ibug', 'knn']:  # wrap model
        assert 'model_val' in tune_dict
        assert 'model_val_wrapper' in tune_dict
        assert 'best_params_wrapper' in tune_dict
        assert 'WrapperClass' in tune_dict
        base_model_val = tune_dict['model_val']
        model_val = tune_dict['model_val_wrapper']
        best_params_wrapper = tune_dict['best_params_wrapper']
        WrapperClass = tune_dict['WrapperClass']

        base_model_test = clone(base_model_val).set_params(**best_params).fit(X_train, y_train)
        model_test = WrapperClass(verbose=args.verbose, logger=logger).set_params(**best_params_wrapper)\
            .fit(base_model_test, X_train, y_train)
    else:
        model_val = tune_dict['model_val']
        model_test = clone(model_val).set_params(**best_params).fit(X_train, y_train)

    train_time = time.time() - start
    logger.info(f'train time: {train_time:.3f}s')

    # display times
    tune_train_time = tune_time_model + tune_time_extra + tune_time_delta + train_time
    logger.info(f'\ntune+train time: {tune_train_time:.3f}s')

    # save results
    result = vars(args)
    result['n_train'] = len(X_train)
    result['n_tune'] = len(X_tune)
    result['n_val'] = len(X_val)
    result['n_test'] = len(X_test)
    result['n_feature'] = X_train.shape[1]
    result['model_params'] = model_test.get_params()
    result['tune_time_model'] = tune_time_model
    result['tune_time_extra'] = tune_time_extra
    result['tune_time_delta'] = tune_time_delta
    result['train_time'] = train_time
    result['tune_train_time'] = tune_train_time
    result['best_delta'] = delta
    result['best_delta_op'] = delta_op
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_experiment_time'] = time.time() - begin

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB): {result['max_rss_MB']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results and models to {os.path.join(out_dir, 'results.npy')}")

    # save results/models
    np.save(os.path.join(out_dir, 'results.npy'), result)
    util.save_model(model=model_val, model_type=args.model_type, out_dir=out_dir, fn='model_val')
    util.save_model(model=model_test, model_type=args.model_type, out_dir=out_dir, fn='model_test')


def main(args):

    # create method identifier
    method_name = util.get_method_identifier(args.model_type, vars(args))

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.custom_dir,
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
    experiment(args, logger, out_dir)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/experiments/')
    parser.add_argument('--custom_dir', type=str, default='train')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='ibug')

    # Method settings
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, IBUG, PGBM
    parser.add_argument('--tree_type', type=str, default='lgb')  # IBUG, constant
    parser.add_argument('--tree_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--tree_sample_order', type=str, default='random')  # IBUG
    parser.add_argument('--affinity', type=str, default='unweighted')  # IBUG

    # Default settings
    parser.add_argument('--tune_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--val_frac', type=float, default=0.2)  # ALL
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--verbose', type=int, default=2)  # ALL
    parser.add_argument('--n_stopping_rounds', type=int, default=25)  # NGBoost, PGBM, IBUG, constant
    parser.add_argument('--scoring', type=str, default='nll')

    args = parser.parse_args()
    main(args)