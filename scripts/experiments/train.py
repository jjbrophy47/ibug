"""
Model training.
"""
import os
import sys
import time
import ngboost
import resource
import argparse
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from bartpy.sklearnmodel import SklearnModel as BartModel

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for ibug
import util
from ibug import IBUGWrapper
from ibug import KNNWrapper

MIN_NUM_TREES = 2


def tune_model(model_type, X_tune, y_tune, X_val, y_val, tree_type=None,
               scoring='nll', bagging_frac=1.0, gridsearch=True, cond_mean_type='base',
               n_stopping_rounds=25, in_dir=None, logger=None, verbose=0, n_jobs=1):
    """
    Hyperparameter tuning.

    Input
        model_type: str, Model type.
        X_tune: 2d array of training data.
        y_tune: 1d array of training targets.
        X_val: 2d array of evaluation data.
        y_val: 1d array of evaluation targets.
        tree_type: str, GBRT type.
        scoring: str, Probabilistic evaluation metric.
        bagging_frac: float, Fraction of training instances to sample per tree.
        gridsearch: bool, If True, do gridsearch tuning.
        cond_mean_type: str, Conditional mean type.
        n_stopping_rounds: int, No. iterations to run without improved validation scoring.
            * Note: Only used when gridsearch is False.
        in_dir: If not None, then load in an already trained model.
        logger: object, Object for logging.
        verbose: int, verbosity level.
        n_jobs: int, number of jobs to run in parallel.

    Return tuned model and dict of best hyperparameters.
    """
    start = time.time()

    # base model
    model = get_model(model_type=model_type, tree_type=tree_type, scoring=scoring, bagging_frac=bagging_frac)

    # result objects
    model_val = None
    tune_dict = {'base_model': model}

    # load in a trained model
    if in_dir is not None:
        if logger:
            logger.info(f'\nloading saved validation model from {in_dir}/...')
        result = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
        model_val = util.load_model(model_type=tree_type, fp=result['saved_models']['model_val'])
        model_test = util.load_model(model_type=tree_type, fp=result['saved_models']['model_test'])
        tune_dict['model_val'] = model_val
        tune_dict['model_test'] = model_test
        tune_dict['tune_time_model'] = result['timing']['tune_model']

    # train a model from scratch
    else:

        # get candidate parameters
        param_grid = get_params(model_type=model_type, tree_type=tree_type, n_train=len(X_tune))
        
        # TEMPORARY
        if model_type == 'ibug' and tree_type == 'lgb' and args.custom_dir == 'ibug_bart':
            param_grid['n_estimators'] = [10, 50, 100, 200]

        # gridsearch
        if model_type in ['constant', 'ibug', 'pgbm', 'bart', 'cbu', 'knn'] and gridsearch:

            if logger:
                logger.info('\nmodel: {}, param_grid: {}'.format(model_type, param_grid))

            cv_results = []
            best_score = None
            best_model = None
            best_params = None

            param_dicts = list(util.product_dict(**param_grid))
            for i, param_dict in enumerate(param_dicts):
                temp_model = clone(model).set_params(**param_dict).fit(X_tune, y_tune)
                y_val_hat = temp_model.predict(X_val)
                param_dict['score'] = mean_squared_error(y_val, y_val_hat)
                cv_results.append(param_dict)

                if logger:
                    logger.info(f'[{i + 1:,}/{len(param_dicts):,}] {param_dict}'
                                f', cum. time: {time.time() - start:.3f}s'
                                f', score: {param_dict["score"]:.3f}')

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
        elif model_type in ['constant', 'ibug', 'knn']:
            assert tree_type in ['lgb', 'xgb', 'cb', 'ngboost', 'pgbm']

            if tree_type == 'lgb':
                model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                             eval_metric='mse', early_stopping_rounds=n_stopping_rounds)
                best_n_estimators = model_val.best_iteration_
            elif tree_type == 'xgb':
                model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                             early_stopping_rounds=n_stopping_rounds)
                best_n_estimators = model_val.best_ntree_limit
            elif tree_type == 'cb':
                model_val = clone(model).fit(X_tune, y_tune, eval_set=[(X_val, y_val)],
                                             early_stopping_rounds=n_stopping_rounds)
                best_n_estimators = model_val.tree_count_

            elif tree_type == 'ngboost':
                model_val = clone(model).fit(X_tune, y_tune, X_val=X_val, Y_val=y_val,
                                             early_stopping_rounds=n_stopping_rounds)
                if model_val.best_val_loss_itr is None:
                    best_n_estimators = model_val.n_estimators
                else:
                    best_n_estimators = model_val.best_val_loss_itr + 1

            elif tree_type == 'pgbm':
                model_val = clone(model).fit(X_tune, y_tune, eval_set=(X_val, y_val),
                                             early_stopping_rounds=n_stopping_rounds)
                best_n_estimators = model_val.learner_.best_iteration

            else:
                raise ValueError(f'Unknown tree type {tree_type}')

            best_n_estimators = max(best_n_estimators, MIN_NUM_TREES)
            best_params = {'n_estimators': best_n_estimators}
        
        # TODO: BART, only tune no. iterations
        elif model_type == 'bart':
            model_val = clone(model).fit(X_tune, y_tune)
            best_params = {}
        
        # TODO: CBU, only tune no. iterations
        elif model_type == 'cbu':
            model_val = clone(model).fit(X_tune, y_tune)
            best_params = {}

        # PGBM, only tune no. iterations
        elif model_type == 'pgbm':
            model_val = clone(model).fit(X_tune, y_tune, eval_set=(X_val, y_val),
                                         early_stopping_rounds=n_stopping_rounds)
            best_n_estimators = model_val.learner_.best_iteration
            best_n_estimators = max(best_n_estimators, MIN_NUM_TREES)
            best_params = {'n_estimators': best_n_estimators}

        # NGBoost, only tune no. iterations
        else:
            assert model_type == 'ngboost'
            model_val = clone(model).fit(X_tune, y_tune, X_val=X_val, Y_val=y_val,
                                         early_stopping_rounds=n_stopping_rounds)
            if model_val.best_val_loss_itr is None:
                best_n_estimators = model_val.n_estimators
            else:
                best_n_estimators = model_val.best_val_loss_itr + 1
            best_n_estimators = max(best_n_estimators, MIN_NUM_TREES)
            best_params = {'n_estimators': best_n_estimators}

        tune_dict['model_val'] = model_val
        tune_dict['best_params'] = best_params
        tune_dict['tune_time_model'] = time.time() - start
        if logger:
            logger.info(f"\nbest params: {tune_dict['best_params']}")
            logger.info(f"tune time (model): {tune_dict['tune_time_model']:.3f}s")

    # IBUG, KNN, and KNN_FI ONLY: tune k (and min. scale)
    tune_dict['tune_time_extra'] = 0
    if model_type in ['ibug', 'knn']:
        best_params_wrapper = {}
        WrapperClass = IBUGWrapper if model_type == 'ibug' else KNNWrapper
        model_val_wrapper = WrapperClass(scoring=scoring, variance_calibration=False,
            cond_mean_type=cond_mean_type, verbose=verbose, n_jobs=n_jobs, logger=logger)

        if logger:
            logger.info('\nTuning k and min. scale...')
        start = time.time()
        model_val_wrapper = model_val_wrapper.fit(model_val, X_tune, y_tune, X_val=X_val, y_val=y_val)
        best_params_wrapper = {'k': model_val_wrapper.k_,
                               'min_scale': model_val_wrapper.min_scale_,
                               'cond_mean_type': model_val_wrapper.cond_mean_type}
        if model_type == 'knn':
            best_params_wrapper['max_feat'] = model_val_wrapper.max_feat_
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
    elif model_type in ['constant', 'ngboost', 'pgbm', 'bart', 'cbu']:
        loc_val, scale_val = get_loc_scale(model_val, model_type, X=X_val, y_train=y_tune)
    tune_dict['loc_val'] = loc_val
    tune_dict['scale_val'] = scale_val

    return tune_dict


def tune_delta(loc, scale, y, ops=['add', 'mult'],
               delta_vals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                           1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
               multipliers=[1.0, 2.5, 5.0],
               scoring='nll', verbose=0, logger=None):
    """
    Add or multiply detla to scale values.

    Input
        loc: 1d array of location values.
        scale: 1d array of scale values.
        y: 1d array of target values (same shape as scale).
        ops: list, List of operations to perform to scale array.
        delta_vals: list, List of candidate delta values.
        multipliers: list, List of values to multiply the base values by.
        scoring: str, Evaluation metric.
        verbose: int, Verbosity level.
        logger: object, Object for logging.

    Return
        - float, best delta value.
        - str, best operation.
    """
    assert ops == ['add', 'mult']
    assert loc.shape == scale.shape == y.shape

    results = []
    for op in ops:
        for delta in delta_vals:
            for multiplier in multipliers:

                if op == 'mult' and delta == 0.0:
                    continue

                if op == 'add':
                    temp_scale = scale + (delta * multiplier)
                else:
                    temp_scale = scale * (delta * multiplier)

                score = util.eval_uncertainty(y=y, loc=loc, scale=temp_scale, metric=scoring)
                results.append({'delta': delta, 'op': op, 'multiplier': multiplier, 'score': score})

    df = pd.DataFrame(results).sort_values('score', ascending=True)

    best_delta = df.iloc[0]['delta'] * df.iloc[0]['multiplier']
    best_op = df.iloc[0]['op']

    if verbose > 0:
        if logger:
            logger.info(f'\ndelta gridsearch:\n{df}')
        else:
            print(f'\ndelta gridsearch:\n{df}')

    return best_delta, best_op


def get_loc_scale(model, model_type, X, y_train=None):
    """
    Predict location and scale for each x in X.

    Input
        model: object, Uncertainty estimator.
        model_type: str, Type of uncertainty estimator.
        X: 2d array of input data.
        y_train: 1d array of targets (constant method only).

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

    elif model_type == 'cbu':  # CatBoost RMSEWithUncertainty
        loc, scale = model.pred_dist(X)
    
    elif model_type == 'bart':
        samples = model.data.y.unnormalize_y(np.array([x.predict(X) for x in model._model_samples]))
        loc, scale = np.mean(samples, axis=0), np.std(samples, axis=0)

    elif model_type == 'pgbm':
        _, loc, variance = model.learner_.predict_dist(X.astype(np.float32),
                                                       output_sample_statistics=True)
        scale = np.sqrt(variance)
        loc = loc.numpy()
        scale = scale.numpy()

    elif model_type == 'ngboost':
        y_dist = model.pred_dist(X)
        loc, scale = y_dist.params['loc'], y_dist.params['scale']

    return loc, scale


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
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000]}

    elif model_type == 'pgbm':
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                  'max_leaves': [15, 31, 61, 91],
                  'learning_rate': [0.01, 0.1],
                  'min_data_in_leaf': [1, 20],
                  'max_bin': [255]}

    elif model_type == 'knn' and tree_type == 'knn':
        k_list = [3, 5, 7, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701]
        params = {'n_neighbors': [k for k in k_list if k <= n_train]}
    
    elif model_type == 'bart':
        params = {'n_trees': [10, 50, 100, 200], 'n_chains': [5]}

    elif model_type == 'cbu':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'min_data_in_leaf': [1, 20],
                      'max_bin': [255]}

    elif model_type in ['constant', 'ibug', 'knn']:
        assert tree_type is not None

        if tree_type == 'lgb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'num_leaves': [15, 31, 61, 91],
                      'learning_rate': [0.01, 0.1],
                      'min_child_samples': [1, 20],
                      'max_bin': [255]}

        elif tree_type == 'xgb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'min_child_weight': [1, 20],
                      'max_bin': [255]}

        elif tree_type == 'cb':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'max_depth': [2, 3, 5, 7, None],
                      'learning_rate': [0.01, 0.1],
                      'min_data_in_leaf': [1, 20],
                      'max_bin': [255]}

        elif tree_type == 'ngboost':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000]}

        elif tree_type == 'pgbm':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'max_leaves': [15, 31, 61, 91],
                      'learning_rate': [0.01, 0.1],
                      'min_data_in_leaf': [1, 20],
                      'max_bin': [255]}

        elif tree_type == 'skrf':
            params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                      'max_depth': [2, 3, 5, 7, None],
                      'min_samples_leaf': [1, 20]}

        else:
            raise ValueError('tree_type unknown: {}'.format(tree_type))
    else:
        raise ValueError('model_type unknown: {}'.format(model_type))

    return params


def get_model(model_type, tree_type, scoring='nll', n_estimators=2000, max_bin=64,
              lr=0.1, max_leaves=8, max_depth=6, min_leaf_samples=1,
              bagging_frac=1.0, max_features='sqrt', random_state=1, verbose=2):
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
        min_leaf_samples: int, Miniumum number of samples per leaf.
        bagging_frac: float, Fraction of training instances to sample per tree.
        max_features: str, Maximum features to sample per decision node.
        random_state: int, Random seed.
        verbose: int, Verbosity level.

    Return initialized models with default values.
    """
    assert bagging_frac > 0 and bagging_frac <= 1.0

    if model_type == 'ngboost':  # defaults: min_data_leaf=1, max_depth=3
        assert scoring in ['nll', 'crps']
        score = ngboost.scores.CRPScore if scoring == 'crps' else ngboost.scores.LogScore
        model = ngboost.NGBRegressor(n_estimators=n_estimators, Score=score,
                                     minibatch_frac=bagging_frac, verbose=verbose)

    elif model_type == 'pgbm':
        import pgbm  # dynamic import (some machines cannot install pgbm)
        model = pgbm.PGBMRegressor(n_estimators=n_estimators, learning_rate=lr,
                                   max_leaves=max_leaves, max_bin=max_bin,
                                   bagging_fraction=bagging_frac,
                                   min_data_in_leaf=min_leaf_samples, verbose=verbose)

    elif model_type == 'knn' and tree_type == 'knn':
        model = KNeighborsRegressor(weights='uniform')
    
    elif model_type == 'bart':
        model = BartModel()
    
    elif model_type == 'cbu':
        model = util.CatBoostRMSEUncertaintyWrapper(n_estimators=n_estimators, max_depth=max_depth,
                                                    learning_rate=lr, max_bin=max_bin,
                                                    min_data_in_leaf=min_leaf_samples, subsample=bagging_frac,
                                                    loss_function='RMSEWithUncertainty',
                                                    custom_metric='RMSE',
                                                    posterior_sampling=True,
                                                    random_state=random_state, logging_level='Silent')

    elif model_type in ['constant', 'ibug', 'knn']:
        assert tree_type is not None

        if tree_type == 'lgb':
            bagging_freq = 1 if bagging_frac < 1.0 else 0
            model = LGBMRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=-1,
                                  num_leaves=max_leaves, max_bin=max_bin,
                                  subsample=bagging_frac, subsample_freq=bagging_freq,
                                  min_child_samples=min_leaf_samples, random_state=random_state)

        elif tree_type == 'xgb':
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                                 max_bin=max_bin, min_child_weight=min_leaf_samples,
                                 subsample=bagging_frac, random_state=random_state)

        elif tree_type == 'cb':
            model = CatBoostRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, max_bin=max_bin,
                                      min_data_in_leaf=min_leaf_samples, subsample=bagging_frac,
                                      random_state=random_state, logging_level='Silent')

        elif tree_type == 'ngboost':
            assert scoring in ['nll', 'crps']
            score = ngboost.scores.CRPScore if scoring == 'crps' else ngboost.scores.LogScore
            model = ngboost.NGBRegressor(n_estimators=n_estimators, Score=score,
                                         minibatch_frac=bagging_frac, verbose=verbose)

        elif tree_type == 'pgbm':
            import pgbm  # dynamic import (some machines cannot install pgbm)
            model = pgbm.PGBMRegressor(n_estimators=n_estimators, learning_rate=lr,
                                       max_leaves=max_leaves, max_bin=max_bin,
                                       bagging_fraction=bagging_frac,
                                       min_data_in_leaf=min_leaf_samples, verbose=verbose)

        elif tree_type == 'skrf':
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_leaf_samples, max_features=max_features,
                                          random_state=random_state)

        else:
            raise ValueError('tree_type unknown: {}'.format(tree_type))
    else:
        raise ValueError('model unknown: {}'.format(model))

    return model


def experiment(args, logger, out_dir, in_dir=None):
    """
    Main method comparing performance of tree ensembles and svm models.
    """
    begin = time.time()  # experiment timer
    rng = np.random.default_rng(args.random_state)  # pseudo-random number generator

    # get data
    X_train, X_test, y_train, _, _ = util.get_data(args.data_dir, args.dataset, args.fold)

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
                           tree_type=args.tree_type, scoring=args.scoring,
                           bagging_frac=args.bagging_frac,
                           cond_mean_type=args.cond_mean_type,
                           n_stopping_rounds=args.n_stopping_rounds,
                           gridsearch=args.gridsearch, in_dir=in_dir,
                           logger=logger, verbose=args.verbose, n_jobs=args.n_jobs)
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

    assert 'base_model' in tune_dict
    base_model = tune_dict['base_model']

    if args.model_type in ['ibug', 'knn']:  # wrap model
        assert 'model_val_wrapper' in tune_dict
        assert 'best_params_wrapper' in tune_dict
        assert 'WrapperClass' in tune_dict

        model_val = tune_dict['model_val_wrapper']
        best_params_wrapper = tune_dict['best_params_wrapper']
        WrapperClass = tune_dict['WrapperClass']

        if 'model_test' in tune_dict:  # loaded model
            logger.info(f'\nloading saved test model from {in_dir}/...')
            base_model_test = tune_dict['model_test']
        else:
            assert 'best_params' in tune_dict
            best_params = tune_dict['best_params']
            base_model_test = clone(base_model).set_params(**best_params).fit(X_train, y_train)

        model_test = WrapperClass(verbose=args.verbose, variance_calibration=False,
                                  n_jobs=args.n_jobs, logger=logger)\
            .set_params(**best_params_wrapper).fit(base_model_test, X_train, y_train)
    else:
        assert 'best_params' in tune_dict
        assert 'model_val' in tune_dict

        best_params = tune_dict['best_params']
        model_val = tune_dict['model_val']
        model_test = clone(base_model).set_params(**best_params).fit(X_train, y_train)

    train_time = time.time() - start
    logger.info(f'train time: {train_time:.3f}s')

    # display times
    tune_train_time = tune_time_model + tune_time_extra + tune_time_delta + train_time
    logger.info(f'\ntune+train time: {tune_train_time:.3f}s')

    # save results
    result = {}
    result['train_args'] = vars(args)
    result['data'] = {'n_train': len(X_train),
                      'n_tune': len(X_tune),
                      'n_val': len(X_val),
                      'n_test': len(X_test),
                      'n_feature': X_train.shape[1],
                      'tune_idxs': tune_idxs,
                      'val_idxs': val_idxs}
    result['model_params'] = model_test.get_params()
    result['timing'] = {'tune_model': tune_time_model, 'tune_extra': tune_time_extra,
                        'tune_delta': tune_time_delta, 'train': train_time, 'tune_train': tune_train_time}
    result['delta'] = {'best_delta': delta, 'best_op': delta_op}
    result['saved_models'] = {'model_val': os.path.join(out_dir, 'model_val'),
                              'model_test': os.path.join(out_dir, 'model_test')}
    result['misc'] = {'max_RSS': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,  # MB if OSX, GB if Linux
                      'total_experiment_time': time.time() - begin}

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['misc']['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB if MacOSX, GB if Linux): {result['misc']['max_RSS']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results and models to {os.path.join(out_dir, 'results.npy')}")

    # save results/models
    util.save_model(model=model_val, model_type=args.model_type, fp=result['saved_models']['model_val'])
    util.save_model(model=model_test, model_type=args.model_type, fp=result['saved_models']['model_test'])
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # create method identifier
    method_name = util.get_method_identifier(args.model_type, vars(args))

    # load trained ngboost or pgbm model
    if args.load_model and args.model_type == 'ibug' and args.tree_type in ['ngboost', 'pgbm']:
        in_method_name = util.get_method_identifier(args.tree_type, vars(args))
        in_dir = os.path.join(args.out_dir,
                              args.custom_dir,
                              args.dataset,
                              args.in_scoring,
                              f'fold{args.fold}',
                              in_method_name)
    else:
        in_dir = None

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
    experiment(args, logger, out_dir, in_dir=in_dir)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/experiments/train/')
    parser.add_argument('--custom_dir', type=str, default='default')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='ibug')

    # Method settings
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, IBUG, PGBM, BART, CBU
    parser.add_argument('--tree_type', type=str, default='lgb')  # IBUG, constant, kNN
    parser.add_argument('--tree_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--tree_subsample_order', type=str, default='random')  # IBUG
    parser.add_argument('--instance_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--affinity', type=str, default='unweighted')  # IBUG
    parser.add_argument('--cond_mean_type', type=str, default='base')  # kNN

    # Default settings
    parser.add_argument('--tune_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--bagging_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--val_frac', type=float, default=0.2)  # ALL
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--verbose', type=int, default=2)  # ALL
    parser.add_argument('--n_stopping_rounds', type=int, default=25)  # NGBoost, PGBM, IBUG, constant
    parser.add_argument('--scoring', type=str, default='crps')

    # Extra settings
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--in_scoring', type=str, default='nll')
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()
    main(args)
