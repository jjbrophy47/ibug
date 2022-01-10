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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from ngboost import NGBRegressor
from pgbm import PGBMRegressor

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import util
from intent import KGBM

# constants
EPSILON = 1e-15


def get_method_identifier(args):
    """
    Return method name concatenated with a hash
    unqiuely identifying this method with the chosen settings.
    """
    settings = {}

    if args.model == 'kgbm':
        settings['affinity'] = args.affinity
        settings['tree_type'] = args.tree_type

    if args.model == 'knn':
        settings['tree_type'] = args.tree_type

    if args.scale_bias in ['add', 'mult']:
        settings['scale_bias'] = args.scale_bias

    if len(settings) > 0:
        hash_str = util.dict_to_hash(settings)
        method_name = f'{args.model}_{hash_str}'
    else:
        method_name = args.model

    return method_name


def include_scale_bias(scale, y, op='add'):
    """
    Add or multiply scale bias value.

    Input
        scale: 1d array of scale values.
        y: 1d array of target values (same shape as scale).
        op: str, operation to perform to scale array.

    Return
        1d array of updated scale values.
    """
    if op == 'add':
        scale = scale + np.std(y)

    elif op == 'mult':
        scale = scale * np.std(y)

    return scale


def product_dict(**kwargs):
    """
    Return cross-product iterable of dicts given
    a set of named lists.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_model(args):
    """
    Return the appropriate classifier.
    """
    if args.model == 'ngboost':
        model = NGBRegressor(verbose=0)
        params = {'n_estimators': [10, 25, 50, 100, 200, 500, 1000, 2000]}

    elif args.model == 'pgbm':
        model = PGBMRegressor(verbose=0)
        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000],
                  'max_leaves': [15, 31, 61, 91],
                  'learning_rate': [0.01, 0.1]}

    elif args.model == 'knn':
        model = KNeighborsRegressor(weights=args.weights)
        params = {'n_neighbors': [3, 5, 7, 11, 15, 31, 61, 91, 121, 151, 201]}

    elif args.model in ['constant', 'kgbm']:
        model = util.get_model(tree_type=args.tree_type,
                               random_state=args.random_state)

        params = {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2000]}

        if args.tree_type == 'lgb':
            params['max_depth'] = [-1]
            params['num_leaves'] = [15, 31, 61, 91]
            params['learning_rate'] = [0.01, 0.1]

        elif args.tree_type == 'sgb':
            params['max_iter'] = params['n_estimators']
            params['max_depth'] = [None]
            params['max_leaf_nodes'] = [15, 31, 61, 91]
            params['max_bins'] = [50, 100, 250]
            del params['n_estimators']

        elif args.tree_type == 'cb':
            params['learning_rate'] = [0.1, 0.3, 0.6, 0.9]
            params['max_depth'] = [2, 3, 4, 5, 6, 7]

        elif args.tree_type == 'xgb':
            params['max_depth'] = [2, 3, 4, 5, 6, 7]

        elif args.tree_type == 'skgbm':
            params['max_depth'] = [2, 3, 4, 5, 6, 7]

    else:
        raise ValueError('model unknown: {}'.format(args.model))

    return model, params


def experiment(args, logger, out_dir):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # start experiment timer
    begin = time.time()

    # pseduo-random number generator
    rng = np.random.default_rng(args.random_state)

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    logger.info('no. train: {:,}'.format(X_train.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # get model
    model, param_grid = get_model(args)
    logger.info('\nmodel: {}, param_grid: {}'.format(args.model, param_grid))

    # tune the model
    start = time.time()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=args.val_frac,
                                                      random_state=args.random_state)

    cv_results = []
    param_dicts = list(product_dict(**param_grid))
    for i, param_dict in enumerate(param_dicts):
        temp_model = clone(model).set_params(**param_dict).fit(X_train, y_train)

        if args.model in ['constant', 'knn', 'kgbm', 'pgbm']:  # optimize MSE
            y_val_hat = temp_model.predict(X_val)
            param_dict['score'] = mean_squared_error(y_val, y_val_hat)

        else:  # optimize NLL
            assert args.model == 'ngboost'
            y_val_dist = temp_model.pred_dist(X_val)
            loc_val, scale_val = y_val_dist.params['loc'], y_val_dist.params['scale']
            scale_val = include_scale_bias(scale=scale_val, y=y_val, op=args.scale_bias)
            param_dict['score'] = util.eval_normal(y=y_val, loc=loc_val, scale=scale_val, nll=True)

        logger.info(f'[{i + 1:,}/{len(param_dicts):,}] {param_dict}'
                    f', cum. time: {time.time() - start:.3f}s')
        cv_results.append(param_dict)

    df = pd.DataFrame(cv_results).sort_values('score', ascending=True)  # lower is better
    logger.info(f'\ngridsearch results:\n{df}')

    best_params = df.astype(object).iloc[0].to_dict()
    del best_params['score']
    logger.info(f'\nbest params: {best_params}')

    tune_time = time.time() - start
    logger.info('\nRESULTS\n\ntune time: {:.3f}s'.format(tune_time))

    start = time.time()
    model = clone(model).set_params(**best_params).fit(X_train, y_train)
    train_time = time.time() - start
    logger.info(f'train time: {train_time:.3f}s')

    # tune k
    if args.model == 'kgbm':
        tree_params = model.get_params()
        start = time.time()
        model = KGBM().fit(model, X_train, y_train, X_val=X_val, y_val=y_val)
        tune_time_kgbm = time.time() - start
        logger.info(f'[KGBM] k: {model.k_}, tune time: {tune_time_kgbm:.3f}s')
        loc, scale = model.pred_dist(X_test)
        total_build_time = tune_time + train_time + tune_time_kgbm
    else:
        total_build_time = tune_time + train_time

    # get location and scale
    start = time.time()

    if args.model == 'constant':
        loc = model.predict(X_test)
        scale = np.full(len(X_test), np.std(y_train), dtype=np.float32)

    elif args.model == 'kgbm':
        loc, scale = model.pred_dist(X_test)
        scale += EPSILON

    elif args.model == 'knn':
        loc = np.zeros(len(X_test), dtype=np.float32)
        scale = np.zeros(len(X_test), dtype=np.float32)

        for i in range(len(X_test)):
            train_idxs = model.kneighbors(X_test[[i]], return_distance=False).flatten()  # shape=(len(X),)
            loc[i] = np.mean(y_train[train_idxs])
            scale[i] = np.std(y_train[train_idxs]) + EPSILON

            if (i + 1) % 100 == 0 and args.verbose > 0:
                logger.info(f'[KGBM - predict]: {i + 1:,} / {len(X_test):,}')

    elif args.model == 'pgbm':
        y_dist = model.predict_dist(X_test, n_forecasts=1000)
        loc, scale = np.mean(y_dist, axis=0), np.std(y_dist, axis=0)

    elif args.model == 'ngboost':
        y_dist = model.pred_dist(X_test)
        loc, scale = y_dist.params['loc'], y_dist.params['scale']

    # include scale bias
    scale = include_scale_bias(scale=scale, y=y_val, op=args.scale_bias)
    total_predict_time = time.time() - start

    # display times
    logger.info(f'\ntotal build time: {total_build_time:.3f}s')
    logger.info(f'total predict time: {total_predict_time:.3f}s')

    # evaluate
    rmse, mae = util.eval_pred(y_test, model=model, X=X_test, logger=None, prefix=args.model)
    nll, crps = util.eval_normal(y=y_test, loc=loc, scale=scale, nll=True, crps=True)
    logger.info(f'CRPS: {crps:.5f}, NLL: {nll:.5f}, RMSE: {rmse:.5f}, MAE: {mae:.5f}')

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

    # save results
    result = {}
    result['model'] = args.model
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
    result['val_frac'] = args.val_frac
    if args.model == 'kgbm':
        result['tree_params'] = tree_params
        result['tune_time_kgbm'] = tune_time_kgbm

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB): {result['max_rss_MB']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results to {os.path.join(out_dir, 'results.npy')}")

    plt.savefig(os.path.join(out_dir, 'test_pred.png'), bbox_inches='tight')
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # create method identifier
    method_name = get_method_identifier(args)

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           method_name)

    # create outut directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/predictive_performance/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='concrete')

    # Experiment settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--model', type=str, default='kgbm')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--scale_bias', type=str, default=None)

    # Tuning settings
    parser.add_argument('--val_frac', type=float, default=0.2)

    # Fixed hyperparameters
    parser.add_argument('--weights', type=str, default='uniform')
    parser.add_argument('--affinity', type=str, default='uniform')

    # Extra settings
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()
    main(args)
