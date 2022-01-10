import os
import sys
import logging
import argparse

import numpy as np
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
from intent.explainers import LeafInfluence


def test_global_influence_regression(args, explainer_cls, str_explainer, kwargs):
    print(f'\n***** test_{str_explainer}_global_influence_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=-1)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    global_inf = explainer.get_global_influence(X_test, y_test)

    s_ids = np.argsort(global_inf)[::-1]  # most pos. to most neg.

    print('\ny_mean:', y_train.mean())
    print('sorted_indices           (head):', s_ids[:5])
    print('y_pred           (sorted, head):', tree.predict(X_train)[s_ids][:5])
    print('y_train          (sorted, head):', y_train[s_ids][:5])
    print('global_influence (sorted, head):', global_inf[s_ids][:5])

    status = 'passed' if global_inf.shape[0] == y_train.shape[0] else 'failed'
    print(f'\n{status}')


def test_global_influence_binary(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_global_influence_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    global_inf = explainer.get_global_influence(X_test, y_test)

    s_ids = np.argsort(global_inf)[::-1]  # most pos. to most neg.

    print('\ny_mean:', y_train.mean())
    print('sorted_indices           (head):', s_ids[:5])
    print('y_pred (pos.)    (sorted, head):', tree.predict_proba(X_train)[:, 1][s_ids][:5])
    print('y_train          (sorted, head):', y_train[s_ids][:5])
    print('global_influence (sorted, head):', global_inf[s_ids][:5])

    status = 'passed' if global_inf.shape[0] == y_train.shape[0] else 'failed'
    print(f'\n{status}')


def test_global_influence_multiclass(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_global_influence_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=args.n_class)
    n_class = len(np.unique(y_train))

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    global_inf = explainer.get_global_influence(X_test, y_test)

    s_ids = np.argsort(global_inf)[::-1]  # most pos. to most neg.

    _, class_count = np.unique(y_train, return_counts=True)
    print('\ny_mean:', class_count / np.sum(class_count))
    print('sorted_indices           (head):', s_ids[:5])
    print('y_pred (pos.)    (sorted, head):\n', tree.predict_proba(X_train)[s_ids][:5])
    print('y_train          (sorted, head):', y_train[s_ids][:5])
    print('global_influence (sorted, head):', global_inf[s_ids][:5])

    status = 'passed' if global_inf.shape[0] == y_train.shape[0] else 'failed'
    print(f'\n{status}')


def test_local_influence_regression(args, explainer_cls, explainer_str, kwargs, logger=None):
    print(f'\n***** test_{explainer_str}_local_influence_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=-1)
    test_ids = np.array([0, 1])[:args.n_local]

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])  # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train, index: {test_idx}, pred: {test_pred}, target: {test_label}')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train   (head, sorted):', y_train[s_ids][:5])
        print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_local_influence_binary(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_local_influence_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)
    test_ids = np.array([0, 1])[:args.n_local]

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])   # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:10])
        print('y_train   (head, sorted):', y_train[s_ids][:10])
        print('influence (head, sorted):', influence[s_ids][:10])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_local_influence_multiclass(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_local_influence_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=args.n_class)
    test_ids = np.array([0, 1])[:args.n_local]

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    # local influence shape=(no. test, no. train, no. class)
    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train   (head, sorted):', y_train[s_ids][:5])
        print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


# new label tests

def test_local_influence_regression_new_label(args, explainer_cls, explainer_str, kwargs, logger=None):
    print(f'\n***** test_{explainer_str}_local_influence_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=-1)
    test_ids = np.array([0, 1])[:args.n_local]

    # shuffle training labels
    rng = np.random.default_rng(args.rs)
    new_y_train = y_train.copy()
    rng.shuffle(new_y_train)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train, new_y=new_y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])  # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train, index: {test_idx}, pred: {test_pred}, target: {test_label}')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train     (head, sorted):', y_train[s_ids][:5])
        print('new_y_train (head, sorted):', new_y_train[s_ids][:5])
        print('influence   (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_local_influence_binary_new_label(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_local_influence_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)
    test_ids = np.array([0, 1])[:args.n_local]

    # shuffle training labels
    rng = np.random.default_rng(args.rs)
    new_y_train = y_train.copy()
    rng.shuffle(new_y_train)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train, new_y=new_y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])   # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:10])
        print('y_train     (head, sorted):', y_train[s_ids][:10])
        print('new_y_train (head, sorted):', new_y_train[s_ids][:10])
        print('influence   (head, sorted):', influence[s_ids][:10])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_local_influence_multiclass_new_label(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_local_influence_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=args.n_class)
    test_ids = np.array([0, 1])[:args.n_local]

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    # shuffle training labels
    rng = np.random.default_rng(args.rs)
    new_y_train = y_train.copy()
    rng.shuffle(new_y_train)

    # local influence shape=(no. test, no. train, no. class)
    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train, new_y=new_y_train)
    influences = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(influence)[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train     (head, sorted):', y_train[s_ids][:5])
        print('new_y_train (head, sorted):', new_y_train[s_ids][:5])
        print('influence   (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


# private
def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def _get_test_data(args, n_class=2):
    """
    Return train and test data for the given objective.
    """
    rng = np.random.default_rng(args.rs)
    X_train = rng.standard_normal((args.n_train, args.n_feat))
    X_test = rng.standard_normal((args.n_test, args.n_feat))

    if n_class >= 2:  # classification
        y_train = rng.integers(n_class, size=args.n_train)
        y_test = rng.integers(n_class, size=args.n_test)

    elif n_class == -1:  # reegression
        y_train = rng.uniform(-100, 100, size=args.n_train)
        y_test = rng.uniform(-100, 100, size=args.n_test)

    else:
        raise ValueError(f'invalid n_class: {n_class}')

    return X_train, X_test, y_train, y_test


def _get_model(args):
    """
    Return tree-ensemble.
    """
    if args.tree_type == 'cb':
        class_fn = CatBoostRegressor if args.model_type == 'regressor' else CatBoostClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth,
                        random_state=args.rs, leaf_estimation_iterations=1,
                        logging_level='Silent')

    elif args.tree_type == 'lgb':
        class_fn = LGBMRegressor if args.model_type == 'regressor' else LGBMClassifier
        tree = class_fn(n_estimators=args.n_tree, num_leaves=args.n_leaf, random_state=args.rs)

    elif args.tree_type == 'sgb':
        class_fn = HistGradientBoostingRegressor if args.model_type == 'regressor' else HistGradientBoostingClassifier
        tree = class_fn(max_iter=args.n_tree, max_leaf_nodes=args.n_leaf, random_state=args.rs)

    elif args.tree_type == 'skgbm':
        class_fn = GradientBoostingRegressor if args.model_type == 'regressor' else GradientBoostingClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs)

    elif args.tree_type == 'skrf':
        class_fn = RandomForestRegressor if args.model_type == 'regressor' else RandomForestClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs,
                        bootstrap=False, max_features='sqrt')

    elif args.tree_type == 'xgb':

        if args.model_type == 'regressor':
            tree = XGBRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                                tree_method='hist', random_state=args.rs)

        elif args.model_type == 'binary':
            tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                 random_state=args.rs, tree_method='hist',
                                 use_label_encoder=False, eval_metric='logloss')

        elif args.model_type == 'multiclass':
            tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                 random_state=args.rs, tree_method='hist',
                                 use_label_encoder=False, eval_metric='mlogloss')
        else:
            raise ValueError(f'Unknown model_type {args.model_type}')

    else:
        raise ValueError(f'Unknown tree_type {args.tree_type}')

    return tree
