import numpy as np
import pandas as pd

from . import util
from .tree import Tree


def parse_skhgbm_ensemble(model, lt_op=0, is_float32=False):
    """
    Parse SKLearn HistGBM model using its underlying tree representations.

    Reference
        - https://scikit-learn.org/stable/modules/generated/\
            sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
        - https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier

    Note
        - RandomForestRegressor and RandomForestClassifier, `_predictors`
            is a list of lists of shape=(no. estimators, no .classes).
        - Each tree's `nodes` attribute is a multitype array that can
            be converted into a pd.DataFrame.
    """

    # validation
    model_params = model.get_params()

    estimators = model._predictors
    n_boost = len(estimators)
    n_class = len(estimators[0])
    trees = np.zeros((n_boost, n_class), dtype=np.dtype(object))

    scale = 1.0

    for i in range(n_boost):  # per boost
        for j in range(n_class):  # per class
            nodes_df = pd.DataFrame(estimators[i][j].nodes)
            left_children_arr = np.array(nodes_df['left'], dtype=np.int32)
            right_children_arr = np.array(nodes_df['right'], dtype=np.int32)
            feature_idx_arr = np.array(nodes_df['feature_idx'], dtype=np.int32)
            threshold_arr = np.array(nodes_df['num_threshold'], dtype=np.float64)
            leaf_vals_arr = np.array(nodes_df['value'], dtype=np.float64)

            # fill unused values with -1
            decision_node_idxs = np.array(nodes_df[nodes_df['is_leaf'] == 0].index, np.int32)
            leaf_idxs = np.array(nodes_df[nodes_df['is_leaf'] == 1].index, np.int32)

            left_children_arr[leaf_idxs] = -1
            right_children_arr[leaf_idxs] = -1
            feature_idx_arr[leaf_idxs] = -1
            threshold_arr[leaf_idxs] = -1
            leaf_vals_arr[decision_node_idxs] = -1

            # create tree
            children_left = list(left_children_arr)
            children_right = list(right_children_arr)
            feature = list(feature_idx_arr)
            threshold = list(threshold_arr)
            leaf_vals = list(leaf_vals_arr * scale)
            trees[i][j] = Tree(children_left, children_right, feature, threshold,
                               leaf_vals, lt_op, is_float32)

    # set bias
    bias = 0.0

    # regression
    if model._estimator_type == 'regressor':
        assert model_params['loss'] == 'least_squares'
        bias = model._baseline_prediction  # target mean
        objective = 'regression'
        factor = 0

    # classification
    else:
        assert model._estimator_type == 'classifier'
        n_class = model.classes_.shape[0]

        # binary
        if n_class == 2:
            bias = model._baseline_prediction  # log space
            objective = 'binary'
            factor = 0

        # multiclass
        else:
            assert n_class > 2
            bias = list(model._baseline_prediction.flatten())  # log space
            objective = 'multiclass'
            factor = 1.0

    params = {}
    params['bias'] = bias
    params['learning_rate'] = model_params['learning_rate']
    params['l2_leaf_reg'] = model_params['l2_regularization']
    params['objective'] = objective
    params['tree_type'] = 'gbdt'
    params['factor'] = factor

    return trees, params


def parse_skgbm_ensemble(model, lt_op=0, is_float32=False):
    """
    Parse SKLearn GBM model using its underlying tree representations.

    Reference
        - https://scikit-learn.org/stable/modules/generated/\
            sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
        - https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

    Note
        - GradientBoostingRegressor and GradientBoostingClassifier, `estimators_`.shape=(no. estimators, no .classes).
        - Each tree's `value`.shape=(no. nodes, 1, 1).
    """

    # validation
    model_params = model.get_params()
    assert model_params['criterion'] == 'friedman_mse'

    estimators = model.estimators_
    trees = np.zeros(estimators.shape, dtype=np.dtype(object))

    scale = model.learning_rate

    for i in range(estimators.shape[0]):  # per tree
        for j in range(estimators.shape[1]):  # per class
            t = estimators[i][j].tree_
            children_left = list(t.children_left)
            children_right = list(t.children_right)
            feature = list(t.feature)
            threshold = list(t.threshold)
            leaf_vals = list(t.value.flatten() * scale)
            trees[i][j] = Tree(children_left, children_right, feature, threshold,
                               leaf_vals, lt_op, is_float32)

    # set bias
    bias = 0.0

    # regression
    if hasattr(model.init_, 'constant_'):
        assert model_params['loss'] == 'ls'  # least squares
        bias = model.init_.constant_.flatten()[0]  # log space
        objective = 'regression'
        factor = 0

    # classification
    else:
        assert model_params['loss'] == 'deviance'
        class_prior = model.init_.class_prior_
        n_class = class_prior.shape[0]

        # binary
        if n_class == 2:
            bias = util.logit(class_prior[1])  # inverse of sigmoid -> log space
            objective = 'binary'
            factor = 0

        # multiclass
        else:
            assert n_class > 2
            bias = list(np.log(class_prior))  # log space
            objective = 'multiclass'
            factor = (n_class) / (n_class - 1)

    params = {}
    params['bias'] = bias
    params['learning_rate'] = model_params['learning_rate']
    params['l2_leaf_reg'] = 0.0
    params['objective'] = objective
    params['tree_type'] = 'gbdt'
    params['factor'] = factor

    return trees, params


def parse_skrf_ensemble(model, lt_op=0, is_float32=False):
    """
    Parse SKLearn RF model using its underlying tree representations.

    Reference
        - https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        - https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

    Note
        - RandomForestRegressor and RandomForestClassifier, `estimators_`.shape=(no. estimators).
        - For multiclass classification, each tree's `value`.shape=(no. nodes, 1, no. classes).
    """
    if model.bootstrap:
        print('WARNING: Parser does not support RF w/ bootstrap')
        print('\t- Parsed model predictions will likely differ significantly from original model.')
        print('\t- Please turn off bootstrapping for correct parsing.')

    estimators = model.estimators_
    n_class = model.n_classes_ if hasattr(model, 'n_classes_') else 0

    if n_class <= 2:  # regression, binary classification
        trees = np.zeros((len(estimators), 1), dtype=np.dtype(object))

    else:  # multiclass
        trees = np.zeros((len(estimators), n_class), dtype=np.dtype(object))

    for i in range(len(estimators)):  # per boosting
        t = estimators[i].tree_
        children_left = list(t.children_left)
        children_right = list(t.children_right)
        feature = list(t.feature)
        threshold = list(t.threshold)

        # regression
        if n_class == 0:
            leaf_vals = list(t.value.flatten())
            trees[i][0] = Tree(children_left, children_right, feature, threshold,
                               leaf_vals, lt_op, is_float32)
            bias = 0.0
            objective = 'regression'
            factor = 0.0

        # binary classification
        elif n_class == 2:
            value = t.value.squeeze()  # value.shape=(no. nodes, 2)
            leaf_vals = (value / value.sum(axis=1).reshape(-1, 1))[:, 1].tolist()
            trees[i][0] = Tree(children_left, children_right, feature, threshold,
                               leaf_vals, lt_op, is_float32)
            bias = 0.0
            objective = 'binary'
            factor = 0.0

        # multiclass classification
        else:
            assert n_class > 2
            value = t.value.squeeze()  # value.shape=(no. nodes, no. classes)
            value /= value.sum(axis=1).reshape(-1, 1)  # normalize
            for j in range(value.shape[1]):  # per class
                leaf_vals = list(value[:, j])
                trees[i][j] = Tree(children_left, children_right, feature, threshold,
                                   leaf_vals, lt_op, is_float32)
            bias = [0.0] * n_class
            objective = 'multiclass'
            factor = (n_class) / (n_class - 1)

    # set bias
    bias = 0.0

    if n_class >= 3:
        bias = [0.0] * n_class

    params = {}
    params['bias'] = bias
    params['learning_rate'] = 0.0
    params['l2_leaf_reg'] = 0.0
    params['objective'] = objective
    params['tree_type'] = 'rf'
    params['factor'] = factor

    return trees, params
