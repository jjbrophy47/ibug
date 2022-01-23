import os
import json
import shutil
from uuid import uuid4

import numpy as np

from .tree import Tree


def parse_cb_ensemble(model):
    """
    Parse CatBoost model based on its json representation.
    """

    # validation
    model_params = model.get_all_params()
    assert model_params['leaf_estimation_iterations'] == 1
    assert model_params['leaf_estimation_method'] == 'Newton'

    # parsing
    n_class = model.classes_.shape[0]
    json_data = _get_json_data_from_cb_model(model)
    trees = np.array([_parse_cb_tree(tree_dict, n_class) for tree_dict in json_data], dtype=np.dtype(object))

    # regression
    if n_class == 0:
        assert model_params['loss_function'] == 'RMSE'
        objective = 'regression'
        bias = model.get_scale_and_bias()[1]  # log space
        factor = 0.0
        trees = trees.reshape(-1, 1)  # shape=(no. tree, 1)

    elif n_class == 2:
        assert model_params['loss_function'] == 'Logloss'
        objective = 'binary'
        bias = model.get_scale_and_bias()[1]  # log space
        factor = 0.0
        trees = trees.reshape(-1, 1)  # shape=(no. tree, 1)

    else:
        assert n_class > 2
        assert model_params['loss_function'] == 'MultiClass'
        objective = 'multiclass'
        bias = model.get_scale_and_bias()[1]
        factor = (n_class) / (n_class - 1)

    params = {}
    params['bias'] = bias
    params['learning_rate'] = model_params['learning_rate']
    params['l2_leaf_reg'] = model_params['l2_leaf_reg']
    params['objective'] = objective
    params['tree_type'] = 'gbdt'
    params['factor'] = factor

    return trees, params


# private
def _parse_cb_tree(tree_dict, n_class, lt_op=0, is_float32=False):
    """
    Data has format:
    {
        'splits': [{'feature_idx': int}, ...]
        'leaf_values': [float, float, ...]
    }

    If multiclass, then 'leaf_values' has shape=(no. leaves in one tree * no. class).

    Notes:
        - No. leaves = 2 ^ no. splits.
        - There is only one split condition PER LEVEL in CB trees.
        - 'split' list is given bottom up (need to reverse splits list).

    Returns one tree if binary class., otherwise returns a list of trees,
    one for each class.
    """
    _validate_data(tree_dict, n_class)

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    if n_class > 2:
        leaf_vals = [[] for j in range(n_class)]

    node_id = 0
    for depth, split_dict in enumerate(reversed(tree_dict['splits'])):

        for i in range(2 ** depth):
            feature.append(split_dict['float_feature_index'])
            threshold.append(split_dict['border'])

            if n_class > 2:
                for j in range(n_class):
                    leaf_vals[j].append(-1)
            else:
                leaf_vals.append(-1)  # arbitrary

            if depth > 0:
                if i % 2 == 0:
                    children_left.append(node_id)
                else:
                    children_right.append(node_id)

            node_id += 1

    # leaf nodes
    for i in range(2 ** (depth + 1)):
        feature.append(-1)  # arbitrary
        threshold.append(-1)  # arbitrary

        if n_class > 2:
            for j in range(n_class):
                leaf_vals[j].append(tree_dict['leaf_values'][(i * n_class) + j])
        else:
            leaf_vals.append(tree_dict['leaf_values'][i])

        if i % 2 == 0:
            children_left.append(node_id)  # leaf
        else:
            children_right.append(node_id)  # leaf

        node_id += 1

    # fill in rest of nodes
    for i in range(2 ** (depth + 1)):
        children_left.append(-1)
        children_right.append(-1)

    # leaf_vals may be a list of lists, go through each one and make a tree for each one
    if n_class > 2:
        result = [Tree(children_left, children_right, feature, threshold,
                       leaf_vals[j], lt_op, is_float32) for j in range(n_class)]
    else:
        result = Tree(children_left, children_right, feature, threshold,
                      leaf_vals, lt_op, is_float32)

    return result


def _get_json_data_from_cb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'CatBoost' in str(model)
    here = os.path.abspath(os.path.dirname(__file__))

    temp_dir = os.path.join(here, f'temp_{uuid4()}')
    os.makedirs(temp_dir, exist_ok=True)

    temp_model_json_fp = os.path.join(temp_dir, 'model.json')

    model.save_model(temp_model_json_fp, format='json')

    with open(temp_model_json_fp) as f:
        json_data = json.load(f)

    shutil.rmtree(temp_dir)

    return json_data['oblivious_trees']


def _validate_data(data_json, n_class):
    """
    Checks to make sure JSON data is valid.
    """
    for split in data_json['splits']:
        assert isinstance(split['float_feature_index'], int)
        assert isinstance(split['border'], (int, float))

    for value in data_json['leaf_values']:
        assert isinstance(value, (int, float, list, tuple))

    num_splits = len(data_json['splits'])
    num_values = len(data_json['leaf_values'])

    if n_class > 2:
        assert num_values == 2 ** num_splits * n_class
    else:
        assert num_values == 2 ** num_splits
