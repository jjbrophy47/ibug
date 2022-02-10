import numpy as np

from . import util
from .tree import Tree


def parse_lgb_ensemble(model, X, y):
    """
    Parse LightGBM model based on its json representation.

    Input
        model: LightGBM tree-ensemble.
        X: 2d array of train data.
        y: 1d array of targets.
    """

    # validate
    model_params = model.get_params()
    assert model_params['reg_alpha'] == 0
    assert model_params['class_weight'] is None
    assert model_params['boosting_type'] == 'gbdt'

    n_class = model.classes_.shape[0] if hasattr(model, 'classes_') else 0

    if n_class == 0:  # regression
        assert model.objective_ == 'regression'
        objective = 'regression'
        factor = 0.0
        bias = np.mean(y)
        initial_guess = bias

    elif n_class == 2:  # binary
        assert model.objective_ == 'binary'
        bias = 0.0
        objective = 'binary'
        factor = 0.0
        bias = util.logit(np.mean(y))
        initial_guess = bias

    else:  # multiclass
        assert n_class > 2
        assert model.objective_ == 'multiclass'
        objective = 'multiclass'
        factor = (n_class) / (n_class - 1)
        _, class_count = np.unique(y, return_counts=True)
        bias = np.log(class_count / np.sum(class_count))
        initial_guess = bias

    # parse trees
    tree_list = []
    json_data = _get_json_data_from_lgb_model(model)
    for i, tree_dict in enumerate(json_data):
        parsed_tree = _parse_lgb_tree(tree_dict, tree_index=i, n_class=n_class, initial_guess=initial_guess)
        tree_list.append(parsed_tree)
    trees = np.array(tree_list, dtype=np.dtype(object))

    if n_class > 2:  # reshape multiclass trees
        n_trees = int(trees.shape[0] / n_class)
        trees = trees.reshape((n_trees, n_class))

    else:  # reshape regression and binary trees
        trees = trees.reshape(-1, 1)  # shape=(no. tree, 1)

    params = {}
    params['bias'] = bias
    params['learning_rate'] = model_params['learning_rate']
    params['l2_leaf_reg'] = model_params['reg_lambda']
    params['objective'] = objective
    params['tree_type'] = 'gbdt'
    params['factor'] = factor

    return trees, params


# private
def _parse_lgb_tree(tree_dict, tree_index, n_class, initial_guess, lt_op=0, is_float32=False):
    """
    Data has format:
    {
        ...
        'tree_structure': {
            'split_feature': int,
            'threshold': float,
            'left child': dict
            'right_child': dict,
            ...
        }
    }

    IF 'left_child' or 'right_child' is a leaf, the dict is:
    {
        'leaf_index': int,
        'leaf_value': float,
        'leaf_weight': int,
        'leaf_count': int
    }

    Notes:
        - The structure is given as recursive dicts.

    Traversal:
        - Breadth-first.

    Desired format:
        https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

    Returns one or a list of Trees (one for each class).
    """

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    node_dict = tree_dict['tree_structure']

    # add root node
    if 'leaf_value' in node_dict:  # leaf
        leaf_val = node_dict['leaf_value']
        leaf_val = _update_leaf_value(leaf_val, tree_index, n_class, initial_guess)
        leaf_vals.append(leaf_val)
        feature.append(-1)
        threshold.append(-1)
        node_dict['left_child'] = None
        node_dict['right_child'] = None

    else:  # decision node
        assert node_dict['decision_type'] == '<='
        assert node_dict['default_left'] is True
        leaf_vals.append(-1)
        feature.append(node_dict['split_feature'])
        threshold.append(node_dict['threshold'])

    node_id = 1
    stack = [(node_dict['left_child'], 1), (node_dict['right_child'], 0)]

    while len(stack) > 0:
        node_dict, is_left = stack.pop(0)

        if node_dict is None:
            if is_left:
                children_left.append(-1)
            else:
                children_right.append(-1)

        else:

            if is_left:
                children_left.append(node_id)
            else:
                children_right.append(node_id)

            if 'split_index' in node_dict:  # split node
                assert node_dict['decision_type'] == '<='
                assert node_dict['default_left'] is True
                feature.append(node_dict['split_feature'])
                threshold.append(node_dict['threshold'])
                leaf_vals.append(-1)
                stack.append((node_dict['left_child'], 1))
                stack.append((node_dict['right_child'], 0))

            else:  # leaf node
                feature.append(-1)
                threshold.append(-1)
                leaf_val = node_dict['leaf_value']
                leaf_val = _update_leaf_value(leaf_val, tree_index, n_class, initial_guess)
                leaf_vals.append(leaf_val)
                stack.append((None, 1))
                stack.append((None, 0))

            node_id += 1

    result = Tree(children_left, children_right, feature, threshold, leaf_vals, lt_op, is_float32)

    return result


def _get_json_data_from_lgb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'LGBM' in str(model)
    json_data = model.booster_.dump_model()['tree_info']  # 1d list of tree dicts
    return json_data


def _update_leaf_value(leaf_val, tree_index, n_class, initial_guess):
    """
    Subtract initial guess from initial tree (tree_index == 0) or first k (no. classes)
        trees for multiclass to be consistent with other modern GBDT implementations.

    Input
        leaf_val: float, non-updated leaf value.
        tree_index: int, boosting iteration.
        n_class: int, no. classes (0 - regression, 2 - binary, >2 - multiclass).
        initial_guess: float or 1d array (mean - regression, class prior - classification).

    Returns leaf value with initial guess subtracted from its value if it is initial tree.
    """

    if n_class <= 2 and tree_index == 0:  # regression or binary
        leaf_val -= initial_guess

    elif n_class > 2 and tree_index < n_class:  # multiclass
        leaf_val -= initial_guess[tree_index]

    return leaf_val
