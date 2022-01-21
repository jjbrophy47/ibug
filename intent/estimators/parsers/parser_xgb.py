import numpy as np

from .tree import Tree


def parse_xgb_ensemble(model):
    """
    Parse XGBoost model based on its string representation.
    """

    # validation
    model_params = model.get_params()
    assert model_params['reg_alpha'] == 0

    string_data = _get_string_data_from_xgb_model(model)
    trees = np.array([_parse_xgb_tree(tree_str) for tree_str in string_data], dtype=np.dtype(object))

    # classification
    if hasattr(model, 'n_classes_'):

        if model.n_classes_ == 2:  # binary
            assert model_params['objective'] == 'binary:logistic'
            assert model_params['scale_pos_weight'] == 1
            trees = trees.reshape(-1, 1)  # shape=(no. trees, 1)
            bias = 0.0  # log space
            objective = 'binary'
            factor = 0.0

        else:
            assert model.n_classes_ > 2
            assert model_params['objective'] == 'multi:softprob'
            assert model_params['scale_pos_weight'] is None
            n_trees = int(trees.shape[0] / model.n_classes_)
            trees = trees.reshape((n_trees, model.n_classes_))
            bias = [0.0] * model.n_classes_  # log space
            objective = 'multiclass'
            factor = 2.0

    else:  # regression
        assert model_params['objective'] == 'reg:squarederror'
        assert model_params['scale_pos_weight'] == 1
        trees = trees.reshape(-1, 1)  # shape=(no. trees, 1)
        bias = model.get_params()['base_score']  # 0.5
        objective = 'regression'
        factor = 0.0

    params = {}
    params['bias'] = bias
    params['learning_rate'] = model_params['learning_rate']
    params['l2_leaf_reg'] = model_params['reg_lambda']
    params['objective'] = objective
    params['tree_type'] = 'gbdt'
    params['factor'] = factor

    return trees, params


# private
def _parse_xgb_tree(tree_str, lt_op=1, is_float32=True):
    """
    Data has format:
    '
        <newlines and tabs><node_id>:[f<feature><<threshold>] yes=<int>,no=<int>,missing=<int>  (decision)
        <newlines and tabs><node_id>:leaf=<leaf_value>  (leaf)
    '

    Notes:
        - The structure is given as a newline and tab-indented string.
        - The node IDs are ordered in a breadth-first manner.

    Desired format:
        https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    """
    # print(tree_str)

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    # each line is a node, the no. preceeding tabs indicates its depth
    lines = tree_str.split('\n')

    # depth-first construction of recursive node dict from tree string representation
    line = lines.pop(0)
    node_dict = _parse_line(line)
    node_dict['left_child'] = _add_node(lines, depth=1)
    node_dict['right_child'] = _add_node(lines, depth=1)

    # add root node
    if node_dict['is_leaf']:  # leaf
        leaf_vals.append(node_dict['leaf_val'])
        feature.append(-1)
        threshold.append(-1)

    else:  # decision node
        leaf_vals.append(-1)
        feature.append(node_dict['feature'])
        threshold.append(node_dict['threshold'])

    node_id = 1
    stack = [(node_dict['left_child'], 1), (node_dict['right_child'], 0)]

    # breadth-first traversal using the recursive node dict
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

            if node_dict['is_leaf']:  # leaf node
                feature.append(-1)
                threshold.append(-1)
                leaf_vals.append(node_dict['leaf_val'])
                stack.append((None, 1))
                stack.append((None, 0))

            else:  # decision node
                feature.append(node_dict['feature'])
                threshold.append(node_dict['threshold'])
                leaf_vals.append(-1)
                stack.append((node_dict['left_child'], 1))
                stack.append((node_dict['right_child'], 0))

            node_id += 1

    result = Tree(children_left, children_right, feature, threshold, leaf_vals, lt_op, is_float32)

    return result


def _add_node(lines, depth):
    """
    Search the remaining lines and parses the first one
    with the specified depth (depth = no. tabs).
    """
    node_dict = None

    for i in range(len(lines)):
        cur_depth = lines[i].count('\t')

        # no more nodes in this direction
        if cur_depth < depth:
            break

        # found more nodes in this direction
        elif cur_depth == depth:
            line = lines.pop(i)
            node_dict = _parse_line(line)
            node_dict['left_child'] = _add_node(lines, depth=depth + 1)
            node_dict['right_child'] = _add_node(lines, depth=depth + 1)
            break

    return node_dict


def _get_string_data_from_xgb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'XGB' in str(model)
    string_data = model.get_booster().get_dump(dump_format='text')  # 1d list of tree strings
    return string_data


def _parse_line(line):
    """
    Parse node string representation and return a dict with appropriate node values.
    """
    res = {}

    if 'leaf' in line:
        res['is_leaf'] = 1
        res['leaf_val'] = _parse_leaf_node_line(line)

    else:
        res['is_leaf'] = 0
        res['feature'], res['threshold'] = _parse_decision_node_line(line)

    return res


def _parse_decision_node_line(line):
    """
    Return feature index and threshold given the string representation of a decision node.
    """
    substr = line[line.find('[') + 1: line.find(']')]
    feature_str, border_str = substr.split('<')
    feature_ndx = int(feature_str[1:])
    border = float(border_str)
    return feature_ndx, border


def _parse_leaf_node_line(line):
    """
    Return the leaf value given the string representation of a leaf node.
    """
    return float(line.split('=')[1])
