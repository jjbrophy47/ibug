import numpy as np

from .tree import Tree


def parse_pgb_ensemble(model, lt_op=0, is_float32=True):
    """
    Parse NGBoost model.
    """

    # validation
    model_params = model.get_params()
    assert model_params['derivatives'] == 'exact'
    assert model_params['distribution'] == 'normal'
    assert model_params['metric'] == 'rmse'
    assert model_params['objective'] == 'mse'

    # parsing
    learner = model.learner_
    learning_rate = learner.learning_rate.item()  # scalar
    bias = learner.initial_estimate.item()  # scalar
    l2_leaf_reg = learner.reg_lambda.item()  # scalar

    bins = learner.bins.numpy()  # shape=(n_feature, n_max_bins)
    nodes_idx = learner.nodes_idx.numpy()  # shape=(n_estimators, n_max_leaves)
    nodes_split_bin = learner.nodes_split_bin.numpy()  # shape=(n_estimators, n_max_leaves)
    nodes_split_feature = learner.nodes_split_feature.numpy()  # shape=(n_estimators, n_max_leaves)
    leaves_idx = learner.leaves_idx.numpy()  # shape=(n_estimators, n_max_leaves)
    leaves_mu = learner.leaves_mu.numpy()  # shape=n_estimators, n_max_leaves)

    n_boost = len(nodes_idx)
    n_class = 1
    trees = np.zeros((n_boost, n_class), dtype=np.dtype(object))

    for i in range(n_boost):  # per boost
        for j in range(n_class):  # per class
            trees[i][j] = parse_pgb_tree(bins=bins, nodes_idx=nodes_idx[i],
                                         nodes_split_bin=nodes_split_bin[i],
                                         nodes_split_feature=nodes_split_feature[i],
                                         leaves_idx=leaves_idx[i], leaves_mu=leaves_mu[i],
                                         learning_rate=learning_rate, lt_op=lt_op,
                                         is_float32=is_float32)

    params = {}
    params['bias'] = bias
    params['learning_rate'] = learning_rate
    params['l2_leaf_reg'] = l2_leaf_reg
    params['objective'] = 'regression'
    params['tree_type'] = 'gbdt'
    params['factor'] = 0.0

    return trees, params


def parse_pgb_tree(bins, nodes_idx, nodes_split_bin, nodes_split_feature,
                   leaves_idx, leaves_mu, learning_rate, lt_op=0, is_float32=False):
    """
    Parse a PGBM tree and return a standardized tree representation.
    """
    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    # add root node
    if np.sum(nodes_idx) == 0:  # leaf
        leaf_vals.append(- leaves_mu[0] * learning_rate)
        feature.append(-1)
        threshold.append(-1)

    else:  # decision node
        leaf_vals.append(-1)
        feature.append(nodes_split_feature[0])
        threshold.append(bins[nodes_split_feature[0], nodes_split_bin[0]])

    node_id = 1
    stack = [(node_id * 2, 0), (node_id * 2 + 1, 1)]

    while len(stack) > 0:
        node, is_left = stack.pop(0)

        if node is None:  # leaf child
            if is_left:
                children_left.append(-1)
            else:
                children_right.append(-1)

        else:  # actual node

            if is_left:
                children_left.append(node_id)
            else:
                children_right.append(node_id)

            if node in nodes_idx:  # decision node
                node_idx_arr = np.where(nodes_idx == node)[0]
                assert len(node_idx_arr) == 1
                node_idx = node_idx_arr[0]

                feature.append(nodes_split_feature[node_idx])
                threshold.append(bins[nodes_split_feature[node_idx], nodes_split_bin[node_idx]])
                leaf_vals.append(-1)

                stack.append((node * 2, 0))
                stack.append((node * 2 + 1, 1))

            else:  # leaf node
                leaf_idx_arr = np.where(leaves_idx == node)[0]
                assert len(leaf_idx_arr) == 1
                leaf_idx = leaf_idx_arr[0]

                feature.append(-1)
                threshold.append(-1)
                leaf_vals.append(- leaves_mu[leaf_idx] * learning_rate)

                stack.append((None, 0))
                stack.append((None, 1))

            node_id += 1

    return Tree(children_left, children_right, feature, threshold, leaf_vals, lt_op, is_float32)
