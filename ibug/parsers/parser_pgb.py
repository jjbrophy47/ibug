import numpy as np

from .tree import Tree


def parse_pgb_ensemble(model, lt_op=0, is_float32=False):
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
                                         learning_rate=learning_rate)

    params = {}
    params['bias'] = bias
    params['learning_rate'] = learning_rate
    params['l2_leaf_reg'] = l2_leaf_reg
    params['objective'] = 'regression'
    params['tree_type'] = 'gbdt'
    params['factor'] = 0.0

    return trees, params


def parse_pgb_tree(bins, nodes_idx, nodes_split_bin, nodes_split_feature,
                   leaves_idx, leaves_mu, learning_rate):
    """
    Parse a PGBM tree and return a standardized tree representation.
    """
    pass
