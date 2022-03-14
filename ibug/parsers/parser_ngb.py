import numpy as np

from .tree import Tree


def parse_ngb_ensemble(model, lt_op=0, is_float32=False):
    """
    Parse NGBoost model.
    """

    # validation
    model_params = model.get_params()
    assert model_params['Base__criterion'] == 'friedman_mse'
    assert str(model_params['Base']).startswith('DecisionTreeRegressor')
    assert 'ngboost.distns.normal.Normal' in str(model.Dist)

    # parsing
    estimators = np.array(model.base_models, dtype=object)  # shape=(n_boost, 1)
    scalings = np.array(model.scalings, dtype=np.float32)  # shape=(n_boost,)
    learning_rate = model.learning_rate  # scalar

    assert estimators.ndim == 2 and scalings.ndim == 1
    assert len(estimators) == len(scalings)

    n_boost = len(estimators)
    n_class = 1
    trees = np.zeros((n_boost, n_class), dtype=np.dtype(object))

    for i in range(n_boost):  # per boost
        for j in range(n_class):  # per class

            t = estimators[i][j].tree_
            scale = scalings[i]

            children_left = list(t.children_left)
            children_right = list(t.children_right)
            feature = list(t.feature)
            threshold = list(t.threshold)
            leaf_vals = list(- t.value.flatten() * scale * learning_rate)
            trees[i][j] = Tree(children_left, children_right, feature, threshold,
                               leaf_vals, lt_op, is_float32)

    params = {}
    params['bias'] = model.init_params[0]
    params['learning_rate'] = learning_rate
    params['l2_leaf_reg'] = 1.0
    params['objective'] = 'regression'
    params['tree_type'] = 'gbdt'
    params['factor'] = 0.0

    return trees, params
