import numpy as np

from .base import Explainer
from .parsers import util


class LeafSim(Explainer):
    """
    Explainer that returns higher influence for train examples with
        the same target and larger similarity in the
        "weighted leaf path" tree-kernel space.

    Local-Influence Semantics
        - More positive values are assigned to train examples with
            higher loss AND are in the same leaf as the test example.

    Note
        - Supports GBDTs and RFs.
        - More efficient version of the TreeSim explainer using the
            'weighted leaf path' tree kernel.
    """
    def __init__(self, logger=None):
        """
        Input
            logger: object, If not None, output to logger.
        """
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Precompute gradients and leaf indices for each x in X.

        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.original_model_ = model
        self.y_train_ = y.copy()
        self.objective_ = self.model_.objective

        self.n_train_ = X.shape[0]

        self.n_boost_ = self.model_.n_boost_
        self.n_class_ = self.model_.n_class_

        self.model_.update_node_count(X)
        self.train_leaves_ = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)
        self.leaf_counts_ = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)
        self.leaf_weights_ = self.model_.get_leaf_weights(-2)  # shape=(total no. leaves,)

        return self

    def get_local_influence(self, X, y):
        """
        - Computes effect of each train example on the loss of the test example.

        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the training data.

        Note
            - Attribute train attribution to the test loss ONLY if the train example
                is in the same leaf(s) as the test example.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        # result container, shape=(X.shape[0], no. train, no. class)
        influence = np.zeros((self.n_train_, X.shape[0]), dtype=util.dtype_t)

        # get leaf indices each example arrives in
        train_leaves = self.train_leaves_  # shape=(no. train, no. boost, no. class)
        train_weights = self._get_leaf_weights(train_leaves)  # shape=(no.train, no boost, no. class)

        test_leaves = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        # compute attributions for each test example
        for i in range(X.shape[0]):
            mask = np.where(train_leaves == test_leaves[i], 1, 0)  # shape=(no. train, no. boost, no. class)
            weighted_mask = train_weights * mask  # shape=(no. train, no. boost, no. class)
            sim = np.sum(weighted_mask, axis=(1, 2))  # shape=(no. train,)

            # determine if each train example helps or hurts test loss
            if self.objective_ in ['binary', 'multiclass']:
                sgn = np.where(self.y_train_ == y[i], 1.0, -1.0)  # shape=(no. train,)

            else:  # if train and test targets both on same side of the prediction, then pos. influence
                assert self.objective_ == 'regression'
                pred = self.original_model_.predict(X[[i]])
                test_sgn = 1.0 if pred >= y[i] else -1.0
                train_sgn = np.where(self.y_train_ >= pred, 1.0, -1.0)  # shape=(no. train,)
                sgn = np.where(train_sgn != test_sgn, 1.0, -1.0)

            influence[:, i] = sim * sgn

        return influence

    def _get_leaf_weights(self, leaf_idxs):
        """
        Retrieve leaf weights given the leaf indices.

        Input
            leaf_idxs: Leaf indices, shape=(no. examples, no. boost, no. class)

        Return
            - 3d array of shape=(no. examples, no. boost, no. class)
        """
        leaf_counts = self.leaf_counts_  # shape=(no. boost, no. class)
        leaf_weights = self.leaf_weights_  # shape=(no. leaves across all trees,)

        # result container
        weights = np.zeros(leaf_idxs.shape, dtype=util.dtype_t)  # shape=(no. examples, no. boost, no. class)

        n_prev_leaves = 0

        for b_idx in range(self.n_boost_):

            for c_idx in range(self.n_class_):
                leaf_count = leaf_counts[b_idx, c_idx]

                weights[:, b_idx, c_idx] = leaf_weights[n_prev_leaves:][leaf_idxs[:, b_idx, c_idx]]

                n_prev_leaves += leaf_count

        return weights
