import time

import numpy as np

from .base import Explainer
from .parsers import util


class LeafInfluenceSPLE(Explainer):
    """
    Efficient implementation of LeafInfluence (single point) method
        w/ label estimation.

    Local-Influence Semantics
        - Inf.(x_i, x_t) ~= L(y, F_{w/o x_i w/ y*}(x_t)) - L(y, F(x_t))
        - Pos. value means removing x_i + adding x_i with label y* increases loss (original x_i helpful).
        - Neg. value means removing x_i + adding x_i with label y* decreases loss (original x_i harmful).

    Reference
        - https://github.com/frederick0329/TracIn

    Paper
        - https://arxiv.org/abs/2002.08484

    Note
        - Only support GBDTs.
    """
    def __init__(self, logger=None):
        """
        Input
            logger: object, If not None, output to logger.
        """
        self.logger = logger

    def fit(self, model, X, y, target_labels=None):
        """
        - Convert model to internal standardized tree structure.
        - Precompute gradients and leaf indices for each x in X.

        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
            target_labels: unused, for compatibility.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        assert self.model_.tree_type != 'rf', 'RF not supported for BoostIn2'

        self.n_train_ = X.shape[0]
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        self.train_leaf_dvs_ = self._compute_leaf_derivatives(X, y)  # (X.shape[0], n_boost, n_class)
        self.train_leaf_idxs_ = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        return self

    def get_local_influence(self, X, y, target_labels=None, verbose=1):
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
        start = time.time()

        X, y = util.check_data(X, y, objective=self.model_.objective)

        # result container, shape=(no. train, no. test, no. class)
        influence = np.zeros((self.n_train_, X.shape[0]), dtype=util.dtype_t)

        # get change in leaf derivatives and test prediction derivatives
        # train_leaf_dvs = self.train_leaf_dvs_  # (no. train, no. boost, no. class)
        test_gradients = self._compute_final_gradients(X, y)  # shape=(X.shape[0], no. class)

        # get leaf indices each example arrives in
        train_leaf_idxs = self.train_leaf_idxs_  # shape=(no. train, no. boost, no. class)
        test_leaf_idxs = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        # label estimation
        if target_labels is not None:
            assert target_labels.shape == (X.shape[0],)

            i = 0
            for target_label in np.unique(target_labels):
                test_idxs = np.where(target_labels == target_label)[0]

                new_y = np.full(train_leaf_idxs.shape[0], target_label, dtype=util.dtype_t)
                train_leaf_dvs = self._compute_leaf_derivatives(self.X_train_, self.y_train_,
                                                                new_y=new_y)  # (n_train, n_boost, n_class)

                for idx in test_idxs:
                    mask = np.where(train_leaf_idxs == test_leaf_idxs[idx], 1, 0)  # shape=(n_train, n_boost, n_class)
                    weighted_train_leaf_dvs = np.sum(train_leaf_dvs * mask, axis=1)  # shape=(n_train, n_class)
                    prod = -test_gradients[i] * weighted_train_leaf_dvs  # shape=(n_train, n_boost, n_class)

                    # sum over boosts and classes
                    influence[:, idx] = np.sum(prod, axis=1)  # shape=(no. train,)

                    # progress
                    i += 1
                    if i > 0 and (i + 1) % 10 == 0 and self.logger and verbose:
                        self.logger.info(f'[INFO - LeafInfluenceSPLE] No. finished: {i+1:>10,} / {X.shape[0]:>10,}, '
                                         f'cum. time: {time.time() - start:.3f}s')

        # removal estimation
        else:
            train_leaf_dvs = self.train_leaf_dvs_  # (n_train, n_boost, n_class)

            # compute attributions for each test example
            for i in range(X.shape[0]):
                mask = np.where(train_leaf_idxs == test_leaf_idxs[i], 1, 0)  # shape=(n_train, n_boost, n_class)
                weighted_train_leaf_dvs = np.sum(train_leaf_dvs * mask, axis=1)  # shape=(n_train, n_class)
                prod = -test_gradients[i] * weighted_train_leaf_dvs  # shape=(n_train, n_boost, n_class)

                # sum over boosts and classes
                influence[:, i] = np.sum(prod, axis=1)  # shape=(n_train,)

                # progress
                if i > 0 and (i + 1) % 10 == 0 and self.logger and verbose:
                    self.logger.info(f'[INFO - LeafInfluenceSPLE] No. finished: {i+1:>10,} / {X.shape[0]:>10,}, '
                                     f'cum. time: {time.time() - start:.3f}s')

        return influence

    # private
    def _compute_final_gradients(self, X, y):
        """
        - Compute gradients for all instances for the final predictions.

        Input
            X: 2d array of train examples.
            y: 1d array of train targets.

        Return
            - 3d array of shape=(X.shape[0], no. class).
        """
        n_train = X.shape[0]

        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        bias = self.model_.bias

        current_approx = np.tile(bias, (n_train, 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)

        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):
            for class_idx in range(n_class):
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

        gradients = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)

        return gradients

    def _compute_leaf_derivatives(self, X, y, new_y=None):
        """
        - Compute leaf derivatives for all train instances across all boosting iterations.

        Input
            X: 2d array of train examples.
            y: 1d array of train targets.
            new_y: 1d array of new train targets.

        Return
            - 3d array of shape=(X.shape[0], no. boost, no. class).

        Note
            - It is assumed that the leaf estimation method is 'Newton'.
        """
        n_train = X.shape[0]

        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        bias = self.model_.bias
        l2_leaf_reg = self.model_.l2_leaf_reg
        lr = self.model_.learning_rate

        # get leaf info
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)
        leaf_idxs = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        # intermediate container
        current_approx = np.tile(bias, (n_train, 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)

        # result container
        leaf_dvs = np.zeros((n_train, n_boost, n_class), dtype=util.dtype_t)  # shape=(X.shape[0], n_boost, n_class)
        j = np.zeros((n_train, n_class), dtype=util.dtype_t)  # shape=(X.shape[0], no. class)

        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):

            g = self.loss_fn_.gradient(y, current_approx)  # shape=(no. train, no. class)
            h = self.loss_fn_.hessian(y, current_approx)  # shape=(no. train, no. class)
            k = self.loss_fn_.third(y, current_approx)  # shape=(no. train, no. class)

            if new_y is not None:
                g2 = self.loss_fn_.gradient(new_y, current_approx)  # shape=(no. train, no. class)
                h2 = self.loss_fn_.hessian(new_y, current_approx)  # shape=(no. train, no. class)
                k2 = self.loss_fn_.third(new_y, current_approx)  # shape=(no. train, no. class)

            for class_idx in range(n_class):
                leaf_count = leaf_counts[boost_idx, class_idx]
                leaf_vals = trees[boost_idx, class_idx].get_leaf_values()  # shape=(no. leaves,)

                for leaf_idx in range(leaf_count):
                    leaf_docs = np.where(leaf_idx == leaf_idxs[:, boost_idx, class_idx])[0]

                    # compute leaf derivative w.r.t. each train example in `leaf_docs` with OLD label
                    num1a = g[leaf_docs, class_idx] + leaf_vals[leaf_idx] * h[leaf_docs, class_idx] / lr  # (no. docs,)
                    num1b = h[leaf_docs, class_idx] + leaf_vals[leaf_idx] * k[leaf_docs, class_idx] / lr  # (no. docs,)
                    num1 = num1a + (num1b * j[leaf_docs, class_idx])  # (no. docs,)
                    denom1 = np.sum(h[leaf_docs, class_idx]) + l2_leaf_reg
                    leaf_dvs1 = -num1 / denom1 * lr  # shape=(no. docs,)

                    # label estimation
                    if new_y is not None:

                        # compute leaf derivative w.r.t. each train example in `leaf_docs` with NEW label
                        num2a = g2[leaf_docs, class_idx] + leaf_vals[leaf_idx] * h2[leaf_docs, class_idx] / lr  # (doc,)
                        num2b = h2[leaf_docs, class_idx] + leaf_vals[leaf_idx] * k2[leaf_docs, class_idx] / lr  # (doc,)
                        num2 = num2a + (num2b * j[leaf_docs, class_idx])  # (no. docs,)
                        denom2 = denom1 - h[leaf_docs, class_idx] + h2[leaf_docs, class_idx]  # (no. docs,)
                        leaf_dvs2 = -num2 / denom2 * lr  # (no. docs,)

                        leaf_dvs[leaf_docs, boost_idx, class_idx] = (leaf_dvs1 - leaf_dvs2)  # (no. docs,)

                    # removal estimation
                    else:
                        leaf_dvs[leaf_docs, boost_idx, class_idx] = leaf_dvs1  # shape=(no. docs,)

                    # update prediction derivatives
                    j[leaf_docs, class_idx] += leaf_dvs[leaf_docs, boost_idx, class_idx]  # shape=(no. docs,)

                # update approximation
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

        return leaf_dvs
