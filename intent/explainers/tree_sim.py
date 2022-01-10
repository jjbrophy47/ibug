import numpy as np
from numpy.linalg import norm

from .base import Explainer
from .parsers import util


class TreeSim(Explainer):
    """
    Explainer that returns higher influence for train examples with
        the same target and larger similarity in a chosen tree-kernel space.

    Local-Influence Semantics
        - More positive values are assigned to train examples with
            n the same leaf as the test example.

    Note
        - Supports GBDTs and RFs.
    """
    def __init__(self, measure='dot_prod', kernel='lpw', logger=None):
        """
        Input
            measure: str, Similarity metric to use.
                'dot_prod': Dot product between examples.
                'cosine': Cosine similarity between examples.
                'euclidean': Similarity is defined as 1 / euclidean distance.
            kernel: str, Transformation of the input using the tree-ensemble structure.
                'to_': Tree output; output of each tree in the ensemble.
                'lp_': Leaf path; one-hot encoding of leaf indices across all trees.
                'lpw': Weighted leaf path; like 'lp' but replaces 1s with 1 / leaf count.
                'lo_': Leaf output; like 'lp' but replaces 1s with leaf values.
                'low': Weighted leaf otput; like 'lo' but replace leaf value with 1 / leaf value.
                'fp_': Feature path; one-hot encoding of node indices across all trees.
                'fpw': Weighted feature path; like 'fp' but replaces 1s with 1 / node count.
                'fo_': Feature output; like 'fp' but replaces leaf 1s with leaf values.
                'fow': Weighted feature output; like 'fo' but replaces leaf 1s with 1 / leaf values.
            logger: object, If not None, output to logger.
        """
        assert measure in ['dot_prod', 'cosine', 'euclidean']
        assert kernel in ['to_', 'lp_', 'lpw', 'lo_', 'low', 'fp_', 'fpw', 'fo_', 'fow']
        self.measure = measure
        self.kernel = kernel
        self.logger = logger

    def fit(self, model, X, y):
        """
        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.model_.update_node_count(X)

        self.original_model_ = model
        self.X_train_ = self._kernel_transform(X)
        self.y_train_ = y.copy()

        self.objective_ = self.model_.objective

        return self

    def get_local_influence(self, X, y):
        """
        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the training data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        X_test_ = self._kernel_transform(X)  # shape=(X.shape[0], no. feature)

        influence = np.zeros((self.X_train_.shape[0], X_test_.shape[0]), dtype=util.dtype_t)

        for test_idx in range(X.shape[0]):

            # compute similarity to the test example
            if self.measure == 'dot_prod':
                sim = np.dot(self.X_train_, X_test_[test_idx])  # shape=(no. train,)

            elif self.measure == 'cosine':
                normalizer = (norm(self.X_train_, axis=1) * norm(X_test_[test_idx]))
                sim = np.dot(self.X_train_, X_test_[test_idx]) / normalizer  # shape=(no. train,)

            else:
                assert self.measure == 'euclidean'
                with np.errstate(divide='ignore'):
                    sim = 1.0 / norm(self.X_train_ - X_test_[test_idx], axis=1)  # shape=(no. train,)
                    sim = np.nan_to_num(sim)  # value is inf. for any examples the same as training

            # determine if each train example helps or hurts test loss
            if self.objective_ in ['binary', 'multiclass']:
                sgn = np.where(self.y_train_ == y[test_idx], 1.0, -1.0)  # shape=(no. train,)

            else:  # if train and test targets both on same side of the prediction, then pos. influence
                assert self.objective_ == 'regression'
                pred = self.original_model_.predict(X[[test_idx]])
                test_sgn = 1.0 if pred >= y[test_idx] else -1.0
                train_sgn = np.where(self.y_train_ >= pred, 1.0, -1.0)  # shape=(no. train,)
                sgn = np.where(train_sgn != test_sgn, 1.0, -1.0)

            # compute influence
            influence[:, test_idx] = sim * sgn

        return influence

    # private
    def _kernel_transform(self, X):
        """
        Transforms each x in X using the specified tree kernel.

        Return
            - 2d array of shape=(X.shape[0], no. kernel features).
        """
        structure_dict = {'t': 'tree', 'l': 'leaf', 'f': 'feature'}
        output_dict = {'p': 'path', 'o': 'output'}
        weight_dict = {'_': 'unweighted', 'w': 'weighted'}
        
        s1, s2, s3 = list(self.kernel)
        structure = structure_dict[s1]
        output = output_dict[s2]
        weight = weight_dict[s3]

        if structure == 'tree':
            X_ = self._tree_kernel_transform(X)

        elif structure == 'leaf':
            X_ = self._leaf_kernel_transform(X, output=output, weight=weight)

        elif structure == 'feature':
            X_ = self._feature_kernel_transform(X, output=output, weight=weight)

        return X_

    def _tree_kernel_transform(self, X):
        """
        Transform each x in X to be a vector of tree outputs.

        Return
            - Regression and binary: 2d array of shape=(no. train, no. trees).
            - Multiclass: 2d array of shape=(no. train, no. trees * no. class).
        """
        trees = self.model_.trees.flatten()
        X_ = np.zeros((X.shape[0], trees.shape[0]))

        for i, tree in enumerate(trees):
            X_[:, i] = tree.predict(X)

        return X_

    def _leaf_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded leaf paths.
        - The `output` and `weight` parameters control the value of the 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. leaves).
            - Multiclass: 2d array of shape=(no. train, ~total no. leaves * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_leaves = np.sum([tree.leaf_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_leaves))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_leaves = 0
        for tree in trees:
            start = n_prev_leaves
            stop = n_prev_leaves + tree.leaf_count_
            X_[:, start: stop] = tree.leaf_path(X, output=output, weighted=weighted)
            n_prev_leaves += tree.leaf_count_

        return X_

    def _feature_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded feature paths.
        - The `output` parameter controls the value of the leaf 1s.
        - The `weight` parameter controls the value of all 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. nodes).
            - Multiclass: 2d array of shape=(no. train, ~total no. nodes * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_nodes = np.sum([tree.node_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_nodes))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_nodes = 0
        for tree in trees:
            start = n_prev_nodes
            stop = n_prev_nodes + tree.node_count_
            X_[:, start: stop] = tree.feature_path(X, output=output, weighted=weighted)
            n_prev_nodes += tree.node_count_

        return X_
