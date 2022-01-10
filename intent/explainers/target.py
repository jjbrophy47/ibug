import numpy as np

from .base import Explainer
from .parsers import util


class Target(Explainer):
    """
    Explainer that randomly returns higher influence
        for train examples with similar targets to the test examples.

    Local-Influence Semantics
        - More positive values are assigned to train examples
            with similar targets to the test examples.

    Note
        - Supports GBDTs and RFs.
    """
    def __init__(self, random_state=1, logger=None):
        """
        Input
            random_state: int, random seed to enhance reproducibility.
            logger: object, If not None, output to logger.
        """
        self.random_state = random_state
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

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self.n_class_ = self.model_.n_class_
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

        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=util.dtype_t)

        for i in range(X.shape[0]):
            influence[:, i] = self._get_influence(target=y[i], seed=i + 1)

        return influence

    # private
    def _get_influence(self, target, seed=1):
        """
        Input
            seed: seeds the random number generator.

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        rng = np.random.default_rng(self.random_state + seed)

        influence = np.zeros(self.X_train_.shape[0], dtype=util.dtype_t)

        if self.objective_ in ['binary', 'multiclass']:
            target_idxs = np.where(self.y_train_ == target)[0]
            non_target_idxs = np.where(self.y_train_ != target)[0]

            influence[target_idxs] = rng.uniform(0.0, 1.0, size=len(target_idxs))
            influence[non_target_idxs] = rng.uniform(-1.0, 0.0, size=len(non_target_idxs))

        # assigns more positive values to examples with targets close to the test target
        else:
            assert self.objective_ == 'regression'
            diffs = np.abs(self.y_train_ - target)
            influence[:] = (1.0 / diffs) + rng.normal(0, np.std(diffs))

        return influence
