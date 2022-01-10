import numpy as np

from .base import Explainer
from .parsers import util


class Random(Explainer):
    """
    Explainer that returns random influence values.

    Local-Influence Semantics
        - Pos. and neg. values are arbitrary.

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
        self.n_train_ = X.shape[0]
        self.n_class_ = self.model_.n_class_
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def get_local_influence(self, X, y, verbose=1):
        """
        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the training data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)
        influence = self.rng_.standard_normal(size=(self.n_train_, X.shape[0]), dtype=util.dtype_t)
        return influence
