import numpy as np

from .base import Explainer
from .parsers import util


class Minority(Explainer):
    """
    Explainer that randomly returns higher influence
        values for minority class train examples.

    Local-Influence Semantics
        - More positive values are assigned to minority classes
            than majority classes.

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

        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=util.dtype_t)

        for i in range(X.shape[0]):
            influence[:, i] = self._get_influence(seed=i + 1)

        return influence

    # private
    def _get_influence(self, seed=1):
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
            classes, counts = np.unique(self.y_train_, return_counts=True)

            class_order = np.argsort(counts)

            v = [0, 1]
            for class_idx in class_order:
                c = classes[class_idx]
                idxs = np.where(self.y_train_ == c)[0]
                influence[idxs] = rng.uniform(v[0], v[1], size=len(idxs))
                v[0] -= 1
                v[1] -= 1

        # assigns more positive values to examples with targets far from the median
        else:
            assert self.objective_ == 'regression'
            diffs = np.abs(self.y_train_ - np.median(self.y_train_))
            influence[:] = diffs + rng.normal(0, np.std(diffs))

        return influence
