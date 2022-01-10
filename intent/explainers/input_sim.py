import numpy as np
from numpy.linalg import norm

from .base import Explainer
from .parsers import util


class InputSim(Explainer):
    """
    Explainer that returns higher influence for train examples with
        larger similarity (in input space).

    Local-Influence Semantics
        - Positive values are assigned to train examples with
            the same label as the test example, scaled by similarity;
            negative values  if opposite labels.

    Note
        - Supports GBDTs and RFs.
    """
    def __init__(self, measure='euclidean', logger=None):
        """
        Input
            measure: str, Similarity metric to use.
                'dot_prod': Dot product between examples.
                'cosine': Cosine similarity between examples.
                'euclidean': Similarity is defined as 1 / euclidean distance.
            logger: object, If not None, output to logger.
        """
        assert measure in ['dot_prod', 'cosine', 'euclidean']
        self.measure = measure
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

        self.original_model_ = model
        self.X_train_ = X.copy()
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

        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=util.dtype_t)

        for test_idx in range(X.shape[0]):

            # compute similarity to the test example
            if self.measure == 'dot_prod':
                sim = np.dot(self.X_train_, X[test_idx])  # shape=(no. train,)

            elif self.measure == 'cosine':
                normalizer = (norm(self.X_train_, axis=1) * norm(X[test_idx]))
                sim = np.dot(self.X_train_, X[test_idx]) / normalizer  # shape=(no. train,)

            else:
                assert self.measure == 'euclidean'
                with np.errstate(divide='ignore'):
                    sim = 1.0 / norm(self.X_train_ - X[test_idx], axis=1)  # shape=(no. train,)
                    sim = np.nan_to_num(sim)  # value is inf. for any examples that are the same as training

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
