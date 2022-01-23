import time
from abc import abstractmethod

from .parsers import parse_model


class Estimator(object):
    """
    Abstract class for all uncertainty estimators.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble (LGB, XGBoost, etc.).
            X: 2d array of training data.
            y: 1d array of training targets.
        """
        start = time.time()
        self.model_ = parse_model(model, X, y)
        assert self.model_.tree_type in ['rf', 'gbdt']
        assert self.model_.objective == 'regression'
        self.parse_time_ = time.time() - start
