import time
from abc import abstractmethod

import numpy as np

from .parsers import parse_model
from .parsers import util


class Explainer(object):
    """
    Abstract class that all explainers must implement.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, model, X, y, target_labels=None):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: 2d array of training data.
            y: 1d array of training targets.
            target_labels: 1d array of target labels (len(X) or no. unique labels).
                Label estimation (LE) methods only!
        """
        start = time.time()
        self.model_ = parse_model(model, X, y)
        assert self.model_.tree_type in ['rf', 'gbdt']
        assert self.model_.objective in ['regression', 'binary', 'multiclass']
        self.parse_time_ = time.time() - start

    def get_self_influence(self, X, y, batch_size=100):
        """
        - Compute influence of each example on its own test loss.

        Input
            X: 2d array of train data.
            y: 2d array of train targets.
            batch_size: No. examples to compute influence for at one time.

        Return
            - 1d array of shape=(no. train,).
                Arrays are returned in the same order as the traing data.
        """
        start = time.time()

        self_influence = np.zeros(X.shape[0], dtype=util.dtype_t)  # shape=(X.shape[0],)

        n_finish = 0
        n_remain = X.shape[0]

        # compute self-influence in small batches
        while n_remain > 0:
            n_batch = batch_size

            if n_remain < batch_size:
                n_batch = n_remain

            idxs = np.arange(n_finish, n_finish + n_batch)
            X_batch = X[idxs].copy()
            y_batch = y[idxs].copy()

            influence = self.get_local_influence(X_batch, y_batch, verbose=0)  # shape=(X.shape[0], batch_size)
            self_influence[idxs] = np.diag(influence[idxs])

            n_finish += n_batch
            n_remain -= n_batch

            # progress
            if self.logger:
                self.logger.info(f'[INFO] No. finished: {n_finish:>10,} / {X.shape[0]:,}, '
                                 f'cum. time: {time.time() - start:.3f}s')

        return self_influence

    @abstractmethod
    def get_local_influence(self, X, y, target_labels=None):
        """
        - Compute influence of each training instance on the test loss.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.
                Could be the actual label or the predicted label depending on the explainer.
            target_labels: 1d array of new training target label (same length as X).
                Label estimation (LE) methods only!

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        pass
