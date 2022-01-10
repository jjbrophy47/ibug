import time
import joblib

import numpy as np
from sklearn.base import clone

from .base import Explainer
from .parsers import util


class LOOLE(Explainer):
    """
    Leave-one-out influence explainer with label estimation
        (leave-one-out + add one in w/ desired label).
        Retrains the model after each train-example operation to get change in loss.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y_t, f_{w/o x_i and w/ y*}(x_t)) - L(y_t, f(x_t))
        - Pos. value means removing + adding x_i (w/ diff. label) increases loss (original x_i helpful).
        - Neg. value means removing + adding x_i (w/ diff. label) decreases loss (original x_i harmful).

    Note
        - Supports both GBDTs and RFs.
        - Supports parallelization.
    """
    def __init__(self, n_jobs=-1, logger=None):
        """
        Input
            n_jobs: int, No. processes to run in parallel.
                -1 means use the no. of available CPU cores.
            logger: object, If not None, output to logger.
        """
        self.n_jobs = n_jobs
        self.logger = logger

    def fit(self, model, X, y, target_labels=None):
        """
        - Fit one model with for each training example,
            with that training example removed.

        Note
            - Very memory intensive to save all models,
                may have to switch to a streaming approach.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
            target_labels: 1d array of new training targets.
                Unused, for compatibility.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.n_class_ = self.model_.n_class_
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.objective_ = self.model_.objective

        # select no. processes to run in parallel
        if self.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert self.n_jobs >= 1
            n_jobs = min(self.n_jobs, joblib.cpu_count())

        self.n_jobs_ = n_jobs
        self.original_model_ = model

        return self

    def get_local_influence(self, X, y, target_labels=None, verbose=1):
        """
        - Compute influence of each training instance on each test loss.

        Input
            X: 2d array of test data.
            y: 1d array of test targets

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)
        return self._run_loo(X_test=X, y_test=y, target_labels=target_labels)

    # private
    def _run_loo(self, X_test=None, y_test=None, target_labels=None):
        """
        - Retrain model for each tain example and measure change in train/test loss.

        Return
            - 2d array of average marginals, shape=(no. train, 1 or X_test.shape[0]).
                * Arrays are returned in the same order as the traing data.
        """
        X_train = self.X_train_
        y_train = self.y_train_
        loss_fn = self.loss_fn_
        n_jobs = self.n_jobs_
        original_model = self.original_model_
        objective = self.objective_

        start = time.time()
        if self.logger:
            self.logger.info('\n[INFO - LOOLE] computing values...')
            self.logger.info(f'[INFO - LOOLE] no. cpus: {n_jobs:,}...')

        # fit each model in parallel
        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            # result container
            original_loss = _get_loss(loss_fn, original_model, objective, X=X_test, y=y_test)  # (X_test.shape[0],)
            influence = np.zeros((0, X_test.shape[0]), dtype=util.dtype_t)

            # trackers
            fits_completed = 0
            fits_remaining = X_train.shape[0]

            # get number of fits to perform for this iteration
            while fits_remaining > 0:
                n = min(100, fits_remaining)

                results = parallel(joblib.delayed(_run_iteration)
                                                 (original_model, X_train, y_train, train_idx, X_test, y_test,
                                                  loss_fn, objective, original_loss,
                                                  target_labels) for train_idx in range(fits_completed,
                                                                                        fits_completed + n))

                # synchronization barrier
                results = np.vstack(results)  # shape=(n, X_test.shape[0])
                influence = np.vstack([influence, results])

                fits_completed += n
                fits_remaining -= n

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO - LOOLE] fits: {fits_completed:,} / {X_train.shape[0]:,}'
                                     f', cum. time: {cum_time:.3f}s')

        return influence


def _run_iteration(model, X_train, y_train, train_idx, X_test, y_test,
                   loss_fn, objective, original_loss, target_labels):
    """
    Fit model after leaving out the specified `train_idx` train example.

    Return
        - 1d array of shape=(X_test.shape[0],) or single float.

    Note
        - Parallelizable method.
    """
    loss = np.zeros(original_loss.shape, dtype=util.dtype_t)  # shape=(X_test.shape[0],)

    # label estimation
    if target_labels is not None:
        new_X_train = X_train

        # train a model after editing the label of this training example
        for target_label in np.unique(target_labels):
            new_y_train = y_train.copy()
            new_y_train[train_idx] = target_label
            new_model = clone(model).fit(new_X_train, new_y_train)

            test_idxs = np.where(target_labels == target_label)[0]
            loss[test_idxs] = _get_loss(loss_fn, new_model, objective,
                                        X=X_test[test_idxs], y=y_test[test_idxs])  # shape=(X_test.shape[0],)

    # removal estimation
    else:
        new_X_train = np.delete(X_train, train_idx, axis=0)
        new_y_train = np.delete(y_train, train_idx)
        new_model = clone(model).fit(new_X_train, new_y_train)
        loss = _get_loss(loss_fn, new_model, objective, X=X_test, y=y_test)  # shape=(X_test.shape[0],)

    influence = loss - original_loss

    return influence


def _get_loss(loss_fn, model, objective, X, y, batch=False):
    """
    Return
        - 1d array of individual losses of shape=(X.shape[0],),
            unless batch=True, then return a single float.

    Note
        - Parallelizable method.
    """
    if objective == 'regression':
        y_pred = model.predict(X)  # shape=(X.shape[0])

    elif objective == 'binary':
        y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities, shape=(X.shape[0],)

    else:
        assert objective == 'multiclass'
        y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

    result = loss_fn(y, y_pred, raw=False, batch=batch)  # shape(X.shape[0],) or single float

    return result
