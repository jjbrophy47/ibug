import time
import joblib

import numpy as np
from sklearn.base import clone

from .base import Explainer
from .parsers import util


class LOO(Explainer):
    """
    Leave-one-out influence explainer. Retrains the model
    for each train example to get change in loss.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y_t, f_{w/o x_i}(x_t)) - L(y_t, f(x_t))
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).

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

    def fit(self, model, X, y):
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

    def get_local_influence(self, X, y, verbose=1):
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
        return self._run_loo(X_test=X, y_test=y, inf='local')

    # private
    def _run_loo(self, X_test=None, y_test=None, batch=False, inf='global'):
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
            self.logger.info('\n[INFO] computing LOO values...')
            self.logger.info(f'[INFO] no. cpus: {n_jobs:,}...')

        # fit each model in parallel
        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            # result container
            if inf == 'local':
                original_loss = _get_loss(loss_fn, original_model, objective, X=X_test, y=y_test)  # (X_test.shape[0],)
                influence = np.zeros((0, X_test.shape[0]), dtype=util.dtype_t)

            elif inf == 'global' and X_test is not None and batch:  # global expected influence
                original_loss = _get_loss(loss_fn, original_model, objective, X=X_test, y=y_test, batch=batch)  # float
                influence = np.zeros((0, 1), dtype=util.dtype_t)

            else:
                assert inf == 'global' and not batch
                original_loss = _get_loss(loss_fn, original_model, objective, X=X_train, y=y_train)  # (no. train,)
                influence = np.zeros((0, 1), dtype=util.dtype_t)

            # trackers
            fits_completed = 0
            fits_remaining = X_train.shape[0]

            # get number of fits to perform for this iteration
            while fits_remaining > 0:
                n = min(100, fits_remaining)

                results = parallel(joblib.delayed(_run_iteration)
                                                 (original_model, X_train, y_train, train_idx, X_test, y_test,
                                                  loss_fn, objective, original_loss,
                                                  batch, inf) for train_idx in range(fits_completed,
                                                                                     fits_completed + n))

                # synchronization barrier
                results = np.vstack(results)  # shape=(n, 1 or X_test.shape[0])
                influence = np.vstack([influence, results])

                fits_completed += n
                fits_remaining -= n

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO] fits: {fits_completed:,} / {X_train.shape[0]:,}'
                                     f', cum. time: {cum_time:.3f}s')

        return influence


def _run_iteration(model, X_train, y_train, train_idx, X_test, y_test,
                   loss_fn, objective, original_loss, batch, inf):
    """
    Fit model after leaving out the specified `train_idx` train example.

    Return
        - 1d array of shape=(X_test.shape[0],) or single float.

    Note
        - Parallelizable method.
    """
    new_X = np.delete(X_train, train_idx, axis=0)
    new_y = np.delete(y_train, train_idx)
    new_model = clone(model).fit(new_X, new_y)

    if inf == 'local':
        loss = _get_loss(loss_fn, new_model, objective, X=X_test, y=y_test)  # shape=(X_test.shape[0],)
        influence = loss - original_loss

    elif inf == 'global' and X_test is not None and batch:
        loss = _get_loss(loss_fn, new_model, objective, X=X_test, y=y_test, batch=True)  # single float
        influence = np.array([loss - original_loss])

    else:
        assert inf == 'global' and not batch

        X_temp = X_train[[train_idx]]
        y_temp = y_train[[train_idx]]

        loss = _get_loss(loss_fn, new_model, objective, X=X_temp, y=y_temp)  # shape=(1,)
        influence = loss - original_loss[train_idx]

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
