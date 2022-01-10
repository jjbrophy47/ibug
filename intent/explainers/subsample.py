import time
import joblib

import numpy as np
from sklearn.base import clone

from .base import Explainer
from .parsers import util


class SubSample(Explainer):
    """
    Explainer that approximates data Shapley values. Trains many models on different
    subsets of the data to obtain expected marginal influence values.

    Local-Influence Semantics (i.e. influence)
        - Inf.(x_i, x_t) := E[L(y_t, f_{w/o x_i}(x_t))] - E[L(y_t, f(x_t))]
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).

    Note
        - Supports both GBDTs and RFs.
        - Supports parallelization.
    """
    def __init__(self, sub_frac=0.7, n_iter=4000, n_jobs=1, random_state=1, logger=None):
        """
        Input
            sub_frac: float, Fraction of train data to use for training.
            n_iter: int, No. sub-models to train.
            n_jobs: int, No. processes to run in parallel.
                -1 means use the no. of available CPU cores.
            random_state: int, Seed for reproducibility.
            logger: object, If not None, output to logger.
        """
        self.sub_frac = sub_frac
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Setup.

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
        return self._run_subsample(X_test=X, y_test=y)

    # private
    def _run_subsample(self, X_test=None, y_test=None):
        """
        - Train multiple models on different training subsets
            and measure expected change in train/test loss.

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

        n_iter = self.n_iter
        sub_frac = self.sub_frac
        random_state = self.random_state

        start = time.time()
        if self.logger:
            self.logger.info('\n[INFO] computing influence values...')
            self.logger.info(f'[INFO] no. cpus: {n_jobs:,}...')

        # fit each model in parallel
        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            # result containers
            in_loss = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=util.dtype_t)
            out_loss = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=util.dtype_t)

            in_count = np.zeros(X_train.shape[0], dtype=np.int32)
            out_count = np.zeros(X_train.shape[0], dtype=np.int32)

            # trackers
            fits_completed = 0
            fits_remaining = n_iter

            # get number of fits to perform for this iteration
            while fits_remaining > 0:
                n = min(100, fits_remaining)

                results = parallel(joblib.delayed(_run_iteration)
                                                 (original_model, X_train, y_train, X_test, y_test,
                                                  loss_fn, objective, sub_frac,
                                                  random_state + i) for i in range(fits_completed,
                                                                                   fits_completed + n))

                # synchronization barrier
                for losses, in_idxs in results:
                    out_idxs = np.setdiff1d(np.arange(X_train.shape[0]), in_idxs)

                    for test_idx, loss in enumerate(losses):
                        in_loss[in_idxs, test_idx] += loss
                        out_loss[out_idxs, test_idx] += loss

                        in_count[in_idxs] += 1
                        out_count[out_idxs] += 1

                fits_completed += n
                fits_remaining -= n

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO] fits: {fits_completed:>7,} / {n_iter:,}'
                                     f', cum. time: {cum_time:.3f}s')

        # compute difference in expected losses
        influence = (out_loss / out_count.reshape(-1, 1)) - (in_loss / in_count.reshape(-1, 1))

        return influence


def _run_iteration(model, X_train, y_train, X_test, y_test, loss_fn, objective, sub_frac, seed):
    """
    Fit model after leaving out the specified `train_idx` train example.

    Return
        - 1d array of shape=(X_test.shape[0],) or single float.

    Note
        - Parallelizable method.
    """
    rng = np.random.default_rng(seed)

    start = time.time()
    idxs = rng.choice(X_train.shape[0], size=int(X_train.shape[0] * sub_frac), replace=False)
    new_X_train = X_train[idxs].copy()
    new_y_train = y_train[idxs].copy()

    new_model = clone(model).fit(new_X_train, new_y_train)
    loss = _get_loss(loss_fn, new_model, objective, X=X_test, y=y_test)  # shape=(X_test.shape[0],)

    return loss, idxs


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
