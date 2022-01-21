import time
import joblib

import numpy as np
import pandas as pd
import seaborn as sns  # TEMP
import matplotlib.pyplot as plt  # TEMP
from scipy.integrate import quad  # TEMP
from scipy.stats import gaussian_kde  # TEMP
from scipy.stats import norm  # TEMP
from sklearn.neighbors import KernelDensity  # TEMP
from sklearn.preprocessing import LabelBinarizer

from .base import Estimator
from .parsers import util


class KGBM(Estimator):
    """
    K-Nearest Neigbors Gradient Boosting Machine.
        Wrapper around any standard GBM model enabling probabilistic forecasting
        using the k-nearest neighbors to a given test example in latent space.
    """
    def __init__(self, k=100, tree_frac=1.0, loc_type='gbm', affinity='unweighted',
                 k_params=[3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701], scoring='nll',
                 min_scale=1e-15, eps=1e-15, random_state=1, verbose=0, logger=None):
        """
        Input
            k: int, no. neighbors to consider for uncertainty estimation.
            tree_frac: float, Fraction of trees to use for the affinity computation
            loc_type: str, prediction should come from original tree or mean of neighbors.
            tree_frac: str, prediction should come from original tree or mean of neighbors.
            affinity: str, If 'weighted', weight affinity by leaf weights.
            k_params: list, k values to try during tuning.
            scoring: str, metric to score probabilistic forecasts.
            min_scale: float, Minimum scale value.
            eps: float, Addendum to scale value.
            random_state: int, Seed for random number generation to enhance reproducibility.
            verbose: int, verbosity level.
            logger: object, If not None, output to logger.
        """
        assert k > 0
        assert tree_frac > 0 and tree_frac <= 1.0
        assert loc_type in ['gbm', 'knn']
        assert affinity in ['unweighted', 'weighted']
        assert isinstance(k_params, list) and len(k_params) > 0
        assert scoring in ['nll', 'crps']
        assert min_scale > 0
        assert eps > 0
        assert isinstance(random_state, int) and random_state > 0

        self.k = k
        self.tree_frac = tree_frac
        self.loc_type = loc_type
        self.affinity = affinity
        self.k_params = k_params
        self.scoring = scoring
        self.min_scale = min_scale
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger

        if affinity == 'weighted':
            raise ValueError("'unweighted' affinity not yet implemented!")

    def fit(self, model, X, y, X_val=None, y_val=None):
        """
        - Compute leaf IDs for each x in X.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        # pseudo-random number generator
        self.rng_ = np.random.default_rng(self.random_state)

        # save results
        self.n_train_ = X.shape[0]
        self.y_train_ = y.copy()
        self.n_boost_ = self.model_.n_boost_
        self.n_class_ = self.model_.n_class_
        self.scale_bias_ = np.std(y)

        self.model_.update_node_count(X)

        self.leaf_counts_ = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        """
        organize training example leaf IDs
        schema: tree ID -> leaf ID -> train IDs
        {
          0: {0: [1, 7, 9], 1: [2, 3, 5]},
          1: {0: [2, 11], 1: [1, 7, 27]}
        }
        """
        train_leaves = self.model_.apply(X).squeeze()  # shape=(len(X), no. boost)
        leaf_counts = self.model_.get_leaf_counts().squeeze()  # shape=(n_boost,)

        leaf_dict = {}
        for boost_idx in range(train_leaves.shape[1]):
            leaf_dict[boost_idx] = {}

            for leaf_idx in range(leaf_counts[boost_idx]):
                leaf_dict[boost_idx][leaf_idx] = np.where(train_leaves[:, boost_idx] == leaf_idx)[0]

        self.leaf_dict_ = leaf_dict

        # randomly select a subset of trees to sample
        if self.tree_frac < 1.0:
            n_idxs = int(self.n_boost_ * self.tree_frac)
            tree_idxs = self.rng_.choice(self.n_boost_, size=n_idxs, replace=False)
        else:
            tree_idxs = np.arange(self.n_boost_)
        self.tree_idxs_ = tree_idxs

        # tune k
        k_params = [k for k in self.k_params if k <= len(X)]

        if X_val is not None and y_val is not None:
            X_val, y_val = util.check_data(X_val, y_val, objective=self.model_.objective)
            self.k_ = self._tune_k(X=X_val, y=y_val, k_params=k_params, scoring=self.scoring)
        else:
            self.k_ = self.k
            self.min_scale_ = self.min_scale

        return self

    def predict(self, X):
        """
        Predict using the parsed model.

        Input
            X: 2d array of data.

        Return
            2d array of shape=(len(X), no. class).
        """
        return self.model_.predict(X)

    def pred_dist(self, X):
        """
        Fit distribution for each x in X.

        Input
            X: 2d array of data.

        Return 2 1d arrays of shape=(len(X),).

        Note
            - k-nearest neighbors computed using affinity to
                the test example.
        """
        start = time.time()

        leaf_dict = self.leaf_dict_  # schema: tree ID -> leaf ID -> train IDs
        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)

        loc = np.zeros(len(X), dtype=np.float32)
        scale = np.zeros(len(X), dtype=np.float32)

        vals = self.y_train_

        affinity = np.zeros(self.n_train_, dtype=np.float32)  # shape=(n_train,)

        for i in range(test_leaves.shape[0]):  # per test example
            affinity[:] = 0  # reset affinity

            # get nearest neighbors
            for tree_idx in self.tree_idxs_:  # per tree
                affinity[leaf_dict[tree_idx][test_leaves[i, tree_idx]]] += 1
            train_vals = vals[np.argsort(affinity)]

            loc[i] = np.mean(train_vals[-self.k:])
            scale[i] = np.std(train_vals[-self.k:])

            # handle extremely small scale values
            if scale[i] <= self.min_scale_:
                scale[i] = self.min_scale_

            # display progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                cum_time = time.time() - start
                if self.logger:
                    self.logger.info(f'[KGBM - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')
                else:
                    print(f'[KGBM - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')

        if self.loc_type == 'gbm':
            loc[:] = self.predict(X).squeeze()  # shape=(len(X),)

        return loc, scale

    def _tune_k(self, X, y, k_params=[3, 5], scoring='nll'):
        """
        Tune k-nearest neighbors for probabilistic prediction.

        Input
            X_train: 2d array of data.
            y: 1d array of targets.
            X_val: 2d array of validation data.
            y_val: 1d array of validation targets.
            k_params: list, values of k to evaluate.
            cv: int, no. cross-validation folds.
            scoring: str, evaluation metric.
        """
        start = time.time()

        leaf_dict = self.leaf_dict_  # schema: tree ID -> leaf ID -> train IDs
        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost, n_class)

        scale = np.zeros((len(X), len(k_params)), dtype=np.float32)
        if self.loc_type == 'gbm':        
            loc = np.tile(self.model_.predict(X), len(k_params)).astype(np.float32)  # shape=(len(X), n_k)
        else:
            loc = np.zeros((len(X), len(k_params)), dtype=np.float32)

        # gather predictions
        vals = self.y_train_
        affinity = np.zeros(self.n_train_, dtype=np.float32)  # shape=(n_train,)

        for i in range(test_leaves.shape[0]):  # per test example
            affinity[:] = 0  # reset affinity

            # get nearest neighbors
            for tree_idx in self.tree_idxs_:  # per tree
                affinity[leaf_dict[tree_idx][test_leaves[i, tree_idx]]] += 1
            train_idxs = np.argsort(affinity)

            for j, k in enumerate(k_params):
                vals_knn = vals[train_idxs[-k:]]
                scale[i, j] = np.std(vals_knn) + self.eps
                if self.loc_type != 'gbm':
                    loc[i, j] = np.mean(vals_knn)

            # progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                if self.logger:
                    self.logger.info(f'[KGBM - tuning] {i + 1:,} / {len(X):,}, cum. time: {time.time() - start:.3f}s')
                else:
                    print(f'[KGBM - tuning] {i + 2:,} / {len(X):,}, cum. time: {time.time() - start:.3f}s')

        # evaluate
        results = []
        if scoring == 'nll':
            for j, k in enumerate(k_params):
                nll = np.mean([-norm.logpdf(y[i], loc=loc[i, j], scale=scale[i, j]) for i in range(len(y))])
                results.append({'k': k, 'score': nll, 'k_idx': j})

            df = pd.DataFrame(results).sort_values('score', ascending=True)
            best_k = df.astype(object).iloc[0]['k']
            best_k_idx = df.astype(object).iloc[0]['k_idx']

            self.loc_val_ = loc[:, best_k_idx]
            self.scale_val_ = scale[:, best_k_idx]
            self.min_scale_ = np.min(self.scale_val_)

            if self.verbose > 0:
                if self.logger:
                    self.logger.info(f'\n[KGBM - tuning] k results:\n{df}')
                else:
                    print(f'\n[KGBM - tuning] k results:\n{df}')
        else:
            raise ValueError(f'Unknown scoring {scoring}')

        return best_k

    def get_params(self):
        """
        Return a dict of parameter values.
        """
        d = {}
        d['k'] = self.k
        d['tree_frac'] = self.tree_frac
        d['loc_type'] = self.loc_type
        d['affinity'] = self.affinity
        d['k_params'] = self.k_params
        d['scoring'] = self.scoring
        d['min_scale'] = self.min_scale
        d['eps'] = self.eps
        if hasattr(self, 'k_'):
            d['k_'] = self.k_
        if hasattr(self, 'min_scale_'):
            d['min_scale_'] = self.min_scale_
        return d


class FlexPGBM(Estimator):
    """
    Flexible Probabilistic Gradient Boosting Machine.
        - Inspired by PGBM, this extension allows ANY distribution to be
            modeled using the k-nearest neighbors to the test example in tree space.
    """
    def __init__(self, k=100, loc_type='gbm', cv=None, logger=None):
        """
        Input
            k: int, no. neighbors to consider for uncertainty estimation.
            loc_type: str, prediction should come from original tree or mean of neighbors.
            cv: int, If not None, no. cross-validation folds to use to tune k.
            logger: object, If not None, output to logger.
        """
        if cv is None:
            assert k > 0
        else:
            assert cv > 0
        assert loc_type in ['gbm', 'knn']
        self.k = k
        self.loc_type = loc_type
        self.cv = cv
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Compute leaf IDs for each x in X.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        # save results
        self.n_train_ = X.shape[0]
        self.y_train_ = y.copy()
        self.n_boost_ = self.model_.n_boost_
        self.n_class_ = self.model_.n_class_

        self.model_.update_node_count(X)

        self.leaf_counts_ = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)
        self.leaf_weights_ = self.model_.get_leaf_weights()  # shape=(total no. leaves,)
        self.train_leaves_ = self.model_.apply(X)  # shape=(len(X), no. boost, no. class)
        self.train_weights_ = self._get_leaf_weights(self.train_leaves_)  # shape=(len(X), n_boost, n_class)

        # # TEMP: Plotting distribution of leaf values
        # train_leaves = np.squeeze(self.train_leaves_)[:, -1]  # shape=(n_train,)
        # print(f'no. leaves: {np.max(train_leaves):,}')
        # for leaf_idx in range(np.max(train_leaves)):
        #     train_idxs = np.where(train_leaves == leaf_idx)[0]
        #     print(len(train_idxs))
        #     sns.histplot(y[train_idxs], kde=True)
        #     plt.show()
        #     if leaf_idx > 50:
        #         exit(0)

        return self

    def predict(self, X):
        """
        Predict using the parsed model.

        Input
            X: 2d array of data.

        Return
            2d array of shape=(len(X), no. class).
        """
        return self.model_.predict(X)

    # def pred_dist_experiment(self, X):
    #     """
    #     Fit distribution for each x in X.

    #     Input
    #         X: 2d array of data.

    #     Return
    #     """
    #     k = self.k
    #     loc_type = self.loc_type

    #     train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
    #     test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

    #     test_weights = np.squeeze(1 / self._get_leaf_weights(test_leaves))  # shapse=(n_test, n_boost, n_class)
    #     avg_test_weights = np.mean(test_weights, axis=1)  # shape=(n_test,)

    #     sns.histplot(avg_test_weights)
    #     plt.show()

    #     loc = np.zeros(len(X), dtype=np.float32)
    #     scale = np.zeros(len(X), dtype=np.float32)

    #     avg_affinity_list = []

    #     for i in range(len(test_leaves)):
    #         vals = self.y_train_

    #         # get nearest neighbors
    #         co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
    #         affinity = np.sum(co_leaf, axis=(1, 2))  # shape=(n_train,)
    #         affinity = affinity / (self.n_boost_ * self.n_class_)  # normalize, shape=(n_train,)
    #         train_idxs = np.argsort(affinity)[::-1][:k]

    #         loc[i] = np.mean(vals[train_idxs])
    #         scale[i] = np.std(vals[train_idxs])

    #         avg_affinity_list.append(np.mean(affinity))

    #         if i < 10:
    #             sns.histplot(affinity)
    #             plt.show()

    #     if self.loc_type == 'og':
    #         loc[:] = self.model_.predict(X).flatten()

    #     sns.histplot(avg_affinity_list)
    #     plt.show()

    #     return loc, scale

    def pred_dist_save2(self, X):
        """
        Fit distribution for each x in X.

        Input
            X: 2d array of data.

        Return 2 1d arrays of shape=(len(X),).

        Note
            - k-nearest neighbors computed using affinity to
                the test example.
        """
        k = self.k
        loc_type = self.loc_type

        train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
        test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

        loc = np.zeros(len(X), dtype=np.float32)
        scale = np.zeros(len(X), dtype=np.float32)

        for i in range(len(test_leaves)):
            vals = self.y_train_

            # get nearest neighbors
            co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
            affinity = np.sum(co_leaf, axis=(1, 2))  # shape=(n_train,)
            affinity = affinity / (self.n_boost_ * self.n_class_)  # normalize, shape=(n_train,)
            train_idxs = np.argsort(affinity)[::-1][:k]

            loc[i] = np.mean(vals[train_idxs])
            scale[i] = np.std(vals[train_idxs])

        if self.loc_type == 'og':
            loc[:] = self.model_.predict(X).flatten()

        return loc, scale

    def pred_dist_kde(self, X):
        """
        Fit distribution for each x in X.

        Input
            X: 2d array of data.

        Return
        """
        train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
        test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

        kde_list = []
        for i in range(len(test_leaves)):
            vals = self.y_train_

            # get nearest neighbors (potentially weighted)
            co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
            affinity = np.sum(co_leaf, axis=(1, 2))  # shape=(n_train,)
            affinity = affinity / (self.n_boost_ * self.n_class_)  # shape=(n_train,)
            train_idxs = np.argsort(affinity)[::-1][:self.k]
            kde_list.append(gaussian_kde(vals[train_idxs], bw_method='scott'))

        return kde_list

    # def pred_dist5(self, X, tau=0.75, m=100, kde=False):
    #     """
    #     Fit distribution for each x in X.

    #     Input
    #         X: 2d array of data.

    #     Return
    #     """
    #     train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
    #     test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

    #     kde = np.zeros(len(X), dtype=np.object)

    #     for i in range(len(test_leaves)):
    #         vals = self.y_train_

    #         # get nearest neighbors (potentially weighted)
    #         co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
    #         affinity = np.sum(co_leaf, axis=(1, 2))  # shape=(n_train,)
    #         affinity = affinity / (self.n_boost_ * self.n_class_)  # shape=(n_train,)

    #         train_idxs = np.where(affinity >= tau)[0]

    #         if len(train_idxs) == 0:
    #             train_idxs = np.argsort(affinity)[::-1][:m]

    #         kde[i] = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(vals[train_idxs].reshape(-1, 1))

    #     return kde

    # def pred_dist4(self, X, tau=0.75, m=100):
    #     """
    #     Fit distribution for each x in X.

    #     Input
    #         X: 2d array of data.

    #     Return
    #     """
    #     train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
    #     test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

    #     train_weights = self.train_weights_  # shape=(n_train, n_boost, n_class)

    #     # result containers
        # loc = np.zeros(len(X), dtype=np.float32)
        # scale = np.zeros(len(X), dtype=np.float32)

    #     # compute weighted location (mean) and scale (std. dev.)
    #     for i in range(len(X)):
    #         vals = self.y_train_

    #         co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
    #         weights = np.sum(co_leaf / (train_leaves.shape[1] * train_leaves.shape[2]), axis=(1, 2))

    #         train_idxs = np.where(weights >= tau)[0]

    #         if len(train_idxs) == 0:
    #             train_idxs = np.argsort(weights)[::-1][:m]

    #         weighted_mean = np.average(vals[train_idxs], weights=weights[train_idxs])
    #         weighted_var = np.average((vals[train_idxs] - weighted_mean) ** 2, weights=weights[train_idxs])

    #         loc[i] = weighted_mean
    #         scale[i] = np.sqrt(weighted_var)

    #         if i % 100 == 0:
    #             print(f'finished {i:,} / {len(X):,}')

    #     return loc, scale

    def pred_dist_save(self, X):
        """
        Fit distribution for each x in X.

        Input
            X: 2d array of data.

        Return 2 1d arrays of shape=(len(X),).

        Note
            - k-nearest neighbors computed using weighted affinity to
                the test example.
        """
        train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
        test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

        train_weights = self.train_weights_  # shape=(n_train, n_boost, n_class)

        # result containers
        loc = np.zeros(len(X), dtype=np.float32)
        scale = np.zeros(len(X), dtype=np.float32)

        # compute weighted location (mean) and scale (std. dev.)
        for i in range(len(X)):
            vals = self.y_train_

            co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
            affinity = np.sum(train_weights * co_leaf, axis=(1, 2))  # shape=(n_train,)
            train_idxs = np.argsort(affinity)[::-1][:self.k]

            loc[i] = np.mean(vals[train_idxs])
            scale[i] = np.std(vals[train_idxs])

        if self.loc_type == 'og':
            loc[:] = self.model_.predict(X).flatten()

        return loc, scale

    # def pred_dist2(self, X, tau=0.75, m=100):
    #     """
    #     Fit distribution for each x in X.

    #     Input
    #         X: 2d array of data.

    #     Return
    #     """
    #     train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
    #     test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

    #     train_weights = self.train_weights_  # shape=(n_train, n_boost, n_class)

    #     # result containers
    #     loc = np.zeros(len(X), dtype=np.float32)
    #     scale = np.zeros(len(X), dtype=np.float32)

    #     # compute weighted location (mean) and scale (std. dev.)
    #     for i in range(len(X)):
    #         vals = self.y_train_

    #         co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
    #         weights = np.sum(train_weights * co_leaf, axis=(1, 2))  # shape=(n_train,)

    #         weighted_mean = np.average(vals, weights=weights)
    #         weighted_var = np.average((vals - weighted_mean) ** 2, weights=weights) 

    #         loc[i] = weighted_mean
    #         scale[i] = np.sqrt(weighted_var)

    #         if i % 100 == 0:
    #             print(f'finished {i:,} / {len(X):,}')

    #     return loc, scale

    # def pred_dist1(self, X):
    #     """
    #     Fit distribution for each x in X.

    #     Input
    #         X: 2d array of data.

    #     Return
    #     """
    #     train_leaves = self.train_leaves_  # shape=(n_train, n_boost, n_class)
    #     test_leaves = self.model_.apply(X)  # shape=(n_test, n_boost, n_class)

    #     n = 0
    #     for i in range(n, n + len(test_leaves)):

    #         # get nearest neighbors (potentially weighted)
    #         co_leaf = np.where(train_leaves == test_leaves[[i]], 1, 0)  # shape=(n_train, n_boost, n_class)
    #         affinity = np.sum(co_leaf, axis=(1, 2))  # shape=(n_train,)
    #         weights = affinity / train_leaves.shape[1]

    #         train_idxs = np.where(weights > 0.65)[0]

    #         kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.y_train_[train_idxs].reshape(-1, 1))

    #         pdf = lambda x: np.exp(kde.score_samples([[x]]))[0]
    #         mean = quad(lambda x: x * pdf(x), a=-np.inf, b=np.inf)[0]
    #         variance = quad(lambda x: (x ** 2) * pdf(x), a=-np.inf, b=np.inf)[0] - mean ** 2

    #         std = np.sqrt(variance)

    #         # build distribution
    #         fig, ax = plt.subplots()
    #         sns.kdeplot(x=self.y_train_[train_idxs], weights=weights[train_idxs], ax=ax)
    #         ax.set_title(f'{X[i]}: {len(train_idxs):,} ({mean:.3f} +/- {std:.3f})')
    #         plt.show()

    def _get_leaf_weights(self, leaf_idxs):
        """
        Retrieve leaf weights given the leaf indices.

        Input
            leaf_idxs: Leaf indices, shape=(no. examples, no. boost, no. class)

        Return
            - 3d array of shape=(no. examples, no. boost, no. class)
        """
        leaf_counts = self.leaf_counts_  # shape=(no. boost, no. class)
        leaf_weights = self.leaf_weights_  # shape=(no. leaves across all trees,)

        # result container
        weights = np.zeros(leaf_idxs.shape, dtype=util.dtype_t)  # shape=(no. examples, no. boost, no. class)

        n_prev_leaves = 0

        for b_idx in range(self.n_boost_):

            for c_idx in range(self.n_class_):
                leaf_count = leaf_counts[b_idx, c_idx]

                weights[:, b_idx, c_idx] = leaf_weights[n_prev_leaves:][leaf_idxs[:, b_idx, c_idx]]

                n_prev_leaves += leaf_count

        return weights


class GBRUT(Estimator):
    """
    Gradient-Boosted Regression Uncertainty Trees
        - Tree-ensemble model that provides 95% confidence intervals with
            point-estimate predictions.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y, F_{w/o x_i}(x_t)) - L(y, F(x_t))
        - Pos. value means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
        - Neg. value means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Note
        - Does NOT take class or instance weight into account.

    Note
        - Only supports GBRTs.
    """
    def __init__(self, atol=1e-5, logger=None):
        """
        Input
            atol: float, Tolerance between actual and predicted leaf values.
            logger: object, If not None, output to logger.
        """
        self.atol = atol
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Compute leaf values using Newton leaf estimation method;
            make sure these match existing leaf values. Put into a 1d array.

        - Copy leaf values and compute new 1d array of leaf values across all trees,
            one new array resulting from removing each training example x in X.

        - Compute 95% confidence intervals for each leaf value, should be the same
            shape as the leaf values.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        # extract tree-ensemble metadata
        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        learning_rate = self.model_.learning_rate
        l2_leaf_reg = self.model_.l2_leaf_reg
        bias = self.model_.bias

        # make sure model is a GBRT using a squared error loss function and no L2 leaf regularization
        assert self.model_.tree_type != 'rf', 'RF not supported for GBRUT!'
        assert self.model_.objective == 'regression', 'Objective not regression!'
        assert 'SquaredLoss' in str(self.loss_fn_), 'Loss function not squared error'
        # assert l2_leaf_reg == 0, 'l2_leaf_reg is not 0!'  # TEMPORARY

        # get no. leaves for each tree
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        # intermediate containers
        current_approx = np.tile(bias, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        n_prev_leaves = 0

        # result containers
        leaf_values = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        leaf_cis = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        leaf_weights = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        n_not_close = 0
        max_diff = 0

        # save gradient information of leaf values for each tree
        for boost_idx in range(n_boost):
            # print(f'\n{boost_idx}')

            doc_preds = np.zeros((X.shape[0], n_class), dtype=util.dtype_t)

            # precompute gradient statistics
            gradient = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)

            # get leaf values
            leaf_count = leaf_counts[boost_idx, 0]
            leaf_vals = trees[boost_idx, 0].get_leaf_values()  # 1d array
            doc2leaf = trees[boost_idx, 0].apply(X)  # shape=(X.shape[0],)

            # update predictions for this class
            doc_preds[:, 0] = leaf_vals[doc2leaf]

            # sanity check to make sure leaf values are correctly computed
            # also need to save some statistics to update leaf values later
            for leaf_idx in range(leaf_count):
                doc_ids = np.where(doc2leaf == leaf_idx)[0]

                # # compute leaf value and 95% CI of leaf value
                # doc_grads = gradient[doc_ids, 0]
                # leaf_prediction = -np.mean(doc_grads) * learning_rate
                # leaf_ci = 1.96 * np.std(doc_grads) / np.sqrt(len(doc_ids)) * learning_rate

                # compute leaf value and 95% CI of leaf value: TEMP
                doc_grads = gradient[doc_ids, 0]
                # leaf_prediction = -np.mean(doc_grads) * learning_rate
                leaf_prediction = -np.sum(doc_grads) / (len(doc_grads) + l2_leaf_reg) * learning_rate
                # leaf_ci = 1.96 * np.std(doc_grads) / np.sqrt(len(doc_ids)) * learning_rate
                leaf_ci = np.std(doc_grads) * learning_rate  # TEMP: S.D.

                # compare predicted leaf value to actual leaf value
                if not np.isclose(leaf_prediction, leaf_vals[leaf_idx], atol=1e-5):
                    n_not_close += 1
                    max_diff = max(max_diff, abs(leaf_prediction - leaf_vals[leaf_idx]))

                # store statistics
                leaf_values[n_prev_leaves + leaf_idx] = leaf_prediction
                leaf_cis[n_prev_leaves + leaf_idx] = leaf_ci
                leaf_weights[n_prev_leaves + leaf_idx] = len(doc_ids)

            n_prev_leaves += leaf_count  # move to next set of tree leaves
            current_approx += doc_preds  # update approximation

        # copy and compute new leaf values resulting from the removal of each x in X.
        start = time.time()
        if self.logger:
            self.logger.info(f'\n[INFO] no. leaf vals not within 1e-5 tol.: {n_not_close:,}, '
                             f'max. diff.: {max_diff:.5f}')
            self.logger.info(f'\n[INFO] computing alternate leaf values...')

        # check predicted leaf values do not differ too much from actual model
        if max_diff > self.atol:
            raise ValueError(f'{max_diff:.5f} (max. diff.) > {self.atol} (tolerance)')

        # save results of this method
        self.leaf_values_ = leaf_values  # shape=(total no. leaves,)
        self.leaf_cis_ = leaf_cis  # shape=(total no. leaves,)
        self.leaf_weights_ = leaf_weights  # shape=(total no. leaves,)
        self.leaf_counts_ = leaf_counts  # shape=(no. boost, no. class)
        self.bias_ = bias
        self.n_boost_ = n_boost
        self.n_class_ = n_class
        self.n_train_ = X.shape[0]

        return self

    def predict(self, X, include_ci=False, include_leaf_weight=False):
        """
        Estimate target value for each x in X with uncertainty.

        Input
            X: 2d array of data.
            include_ci: bool; if True, return array of 95%
                confidence intervals with predictions.

        Return
            If include_ci is True, 2 1d arrays of shape=(X.shape[0],),
                otherwise just 1 1d array of shape=(X.shape[0],).
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        pred = np.full(X.shape[0], self.bias_).astype(util.dtype_t)  # shape=(X.shape[0],)
        ci = np.full(X.shape[0], 0).astype(util.dtype_t)  # shape=(X.shape[0],)
        leaf_weight = np.full(X.shape[0], 0).astype(util.dtype_t)  # shape=(X.shape[0],)

        # get prediction and CI of each test example using leaf values and CIs of leaf values
        n_prev_leaves = 0
        for boost_idx in range(self.n_boost_):  # per boosting iteration

            for test_idx in range(X.shape[0]):  # per test example
                leaf_idx = doc2leaf[test_idx][boost_idx][0]

                pred[test_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                ci[test_idx] += self.leaf_cis_[n_prev_leaves + leaf_idx]
                leaf_weight[test_idx] += self.leaf_weights_[n_prev_leaves + leaf_idx]

            n_prev_leaves += self.leaf_counts_[boost_idx]

        result = (pred,)

        if include_ci:
            result += (ci,)

        if include_leaf_weight:
            result += (leaf_weight,)

        return result
