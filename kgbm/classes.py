import time
import joblib

import numpy as np
import pandas as pd
# import seaborn as sns  # TEMP
# import matplotlib.pyplot as plt  # TEMP
# from scipy.integrate import quad  # TEMP
# from scipy.stats import gaussian_kde  # TEMP
from scipy.stats import norm  # TEMP
# from sklearn.neighbors import KernelDensity  # TEMP
# from sklearn.preprocessing import LabelBinarizer

from .base import Estimator
from .parsers import util


class KGBMWrapper(Estimator):
    """
    K-Nearest Neigbors Gradient Boosting Machine.
        Wrapper around any GBM model enabling probabilistic forecasting
        by modeling the output distribution of the k-nearest neighbors to a
        given test example in the learnt tree-kernel space.
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

    def pred_dist(self, X, return_kneighbors=False):
        """
        Extract distribution for each x in X.

        Input
            X: 2d array of data.
            return_kneighbors: bool, If True, return neighbor indices and values
                for each x in X.

        Return
            - If return_neighbors=True, return 2 2d arrays of shape=(len(X), self.k_)
                + Neighbor indices and values.
            - If return_neighbors=False, return 2 1d arrays of shape=(len(X),)
                + Location and shape.

        Note
            - k-nearest neighbors computed using affinity to
                the test example.
        """
        start = time.time()

        leaf_dict = self.leaf_dict_  # schema: tree ID -> leaf ID -> train IDs
        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)

        # intermediate tracker
        affinity = np.zeros(self.n_train_, dtype=np.float32)  # shape=(n_train,)        

        # result objects
        if return_kneighbors:
            neighbor_idxs = np.zeros((len(X), self.k_), dtype=np.int32)
            neighbor_vals = np.zeros((len(X), self.k_), dtype=np.float32)
        else:
            loc = np.zeros(len(X), dtype=np.float32)
            scale = np.zeros(len(X), dtype=np.float32)

        for i in range(test_leaves.shape[0]):  # per test example
            affinity[:] = 0  # reset affinity

            # compute affinity
            for tree_idx in self.tree_idxs_:  # per tree
                affinity[leaf_dict[tree_idx][test_leaves[i, tree_idx]]] += 1

            # get k nearest neighbors
            train_idxs = np.argsort(affinity)[-self.k:]
            train_vals = self.y_train_[train_idxs]

            # add to result
            if return_kneighbors:
                neighbor_idxs[i, :] = train_idxs
                neighbor_vals[i, :] = train_vals
            else:
                loc[i] = np.mean(train_vals)
                scale[i] = np.std(train_vals)
                if scale[i] <= self.min_scale_:  # handle extremely small scale values
                    scale[i] = self.min_scale_

            # display progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                cum_time = time.time() - start
                if self.logger:
                    self.logger.info(f'[KGBM - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')
                else:
                    print(f'[KGBM - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')

        # assemble output
        if return_kneighbors:
            result = neighbor_idxs, neighbor_vals
        else:
            if self.loc_type == 'gbm':
                loc[:] = self.predict(X).squeeze()  # shape=(len(X),)            
            result = loc, scale

        return result

    # private
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
