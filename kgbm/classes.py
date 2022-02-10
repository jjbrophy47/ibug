import time
import joblib

import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.stats import norm
from scipy.stats import sem

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
                 k_params=[3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301,
                           401, 501, 601, 701], scoring='nll', min_scale=1e-15,
                 eps=1e-15, random_state=1, verbose=0, logger=None):
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

        # use portion of trees to sample
        if self.tree_frac < 1.0:
            n_idxs = int(self.n_boost_ * self.tree_frac)
            # tree_idxs = np.arange(self.n_boost_)[:n_idxs]  # first trees
            tree_idxs = self.rng_.choice(self.n_boost_, size=n_idxs, replace=False)  # random trees
            # tree_idxs = np.arange(self.n_boost_)[-n_idxs:]  # last trees
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

    def pred_dist(self, X, return_kneighbors=False, return_affinity_stats=False):
        """
        Extract distribution for each x in X.

        Input
            X: 2d array of data.
            return_kneighbors: bool, If True, return neighbor indices and values
                for each x in X.

        Return
            - If return_neighbors=True, return 2 2d arrays of shape=(len(X), self.k_)
                + Neighbor indices and values.
            - If return_affinity_stats=True, return a dict containing:
                + 'cnt_ens': Instance counts at leaves, averaged over test instances.
                + 'unq_cnt_ens': Unique instance counts at leaves, averaged over test instances.
                + 'cnt_tree': Instance counts at leaves, averaged over trees then test instances.
                + 'unq_cnt_tree': Unique instance counts at leaves, averaged over trees then test instances.
            - Otherwise, return 2 1d arrays of shape=(len(X),)
                + Location and shape.

        Note
            - k-nearest neighbors computed using affinity to
                the test example.
        """
        if return_kneighbors:
            assert not return_affinity_stats
        elif return_affinity_stats:
            assert not return_kneighbors

        start = time.time()

        leaf_dict = self.leaf_dict_  # schema: tree ID -> leaf ID -> train IDs
        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)

        # intermediate tracker
        affinity = np.zeros(self.n_train_, dtype=np.float32)  # shape=(n_train,)        

        # result objects
        if return_kneighbors:
            neighbor_idxs = np.zeros((len(X), self.k_), dtype=np.int32)
            neighbor_vals = np.zeros((len(X), self.k_), dtype=np.float32)
        elif return_affinity_stats:
            instances = np.zeros((len(X), 2), dtype=np.int32)  # shape=(n_test, 2)
        else:
            loc = np.zeros(len(X), dtype=np.float32)
            scale = np.zeros(len(X), dtype=np.float32)

        for i in range(test_leaves.shape[0]):  # per test example
            affinity[:] = 0  # reset affinity

            # compute affinity
            for tree_idx in self.tree_idxs_:  # per tree
                affinity[leaf_dict[tree_idx][test_leaves[i, tree_idx]]] += 1

            # get k nearest neighbors
            train_idxs = np.argsort(affinity)[-self.k_:]
            train_vals = self.y_train_[train_idxs]

            # add to result
            if return_kneighbors:
                neighbor_idxs[i, :] = train_idxs
                neighbor_vals[i, :] = train_vals
            elif return_affinity_stats:
                instances[i, 0] = np.sum(affinity)
                instances[i, 1] = len(np.where(affinity > 0)[0])
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
        elif return_affinity_stats:
            instances_mean = np.mean(instances, axis=0)  # shape=(2,)
            instances_sem = sem(instances, axis=0)  # shape=(2,)
            avg_instances = instances / len(self.tree_idxs_)  # shape=(len(X), 2)
            avg_instances_mean = np.mean(avg_instances, axis=0)  # shape=(2,)
            avg_instances_sem = sem(avg_instances, axis=0)  # shape=(2,)
            result = {'cnt_ens': {'mean': instances_mean[0], 'sem': instances_sem[0]},
                      'cnt_ens_unq': {'mean': instances_mean[1], 'sem': instances_sem[1]},
                      'cnt_tree': {'mean': avg_instances_mean[0], 'sem': avg_instances_sem[0]},
                      'cnt_tree_unq': {'mean': avg_instances_mean[1], 'sem': avg_instances_sem[1]}}
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

            # get min. scale
            candidate_idxs = np.where(self.scale_val_ > self.eps)[0]
            if len(candidate_idxs) > 0:
                self.min_scale_ = np.min(self.scale_val_[candidate_idxs])
            else:
                warn_msg = f'[KNNWrapper - WARNING] All validation predictions had 0 variance, '
                warn_msg += f'setting rho (min. variance) to {self.eps}...'
                warn_msg += f' this may lead to poor results.'
                if self.logger:
                    self.logger(warn_msg)
                else:
                    print(warn_msg)
                self.min_scale_ = self.eps

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


class KNNWrapper(Estimator):
    """
    K-Nearest neigbors regressor with uncertainty estimation.
        Wrapper around a KNN regressor model enabling probabilistic forecasting
        by modeling the output distribution of the k-nearest neighbors to a
        given test example.
    """
    def __init__(self, k=100, tree_frac=1.0, loc_type='knn',
                 k_params=[3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301,
                           401, 501, 601, 701], scoring='nll', min_scale=1e-15,
                 eps=1e-15, random_state=1, verbose=0, logger=None):
        """
        Input
            k: int, no. neighbors to consider for uncertainty estimation.
            tree_frac: float, Fraction of trees to use for the affinity computation
            loc_type: str, prediction should come from original tree or mean of neighbors.
            k_params: list, k values to try during tuning.
            scoring: str, metric to score probabilistic forecasts.
            rho: float, Minimum scale value.
            eps: float, Addendum to scale value.
            random_state: int, Seed for random number generation to enhance reproducibility.
            verbose: int, verbosity level.
            logger: object, If not None, output to logger.
        """
        assert k > 0
        assert loc_type == 'knn'
        assert isinstance(k_params, list) and len(k_params) > 0
        assert scoring in ['nll', 'crps']
        assert min_scale > 0
        assert eps > 0
        assert isinstance(random_state, int) and random_state > 0

        self.k = k
        self.loc_type = loc_type
        self.k_params = k_params
        self.scoring = scoring
        self.min_scale = min_scale
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger

    def get_params(self):
        """
        Return a dict of parameter values.
        """
        d = {}
        d['k'] = self.k
        d['loc_type'] = self.loc_type
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
            model: KNN regressor.
            X: training data.
            y: training targets.
        """
        X, y = util.check_data(X, y, objective='regression')
        self.model_ = model

        # pseudo-random number generator
        self.rng_ = np.random.default_rng(self.random_state)

        # save results
        self.n_train_ = X.shape[0]
        self.y_train_ = y.copy()

        # tune k
        k_params = [k for k in self.k_params if k <= len(X)]

        if X_val is not None and y_val is not None:
            X_val, y_val = util.check_data(X_val, y_val, objective='regression')
            assert len(X) == len(y)
            assert len(X_val) == len(y_val)
            assert X.shape[1] == X_val.shape[1]
            loc_val, scale_val, k, min_scale = self._tune_k(X_train=X, y_train=y, X_val=X_val, y_val=y_val,
                                                            k_params=k_params, scoring=self.scoring)
            self.loc_val_ = loc_val
            self.scale_val_ = scale_val
            self.k_ = k
            self.min_scale_ = min_scale
        else:
            self.k_ = self.k
            self.min_scale_ = self.min_scale

        best_params = {'n_neighbors': self.k_}
        self.uncertainty_estimator = clone(model).set_params(**best_params).fit(X, y)

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
        Return mean and variance for each x in X.

        Input
            X: 2d array of data.

        Return
            - Location and shape, 2 1d arrays of shape=(len(X),).
                + Location and shape
        """
        start = time.time()

        # result objects
        loc = self.predict(X)  # shape=(len(X))
        scale = np.zeros(len(X), dtype=np.float32)  # shape=(len(X))

        neighbors = self.uncertainty_estimator.kneighbors(X, return_distance=False)  # shape=(len(X), self.k_)
        for i, train_idxs in enumerate(neighbors):  # per test example
            train_vals = self.y_train_[train_idxs]

            # add to result
            scale[i] = np.std(train_vals)
            if scale[i] < self.min_scale_:  # handle extremely small scale values
                scale[i] = self.min_scale_

            # display progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                cum_time = time.time() - start
                if self.logger:
                    self.logger.info(f'[KNN - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')
                else:
                    print(f'[KNN - predict]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')

        # assemble output         
        result = loc, scale

        return result

    # private
    def _tune_k(self, X_train, y_train, X_val, y_val, k_params=[3, 5], scoring='nll'):
        """
        Tune k-nearest neighbors for probabilistic prediction.

        Input
            X_train: 2d array of train data.
            y_train: 1d array of train targets.
            X_val: 2d array of validation data.
            y_val: 1d array of validation targets.
            k_params: list, values of k to evaluate.
            scoring: str, evaluation metric.

        Return
            k that has the best validation scores.
                min_scale associated with k is also returned.
        """
        start = time.time()

        loc_vals = np.expand_dims(self.predict(X_val), axis=1)
        loc = np.tile(loc_vals, len(k_params)).astype(np.float32)  # shape=(len(X), len(k_params))
        scale = np.zeros((len(X_val), len(k_params)), dtype=np.float32)  # shape=(len(X), len(k_params))

        for i in range(len(X_val)):  # per test example
            affinity = np.linalg.norm(X_val[i] - X_train, axis=1)  # shape=(len(X_train),)
            train_idxs = np.argsort(affinity)  # smallest to largest distance

            # evaluate different k
            for j, k in enumerate(k_params):
                train_vals_k = y_train[train_idxs[:k]]
                scale[i, j] = np.std(train_vals_k) + self.eps

            # progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                if self.logger:
                    self.logger.info(f'[KNN - tuning] {i + 1:,} / {len(X_val):,}, '
                                     f'cum. time: {time.time() - start:.3f}s')
                else:
                    print(f'[KNN - tuning] {i + 2:,} / {len(X_val):,}, '
                          f'cum. time: {time.time() - start:.3f}s')

        # evaluate
        results = []
        if scoring == 'nll':
            for j, k in enumerate(k_params):
                nll = np.mean([-norm.logpdf(y_val[i], loc=loc[i, j], scale=scale[i, j]) for i in range(len(y_val))])
                results.append({'k': k, 'score': nll, 'k_idx': j})

            df = pd.DataFrame(results).sort_values('score', ascending=True)
            best_k = df.astype(object).iloc[0]['k']
            best_k_idx = df.astype(object).iloc[0]['k_idx']

            if self.verbose > 0:
                if self.logger:
                    self.logger.info(f'\n[KNN - tuning] k results:\n{df}')
                else:
                    print(f'\n[KNN - tuning] k results:\n{df}')
        else:
            raise ValueError(f'Unknown scoring {scoring}')

        loc_val = loc[:, best_k_idx]
        scale_val = scale[:, best_k_idx]

        # get min. scale
        candidate_idxs = np.where(scale_val > self.eps)[0]
        if len(candidate_idxs) > 0:
            min_scale = np.min(scale_val[candidate_idxs])
        else:
            warn_msg = f'[KNNWrapper - WARNING] All validation predictions had 0 variance, '
            warn_msg += f'setting rho (min. variance) to {self.eps}...'
            warn_msg += f' this may lead to poor results.'
            if self.logger:
                self.logger.info(warn_msg)
            else:
                print(warn_msg)
            min_scale = self.eps

        return loc_val, scale_val, best_k, min_scale
