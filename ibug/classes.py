import time
import joblib

import numpy as np
import pandas as pd
import properscoring as ps
from sklearn.base import clone
from scipy.stats import norm
from scipy.sparse import csr_matrix

from .base import Estimator
from .parsers import util
from .parsers.tree import Tree
from .parsers.tree import TreeEnsemble


class IBUGWrapper(Estimator):
    """
    K-Nearest Neigbors Gradient Boosting Machine.
        Wrapper around any GBM model enabling probabilistic forecasting
        by modeling the output distribution of the k-nearest neighbors to a
        given test example in the learnt tree-kernel space.
    """
    def __init__(self,
                 k=100,
                 gamma=1.0,
                 delta=0.0,
                 variance_calibration=True,
                 tree_subsample_frac=1.0,
                 tree_subsample_order='random',
                 instance_subsample_frac=1.0,
                 affinity='unweighted',
                 min_scale=1e-15,
                 eps=1e-15,
                 scoring='nll',
                 n_jobs=1,
                 random_state=1,
                 verbose=0,
                 logger=None):
        """
        Input
            k: int, no. neighbors to consider for uncertainty estimation.
            gamma: float, scale value applied to all variance estimates. '1.0' means no transformation.
            delta: float, value added to all variance estimates. '0.0' means no transformation.
            variance_calibration: bool, If True, tune variance valibration parameters gamma and delta.
                Note: Only tunes when X_val and y_val are supplied in the 'fit' method.
            tree_subsample_frac: float, Fraction of trees to use for the affinity computation.
            tree_subsample_order: str, Order to sample trees in; {'random', 'ascending', 'descending'}.
            instance_subsample_frac: float, Fraction of instances to subsample for the affinity computation.
            affinity: str, If 'weighted', weight affinity by leaf weights.
            min_scale: float, Minimum scale value.
            eps: float, Addendum to scale value.
            scoring: str, metric to score probabilistic forecasts.
            n_jobs: int, number of jobs to run in parallel. -1 means use all available cores.
            random_state: int, Seed for random number generation to enhance reproducibility.
            verbose: int, verbosity level.
            logger: object, If not None, output to logger.
        """
        assert k > 0
        assert gamma > 0.0
        assert delta >= 0.0
        assert variance_calibration in [True, False]
        assert tree_subsample_frac > 0 and tree_subsample_frac <= 1.0
        assert tree_subsample_order in ['random', 'ascending', 'descending']
        assert instance_subsample_frac > 0 and tree_subsample_frac <= 1.0
        assert affinity in ['unweighted', 'weighted']
        assert scoring in ['nll', 'crps']
        assert min_scale > 0
        assert eps > 0
        assert n_jobs >= -1
        assert isinstance(random_state, int) and random_state > 0
        assert isinstance(verbose, int) and verbose >= 0

        self.k = k
        self.gamma = gamma
        self.delta = delta
        self.variance_calibration = variance_calibration
        self.tree_subsample_frac = tree_subsample_frac
        self.tree_subsample_order = tree_subsample_order
        self.instance_subsample_frac = instance_subsample_frac
        self.affinity = affinity
        self.scoring = scoring
        self.min_scale = min_scale
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger

        if affinity == 'weighted':
            raise ValueError("'unweighted' affinity not yet implemented!")

        if instance_subsample_frac < 1.0:
            raise ValueError("Instance subsampling not yet implemented!")

    def __getstate__(self):
        """
        Return object state as dict.
        """
        state = self.__dict__.copy()
        del state['model_']
        state['model_dict_'] = self.model_.__getstate__()
        return state

    def __setstate__(self, state_dict):
        """
        Set object state.
        """
        model_dict = state_dict['model_dict_']
        del state_dict['model_dict_']

        self.__dict__ = state_dict.copy()

        # rebuild trees
        trees = np.zeros(shape=model_dict['trees_arr_dict'].shape, dtype=np.object)
        for i in range(trees.shape[0]):
            trees[i, 0] = Tree(**model_dict['trees_arr_dict'][i, 0])

        # rebuild ensemble
        model_dict['trees'] = trees
        del model_dict['trees_arr_dict']
        self.model_ = TreeEnsemble(**model_dict)

        return self

    def get_params(self):
        """
        Return a dict of parameter values.
        """
        d = {}
        d['k'] = self.k
        d['gamma'] = self.gamma
        d['delta'] = self.delta
        d['variance_calibration'] = self.variance_calibration
        d['tree_subsample_frac'] = self.tree_subsample_frac
        d['tree_subsample_order'] = self.tree_subsample_order
        d['instance_subsample_frac'] = self.instance_subsample_frac
        d['affinity'] = self.affinity
        d['min_scale'] = self.min_scale
        d['eps'] = self.eps
        d['scoring'] = self.scoring
        d['n_jobs'] = self.n_jobs
        d['random_state'] = self.random_state
        d['verbose'] = self.verbose
        if hasattr(self, 'base_model_params_'):
            d['base_model_params_'] = self.base_model_params_
        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, model, X, y, X_val=None, y_val=None):
        """
        - Compute leaf IDs for each x in X.

        Input
            model: tree ensemble.
            X: 2d array of training data.
            y: 1d array of training targets.
            X_val: 2d array of validation data.
            y_val: 1d array of validation targets.
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
        self.base_model_params_ = model.get_params()
        if self.affinity == 'weighted':
            self.model_.update_node_count(X)
            self.leaf_weights_ = self.model_.get_leaf_weights(scale=1.0)

        # build leaf matrix
        self.leaf_mat_, self.leaf_counts_, self.leaf_cum_sum_ = self._build_leaf_matrix(X)

        # select which trees to sample
        self.set_tree_subsampling(frac=self.tree_subsample_frac, order=self.tree_subsample_order)

        # select which instances to sample
        self.set_instance_subsampling(frac=self.instance_subsample_frac)

        # tune parameters
        if X_val is not None and y_val is not None:
            X_val, y_val = util.check_data(X_val, y_val, objective=self.model_.objective)
            self.k_, self.min_scale_, self.loc_val_, self.scale_val_ = self._tune_k(X=X_val, y=y_val,
                                                                                    scoring=self.scoring)

            if self.variance_calibration:
                self.gamma_, self.delta_ = self._tune_calibration(loc=self.loc_val_, scale=self.scale_val_,
                                                                  y=y_val, scoring=self.scoring)
            else:
                self.gamma_, self.delta_ = self.gamma, self.delta
        else:
            self.k_ = self.k
            self.min_scale_ = self.min_scale
            self.delta_ = self.delta
            self.gamma_ = self.gamma

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
            return_kneighbors: bool, If True, also return neighbor indices and values
                for each x in X.

        Return
            - Location and shape: 2 1d arrays of shape=(len(X),)
            - If return_neighbors=True, return 2 additional 2d arrays of shape=(len(X), self.k_)
                + Neighbor indices and values.

        Note
            - k-nearest neighbors computed using affinity to
                the test example.
        """
        start = time.time()

        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)
        test_leaves[:, 1:] += self.leaf_cum_sum_[:-1]  # shape=(n_test, n_boost)

        # result objects
        loc = self.predict(X).squeeze()  # shape=(len(X),)
        scale = np.zeros(len(X), dtype=np.float32)  # shape=(len(X),)
        if return_kneighbors:
            neighbor_idxs = np.zeros((len(X), self.k_), dtype=np.int32)  # shape=(len(X), k)
            neighbor_vals = np.zeros((len(X), self.k_), dtype=np.float32)  # shape=(len(X), k)

        for i, leaf_idxs in enumerate(test_leaves):  # per test example
            leaf_idxs = leaf_idxs[self.tree_idxs_]  # subsample trees
            affinity = self.leaf_mat_[leaf_idxs].sum(axis=0)  # shape=(n_train,)
            train_idxs = np.asarray(affinity.argsort())[0][-self.k_:]  # k neighbors
            train_vals = self.y_train_[train_idxs]

            # add to result
            scale[i] = np.std(train_vals)
            if scale[i] <= self.min_scale_:  # handle extremely small scale values
                scale[i] = self.min_scale_
            if return_kneighbors:
                neighbor_idxs[i, :] = train_idxs
                neighbor_vals[i, :] = train_vals

            # display progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                cum_time = time.time() - start
                if self.logger:
                    self.logger.info(f'[IBUG - predict]: {i + 1:,} / {len(X):,}, '
                                     f'cum. time: {cum_time:.3f}s')
                else:
                    print(f'[IBUG - predict]: {i + 1:,} / {len(X):,}, '
                          f'cum. time: {cum_time:.3f}s')

        # variance valibration
        if self.variance_calibration:
            scale = scale * self.gamma_ + self.delta_

        # assemble output
        result = (loc, scale)
        if return_kneighbors:
            result += (neighbor_idxs, neighbor_vals)

        return result

    def set_tree_subsampling(self, frac, order):
        """
        Select which trees to sample.

        Input
            frac: float, Fraction of trees to subsample.
            order: str, Order of trees to subsample.
        """
        assert frac > 0.0 and frac <= 1.0
        assert order in ['random', 'ascending', 'descending']

        # modify instance parameters
        self.tree_subsample_frac = frac
        self.tree_subsample_order = order

        if frac < 1.0:
            n_idxs = max(1, int(self.n_boost_ * self.tree_subsample_frac))

            if self.tree_subsample_order == 'ascending':
                self.tree_idxs_ = np.arange(self.n_boost_)[:n_idxs]  # first to last

            elif self.tree_subsample_order == 'random':
                self.tree_idxs_ = self.rng_.choice(self.n_boost_, size=n_idxs, replace=False)  # random trees

            else:
                assert self.tree_subsample_order == 'descending'
                self.tree_idxs_ = np.arange(self.n_boost_)[::-1][:n_idxs]  # last to first
        else:
            self.tree_idxs_ = np.arange(self.n_boost_)

    def set_instance_subsampling(self, frac):
        """
        Select which instances to include in the affinity computation.

        Input
            frac: float, Fraction of the TOTAL no. train instances to sample.
        """
        assert frac > 0.0 and frac <= 1.0

        # modify instance parameters
        self.instance_subsample_frac = frac

        if self.instance_subsample_frac < 1.0:
            n_sample = int(self.n_train_ * self.instance_subsample_frac)
            self.sample_idxs_ = self.rng_.choice(self.n_train_, size=n_sample, replace=False)
        else:
            self.sample_idxs_ = np.arange(self.n_train_)

    def get_affinity_stats(self, X):
        """
        Compute average affinity statistics over all x in X.

        Input
            X: 2d array of data.

        Return
            Dict containing:
                * 'mean': Instance counts at leaves, averaged over test, shape=(n_boost,).

        NOTE
            Return array values are FRACTIONS of the total number of training examples.
        """
        start = time.time()

        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)
        test_leaves[:, 1:] += self.leaf_cum_sum_[:-1]  # shape=(n_test, n_boost)

        # result objects
        instances = np.zeros((len(X), len(self.tree_idxs_)), dtype=np.float32)  # shape=(n_test, n_boost)

        for i, leaf_idxs in enumerate(test_leaves):  # per test example
            leaf_idxs = leaf_idxs[self.tree_idxs_]  # subsample trees
            instances[i] = np.asarray(self.leaf_mat_[leaf_idxs].sum(axis=1)).flatten() / self.n_train_  # (n_boost,)

            # display progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                cum_time = time.time() - start
                if self.logger:
                    self.logger.info(f'[IBUG - affinity]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')
                else:
                    print(f'[IBUG - affinity]: {i + 1:,} / {len(X):,}, cum. time: {cum_time:.3f}s')

        # assemble output
        result = {'mean': np.mean(instances, axis=0)}
        return result

    def get_leaf_stats(self):
        """
        Compute statistics about leaf density.

        Return
            Dict containing:
                * 'max': 1d array of largest leaf densities (% of train), one per tree; shape=(n_boost,).
                * 'min': 1d array of smallest leaf densities (% of train), one per tree; shape=(n_boost,).

        NOTE
            Return array values are FRACTIONS of the total number of training examples.
        """
        assert self.n_boost_ == len(self.leaf_counts_)

        # result
        max_arr = np.zeros(self.n_boost_, dtype=np.float32)  # shape=(n_boost,)
        min_arr = np.zeros(self.n_boost_, dtype=np.float32)  # shape=(n_boost,)

        leaf_densities = np.asarray(self.leaf_mat_.sum(axis=1)).flatten()  # shape=(total no. leaves,)

        n_prev_leaves = 0
        for i, leaf_count in enumerate(self.leaf_counts_):  # per tree
            leaf_arr = leaf_densities[n_prev_leaves: n_prev_leaves + leaf_count]
            max_arr[i] = np.max(leaf_arr) / self.n_train_
            min_arr[i] = np.min(leaf_arr) / self.n_train_
            n_prev_leaves += leaf_count

        # assemble output
        result = {'max': max_arr[self.tree_idxs_], 'min': min_arr[self.tree_idxs_]}
        return result

    # private
    def _build_leaf_matrix(self, X):
        """
        Build sparse leaf matrix of shape=(total no. leaves, len(X)).
            Example:
                row = [1 7 11 | 0 5 11]  # `train_leaves` (after adjustment) flattened
                col = [0 0 0  | 1 1 1 ]  # column indices

        Input
            X: 2d array of training data.

        Return
            - Sparse Leaf matrix of shape=(total_n_leaves, len(X)).
            - 1d array of leaf counts of shape=(n_boost,).
            - 1d array of cumulative leaf counts of shape=(n_boost,).
        """
        leaf_counts = self.model_.get_leaf_counts().squeeze()  # shape=(n_boost,)
        leaf_cum_sum = np.cumsum(leaf_counts)
        total_num_leaves = np.sum(leaf_counts)

        train_leaves = self.model_.apply(X).squeeze()  # shape=(len(X), n_boost)
        train_leaves[:, 1:] += leaf_cum_sum[:-1]  # shape=(len(X), n_boost)

        row = train_leaves.flatten().astype(np.int32)  # shape=(n_boost * len(X))
        col = np.concatenate([[i] * self.n_boost_ for i in range(len(X))]).astype(np.int32)  # (len(X) * n_boost)
        data = np.ones(self.n_boost_ * len(X), dtype=np.float32)  # shape=(n_boost * len(X),)
        leaf_mat = csr_matrix((data, (row, col)), shape=(total_num_leaves, len(X)), dtype=np.float32)

        return leaf_mat, leaf_counts, leaf_cum_sum

    def _tune_k(self, X, y, scoring='nll',
                k_params=[3, 5, 7, 9, 11, 15, 31, 61, 91, 121,
                          151, 201, 301, 401, 501, 601, 701]):
        """
        Tune k-nearest neighbors for probabilistic prediction.

        Input
            X_train: 2d array of data.
            y: 1d array of targets.
            scoring: str, evaluation metric.
            k_params: list, values of k to evaluate.

        Return
            Best k.
        """
        start = time.time()
        k_params = [k for k in k_params if k <= self.n_train_]  # remove k larger than n_train

        test_leaves = self.model_.apply(X).squeeze()  # shape=(n_test, n_boost)
        test_leaves[:, 1:] += self.leaf_cum_sum_[:-1]  # shape=(len(X), n_boost)

        scale = np.zeros((len(X), len(k_params)), dtype=np.float32)
        loc = np.tile(self.model_.predict(X), len(k_params)).astype(np.float32)  # shape=(len(X), n_k)

        # gather predictions
        vals = self.y_train_

        for i, leaf_idxs in enumerate(test_leaves):  # per test example
            leaf_idxs = leaf_idxs[self.tree_idxs_]  # subsample trees
            affinity = self.leaf_mat_[leaf_idxs].sum(axis=0)  # shape=(n_train,)
            train_idxs = np.asarray(affinity.argsort())[0]  # neighbors

            for j, k in enumerate(k_params):
                vals_knn = vals[train_idxs[-k:]]
                scale[i, j] = np.std(vals_knn) + self.eps

            # progress
            if (i + 1) % 100 == 0 and self.verbose > 0:
                if self.logger:
                    self.logger.info(f'[IBUG - tuning] {i + 1:,} / {len(X):,}, '
                                     f'cum. time: {time.time() - start:.3f}s')
                else:
                    print(f'[IBUG - tuning] {i + 2:,} / {len(X):,}, '
                          f'cum. time: {time.time() - start:.3f}s')

        # evaluate
        assert scoring in ['nll', 'crps']

        results = []
        for j, k in enumerate(k_params):
            score = self._eval_normal(y=y, loc=loc[:, j], scale=scale[:, j], scoring=scoring)
            results.append({'k': k, 'score': score, 'k_idx': j})

        df = pd.DataFrame(results).sort_values('score', ascending=True)
        best_k = df.astype(object).iloc[0]['k']
        best_k_idx = df.astype(object).iloc[0]['k_idx']

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
                self.logger(warn_msg)
            else:
                print(warn_msg)
            min_scale = self.eps

        if self.verbose > 0:
            if self.logger:
                self.logger.info(f'\n[IBUG - tuning] k results:\n{df}')
            else:
                print(f'\n[IBUG - tuning] k results:\n{df}')

        return best_k, min_scale, loc_val, scale_val

    def _tune_calibration(self, loc, scale, y,
                          base_vals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                                     1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
                          multipliers=[1.0, 2.5, 5.0],
                          ops=['add', 'mult'], scoring='nll'):
        """
        Add or multiply detla to scale values.

        Input
            loc: 1d array of location values.
            scale: 1d array of scale values.
            y: 1d array of target values (same shape as scale).
            base_vals: list, List of base candidate delta values.
            multipliers: list, List of values to multiply the base values by.
            ops: list, List of operations to perform to the scale array.
            scoring: str, Evaluation metric.

        Return
            - gamma, float, value to scale each variance estimate by.
            - delta, float, value to add to each variance estimate.
        """
        assert scoring in ['nll', 'crps']
        assert ops == ['add', 'mult']
        assert loc.shape == scale.shape == y.shape

        results = []
        for op in ops:
            for base_val in base_vals:
                for multiplier in multipliers:

                    if op == 'mult' and base_val == 0.0:
                        continue

                    if op == 'add':
                        temp_scale = scale + (base_val * multiplier)
                    else:
                        temp_scale = scale * (base_val * multiplier)

                    score = self._eval_normal(y=y, loc=loc, scale=temp_scale, scoring=scoring)
                    results.append({'base_val': base_val, 'op': op, 'multiplier': multiplier, 'score': score})

        df = pd.DataFrame(results).sort_values('score', ascending=True)

        best_val = df.iloc[0]['base_val'] * df.iloc[0]['multiplier']
        best_op = df.iloc[0]['op']

        if best_op == 'add':
            gamma = 1.0
            delta = best_val
        else:
            assert best_op == 'mult'
            gamma = best_val
            delta = 0.0

        if self.verbose > 0:
            if self.logger:
                self.logger.info(f'\ndelta gridsearch:\n{df}')
            else:
                print(f'\ndelta gridsearch:\n{df}')

        return gamma, delta

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

    def _eval_normal(self, y, loc, scale, scoring):
        """
        Evaluate pedictions assuming the output follows a normal distribution.

        Input
            y: 1d array of targets
            loc: 1d array of mean values (same length as y).
            scale: 1d array of std. dev. values (same length as y).
            scoring: str, evaluation metric.

        Return
            - Float, Average score over all examples.
        """
        if scoring == 'nll':
            result = eval_normal(y=y, loc=loc, scale=scale, nll=True, crps=False)

        elif scoring == 'crps':
            result = eval_normal(y=y, loc=loc, scale=scale, nll=False, crps=True)

        else:
            raise ValueError(f'Unknown scoring {scoring}')

        return result


class KNNWrapper(Estimator):
    """
    K-Nearest neigbors regressor with uncertainty estimation.
        Wrapper around a KNN regressor model enabling probabilistic forecasting
        by modeling the output distribution of the k-nearest neighbors to a
        given test example.
    """
    def __init__(self,
                 k=100,
                 gamma=1.0,
                 delta=0.0,
                 variance_calibration=True,
                 scoring='nll',
                 min_scale=1e-15,
                 eps=1e-15,
                 random_state=1,
                 verbose=0,
                 logger=None):
        """
        Input
            k: int, no. neighbors to consider for uncertainty estimation.
            gamma: float, scale value applied to all variance estimates. '1.0' means no transformation.
            delta: float, value added to all variance estimates. '0.0' means no transformation.
            variance_calibration: bool, If True, tune variance valibration parameters gamma and delta.
                Note: Only tunes when X_val and y_val are supplied in the 'fit' method.
            tree_frac: float, Fraction of trees to use for the affinity computation
            loc_type: str, prediction should come from original tree or mean of neighbors.
            scoring: str, metric to score probabilistic forecasts.
            rho: float, Minimum scale value.
            eps: float, Addendum to scale value.
            random_state: int, Seed for random number generation to enhance reproducibility.
            verbose: int, verbosity level.
            logger: object, If not None, output to logger.
        """
        assert k > 0
        assert gamma > 0.0
        assert delta >= 0.0
        assert variance_calibration in [True, False]
        assert scoring in ['nll', 'crps']
        assert min_scale > 0
        assert eps > 0
        assert isinstance(random_state, int) and random_state > 0

        self.k = k
        self.gamma = gamma
        self.delta = delta
        self.variance_calibration = variance_calibration
        self.min_scale = min_scale
        self.eps = eps
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger

    def get_params(self):
        """
        Return a dict of parameter values.
        """
        d = {}
        d['k'] = self.k
        d['gamma'] = self.gamma
        d['delta'] = self.delta
        d['variance_calibration'] = self.variance_calibration
        d['min_scale'] = self.min_scale
        d['eps'] = self.eps
        d['scoring'] = self.scoring
        d['random_state'] = self.random_state
        d['verbose'] = self.verbose
        if hasattr(self, 'base_model_params_'):
            d['base_model_params_'] = self.model_.get_params()
        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, model, X, y, X_val=None, y_val=None):
        """
        - Compute leaf IDs for each x in X.

        Input
            model: KNN regressor.
            X: 2d array of training data.
            y: 1d array of training targets.
            X_val: 2d array of validation data.
            y_val: 1d array of validation targets.
        """
        X, y = util.check_data(X, y, objective='regression')
        self.model_ = model
        self.base_model_params_ = self.model_.get_params()

        # pseudo-random number generator
        self.rng_ = np.random.default_rng(self.random_state)

        # save results
        self.n_train_ = X.shape[0]
        self.y_train_ = y.copy()

        if X_val is not None and y_val is not None:
            X_val, y_val = util.check_data(X_val, y_val, objective='regression')
            assert len(X) == len(y)
            assert len(X_val) == len(y_val)
            assert X.shape[1] == X_val.shape[1]
            self.k_, self.min_scale_, self.loc_val_, self.scale_val_ = self._tune_k(X_train=X, y_train=y,
                                                                                    X_val=X_val, y_val=y_val,
                                                                                    scoring=self.scoring)

            if self.variance_calibration:
                self.gamma_, self.delta_ = self._tune_calibration(loc=self.loc_val_, scale=self.scale_val_,
                                                                  y=y_val, scoring=self.scoring)
            else:
                self.gamma_, self.delta_ = self.gamma, self.delta

        else:
            self.k_ = self.k
            self.min_scale_ = self.min_scale
            self.gamma_ = self.gamma
            self.delta_ = self.delta

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

        # variance calibration
        if self.variance_calibration:
            scale = scale * self.gamma_ + self.delta_

        # assemble output         
        result = loc, scale

        return result

    # private
    def _tune_k(self, X_train, y_train, X_val, y_val, scoring='nll',
                k_params=[3, 5, 7, 9, 11, 15, 31, 61, 91, 121,
                          151, 201, 301, 401, 501, 601, 701]):
        """
        Tune k-nearest neighbors for probabilistic prediction.

        Input
            X_train: 2d array of train data.
            y_train: 1d array of train targets.
            X_val: 2d array of validation data.
            y_val: 1d array of validation targets.
            scoring: str, evaluation metric.
            k_params: list, values of k to evaluate.

        Return
            - k that has the best validation scores.
            - min_scale associated with k is also returned.
            - location values.
            - scale values.
        """
        start = time.time()
        k_params = [k for k in k_params if k <= len(X_train)]

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
        assert scoring in ['nll', 'crps']

        results = []
        for j, k in enumerate(k_params):
            score = self._eval_normal(y=y_val, loc=loc[:, j], scale=scale[:, j], scoring=scoring)
            results.append({'k': k, 'score': score, 'k_idx': j})

        df = pd.DataFrame(results).sort_values('score', ascending=True)
        best_k = df.astype(object).iloc[0]['k']
        best_k_idx = df.astype(object).iloc[0]['k_idx']

        if self.verbose > 0:
            if self.logger:
                self.logger.info(f'\n[KNN - tuning] k results:\n{df}')
            else:
                print(f'\n[KNN - tuning] k results:\n{df}')

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

        return best_k, min_scale, loc_val, scale_val

    def _tune_calibration(self, loc, scale, y,
                          base_vals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                                     1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3],
                          multipliers=[1.0, 2.5, 5.0],
                          ops=['add', 'mult'], scoring='nll'):
        """
        Add or multiply detla to scale values.

        Input
            loc: 1d array of location values.
            scale: 1d array of scale values.
            y: 1d array of target values (same shape as scale).
            base_vals: list, List of base candidate delta values.
            multipliers: list, List of values to multiply the base values by.
            ops: list, List of operations to perform to the scale array.
            scoring: str, Evaluation metric.

        Return
            - gamma, float, value to scale each variance estimate by.
            - delta, float, value to add to each variance estimate.
        """
        assert scoring in ['nll', 'crps']
        assert ops == ['add', 'mult']
        assert loc.shape == scale.shape == y.shape

        results = []
        for op in ops:
            for base_val in base_vals:
                for multiplier in multipliers:

                    if op == 'mult' and base_val == 0.0:
                        continue

                    if op == 'add':
                        temp_scale = scale + (base_val * multiplier)
                    else:
                        temp_scale = scale * (base_val * multiplier)

                    score = self._eval_normal(y=y, loc=loc, scale=temp_scale, scoring=scoring)
                    results.append({'base_val': base_val, 'op': op, 'multiplier': multiplier, 'score': score})

        df = pd.DataFrame(results).sort_values('score', ascending=True)

        best_val = df.iloc[0]['base_val'] * df.iloc[0]['multiplier']
        best_op = df.iloc[0]['op']

        if best_op == 'add':
            gamma = 1.0
            delta = best_val
        else:
            assert best_op == 'mult'
            gamma = best_val
            delta = 0.0

        if self.verbose > 0:
            if self.logger:
                self.logger.info(f'\ndelta gridsearch:\n{df}')
            else:
                print(f'\ndelta gridsearch:\n{df}')

        return gamma, delta

    def _eval_normal(self, y, loc, scale, scoring):
        """
        Evaluate pedictions assuming the output follows a normal distribution.

        Input
            y: 1d array of targets
            loc: 1d array of mean values (same length as y).
            scale: 1d array of std. dev. values (same length as y).
            scoring: str, evaluation metric.

        Return
            - Float, Average score over all examples.
        """
        if scoring == 'nll':
            result = eval_normal(y=y, loc=loc, scale=scale, nll=True, crps=False)

        elif scoring == 'crps':
            result = eval_normal(y=y, loc=loc, scale=scale, nll=False, crps=True)

        else:
            raise ValueError(f'Unknown scoring {scoring}')

        return result


# utility methods
def eval_normal(y, loc, scale, nll=True, crps=False):
    """
    Evaluate each predicted normal distribution.

    Input
        y: 1d array of targets
        loc: 1d array of mean values (same length as y).
        scale: 1d array of std. dev. values (same length as y).
        nll: bool, If True, return the avg. neg. log likelihood.
        crps: bool, If True, return the avg. CRPS score.

    Return
        Tuple of scores.
    """
    assert nll or crps
    assert y.shape == loc.shape == scale.shape

    result = ()
    if nll:
        result += (np.mean([-norm.logpdf(y[i], loc=loc[i], scale=scale[i]) for i in range(len(y))]),)
    if crps:
        result += (np.mean([ps.crps_gaussian(y[i], mu=loc[i], sig=scale[i]) for i in range(len(y))]),)

    if len(result) == 1:
        result = result[0]

    return result
