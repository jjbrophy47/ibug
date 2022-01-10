import time
import joblib

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class LeafInfluenceLE(Explainer):
    """
    LeafInfluenceLE: Explainer that adapts the influence functions method to tree ensembles
        with label estimation.

    Local-Influence Semantics
        - Inf.(x_i, x_t) ~= L(y, F_{w/o x_i w/ y*}(x_t)) - L(y, F(x_t))
        - Pos. value means removing x_i and adding x_i w/ y* increases the loss (original x_i helpful).
        - Neg. value means removing x_i and adding x_i w/ y* decreases the loss (original x_i harmful).

    Note
        - For GBDT, influence values are multipled by -1; this makes the semantics of
            LeafInfluence values more consistent with the other influence methods
            that approximate changes in loss.
        - Does NOT take class or instance weight into account.

    Reference
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/influence/leaf_influence.py

    Paper
        - https://arxiv.org/abs/1802.06640

    Note
        - Only supports GBDTs.
    """
    def __init__(self, update_set=-1, atol=1e-5, n_jobs=1, logger=None):
        """
        Input
            update_set: int, No. neighboring leaf values to use for approximating leaf influence.
                0: Use no other leaves, influence is computed independent of other trees.
                -1: Use all other trees, most accurate but also most computationally expensive.
                1+: Trade-off between accuracy and computational resources.
            atol: float, Tolerance between actual and predicted leaf values.
            n_jobs: int, No. processes to run in parallel.
                -1 means use the no. of available CPU cores.
            logger: object, If not None, output to logger.
        """
        assert update_set >= -1
        self.update_set = update_set
        self.atol = atol
        self.n_jobs = n_jobs
        self.logger = logger

    def fit(self, model, X, y, target_labels=None):
        """
        - Compute leaf values using Newton leaf estimation method;
            make sure these match existing leaf values. Put into a 1d array.

        - Copy leaf values and compute new 1d array of leaf values across all trees,
            one new array resulting from removing each training example x in X.

        - Should end up with a 2d array of shape=(no. train, no. leaves across all trees).
            A bit memory intensive depending on the no. leaves, but should speed up the
            explanation for ANY set of test examples. This array can also be saved to
            disk to avoid recomputing these influence values.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
            target_labels: 1d array of target labels.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        assert self.model_.tree_type != 'rf', 'RF not supported for LeafInfluence'

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        # extract tree-ensemble metadata
        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        learning_rate = self.model_.learning_rate
        l2_leaf_reg = self.model_.l2_leaf_reg
        bias = self.model_.bias

        # get no. leaves for each tree
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        # intermediate containers
        current_approx = np.tile(bias, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        leaf2docs = []  # list of leaf_idx -> doc_ids dicts
        n_prev_leaves = 0

        # result containers
        da_vector_multiplier = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)
        denominator = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        leaf_values = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        n_not_close = 0
        max_diff = 0

        assert target_labels is not None
        unique_target_labels = np.unique(target_labels)

        naive_gradient_addendum_list = []
        gradient_list = []
        hessian_list = []

        for target_label in unique_target_labels:
            naive_gradient_addendum_list.append(np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t))

        # save gradient information of leaf values for each tree
        for boost_idx in range(n_boost):
            doc_preds = np.zeros((X.shape[0], n_class), dtype=util.dtype_t)

            # precompute gradient statistics
            gradient = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)
            hessian = self.loss_fn_.hessian(y, current_approx)  # shape=(X.shape[0], no. class)
            third = self.loss_fn_.third(y, current_approx)  # shape=(X.shape[0], no. class)

            # precompute statistics for label estimation
            for target_label in unique_target_labels:
                new_y = np.full(y.shape, target_label, dtype=util.dtype_t)  # shape=(X.shape[0], no. class)
                gradient_list.append(self.loss_fn_.gradient(new_y, current_approx))
                hessian_list.append(self.loss_fn_.hessian(new_y, current_approx))

            for class_idx in range(n_class):

                # get leaf values
                leaf_count = leaf_counts[boost_idx, class_idx]
                leaf_vals = trees[boost_idx, class_idx].get_leaf_values()
                doc2leaf = trees[boost_idx, class_idx].apply(X)
                leaf2doc = {}

                # update predictions for this class
                doc_preds[:, class_idx] = leaf_vals[doc2leaf]

                # sanity check to make sure leaf values are correctly computed
                # also need to save some statistics to update leaf values later
                for leaf_idx in range(leaf_count):
                    doc_ids = np.where(doc2leaf == leaf_idx)[0]
                    leaf2doc[leaf_idx] = set(doc_ids)

                    # compute leaf values using gradients and hessians
                    leaf_enumerator = np.sum(gradient[doc_ids, class_idx])
                    leaf_denominator = np.sum(hessian[doc_ids, class_idx]) + l2_leaf_reg
                    leaf_prediction = -leaf_enumerator / leaf_denominator * learning_rate

                    # compare leaf values to actual leaf values
                    if not np.isclose(leaf_prediction, leaf_vals[leaf_idx], atol=1e-5):
                        n_not_close += 1
                        max_diff = max(max_diff, abs(leaf_prediction - leaf_vals[leaf_idx]))

                    # store statistics
                    denominator[n_prev_leaves + leaf_idx] = leaf_denominator
                    leaf_values[n_prev_leaves + leaf_idx] = leaf_prediction

                n_prev_leaves += leaf_count  # move to next set of tree leaves
                leaf2docs.append(leaf2doc)  # list of dicts, one per tree

            # precompute influence statistics
            da_vector_multiplier[:, boost_idx, :] = doc_preds / learning_rate * third + hessian

            naive_gradient_a = hessian * doc_preds / learning_rate + gradient
            for i in range(len(unique_target_labels)):
                naive_gradient_b = hessian_list[i] * doc_preds / learning_rate + gradient_list[i]
                naive_gradient_addendum_list[i][:, boost_idx, :] = naive_gradient_a - naive_gradient_b

            current_approx += doc_preds  # update approximation

        # copy and compute new leaf values resulting from the removal of each x in X.
        start = time.time()
        if self.logger:
            self.logger.info(f'\n[INFO - LILE] no. leaf vals not within 1e-5 tol.: {n_not_close:,}, '
                             f'max. diff.: {max_diff:.5f}')
            self.logger.info(f'\n[INFO - LILE] computing alternate leaf values...')

        # check predicted leaf values do not differ too much from actual model
        if max_diff > self.atol:
            raise ValueError(f'{max_diff:.5f} (max. diff.) > {self.atol} (tolerance)')

        # select no. processes to run in parallel
        if self.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert self.n_jobs >= 1
            n_jobs = min(self.n_jobs, joblib.cpu_count())

        if self.logger:
            self.logger.info(f'[INFO - LILE] no. cpus: {n_jobs:,}...')

        leaf_derivatives_list = []
        for naive_gradient_addendum, target_label in zip(naive_gradient_addendum_list, unique_target_labels):

            if self.logger:
                self.logger.info(f'[INFO - LILE] target label: {target_label}')

            # process each training example removal in parallel
            with joblib.Parallel(n_jobs=n_jobs) as parallel:

                # result container
                leaf_derivatives = np.zeros((0, np.sum(leaf_counts)), dtype=util.dtype_t)

                # trackers
                n_completed = 0
                n_remaining = X.shape[0]

                # get number of fits to perform for this iteration
                while n_remaining > 0:
                    n = min(100, n_remaining)

                    results = parallel(joblib.delayed(_compute_leaf_derivatives)
                                                     (train_idx, leaf_counts, leaf2docs, denominator,
                                                      da_vector_multiplier, naive_gradient_addendum,
                                                      n_boost, n_class, X.shape[0], learning_rate,
                                                      self.update_set) for train_idx in range(n_completed,
                                                                                              n_completed + n))

                    # synchronization barrier
                    results = np.vstack(results)  # shape=(n, 1 or X_test.shape[0])
                    leaf_derivatives = np.vstack([leaf_derivatives, results])

                    n_completed += n
                    n_remaining -= n

                    if self.logger:
                        cum_time = time.time() - start
                        self.logger.info(f'[INFO - LILE] {n_completed:,} / {X.shape[0]:,}, cum. time: {cum_time:.3f}s')

            leaf_derivatives_list.append(leaf_derivatives)

        # save results of this method
        self.leaf_values_ = leaf_values  # shape=(total no. leaves,)
        self.leaf_derivatives_list_ = leaf_derivatives_list  # shape=(n_target_labels, n_train, total_n_leaves)
        self.unique_target_labels_ = unique_target_labels  # 1d
        self.leaf_counts_ = leaf_counts  # shape=(no. boost, no. class)
        self.bias_ = bias
        self.n_boost_ = n_boost
        self.n_class_ = n_class
        self.n_train_ = X.shape[0]

        return self

    def get_local_influence(self, X, y, target_labels=None):
        """
        - Compute influence of each training example on each test example loss.

        Return
            - 2d array of shape=(no. train, X.shape[0])
                * Train influences are in the same order as the original training order.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        influence = np.zeros((self.n_train_, X.shape[0], self.n_class_), dtype=util.dtype_t)

        if self.logger:
            self.logger.info('\n[INFO - LILE] computing influence for each test example...')

        assert target_labels is not None
        assert np.all(self.unique_target_labels_ == np.unique(target_labels))

        # compute influence of each training example on the test example
        for i, target_label in enumerate(np.unique(target_labels)):
            test_idxs = np.where(target_labels == target_label)[0]

            for remove_idx in range(self.n_train_):
                influence[remove_idx, test_idxs] = self._loss_derivative(X[test_idxs], y[test_idxs], remove_idx,
                                                                         self.leaf_derivatives_list_[i])

        # reshape result
        influence = influence.sum(axis=2)  # sum over class, shape=(no. train, X.shape[0])

        return influence

    # private
    def _loss_derivative(self, X, y, remove_idx, leaf_derivatives):
        """
        Compute the influence on the set of examples (X, y) using the updated
            set of leaf values from removing `remove_idx`.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.
            remove_idx: index of removed train instance.
            leaf_derivatives: 1d array of leaf derivatives, shape=(total_n_leaves,).

        Return
            - Array of test influences of shape=(X.shape[0], no. class).

        Note
            - We multiply the result by -1 to have consistent semantics
                with other influence methods that approx. loss.
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        og_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        new_pred = np.zeros((X.shape[0], self.n_class_), dtype=util.dtype_t)  # shape=(X.shape[0], no. class)

        # get prediction of each test example using the original and new leaf values
        tree_idx = 0
        n_prev_leaves = 0

        for boost_idx in range(self.n_boost_):  # per boosting iteration
            for class_idx in range(self.n_class_):  # per class

                for test_idx in range(X.shape[0]):  # per test example
                    leaf_idx = doc2leaf[test_idx][boost_idx][class_idx]
                    og_pred[test_idx, class_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                    new_pred[test_idx, class_idx] += leaf_derivatives[remove_idx][n_prev_leaves + leaf_idx]

                n_prev_leaves += self.leaf_counts_[boost_idx, class_idx]
            tree_idx += 1

        return -self.loss_fn_.gradient(y, og_pred) * new_pred


def _compute_leaf_derivatives(remove_idx, leaf_counts, leaf2docs, denominator,
                              da_vector_multiplier, naive_gradient_addendum,
                              n_boost, n_class, n_train, learning_rate, update_set):
    """
    Compute leaf value derivatives based on the example being removed.

    Return
        - 1d array of leaf value derivatives of shape=(total no. leaves,).

    Note
        - Parallelizable method.
    """
    leaf_derivatives = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)

    # intermediate containers
    da = np.zeros((n_train, n_class), dtype=util.dtype_t)
    tree_idx = 0
    n_prev_leaves = 0

    for boost_idx in range(n_boost):

        for class_idx in range(n_class):

            leaf_count = leaf_counts[boost_idx, class_idx]
            update_docs = _get_docs_to_update(update_set, leaf_count, leaf2docs[tree_idx], remove_idx, da)

            for leaf_idx in range(leaf_count):

                # get intersection of leaf documents and update documents
                leaf_docs = leaf2docs[tree_idx][leaf_idx]
                update_leaf_docs = sorted(update_docs.intersection(leaf_docs))

                # compute and save leaf derivative
                grad_enumerator = np.dot(da[update_leaf_docs, class_idx],
                                         da_vector_multiplier[update_leaf_docs, boost_idx, class_idx])

                if remove_idx in update_leaf_docs:
                    grad_enumerator += naive_gradient_addendum[remove_idx, boost_idx, class_idx]

                leaf_derivative = -grad_enumerator / denominator[n_prev_leaves + leaf_idx] * learning_rate

                # update da
                da[update_leaf_docs, class_idx] += leaf_derivative

                # save
                leaf_derivatives[n_prev_leaves + leaf_idx] = leaf_derivative

            n_prev_leaves += leaf_count
            tree_idx += 1

    return leaf_derivatives


def _get_docs_to_update(update_set, leaf_count, leaf_docs, remove_idx, da):
    """
    Return a set of document indices to be udpated for this tree.

    Return
        - Set of training indices.

    Note
        -Parallelizable method.
    """

    # update only the remove example
    if update_set == 0:
        result = set({remove_idx})

    # update all train
    elif update_set == -1:
        result = set(np.arange(da.shape[0], dtype=np.int32))  # shape=(no. train,)

    # update examples for the top leaves
    else:

        # sort leaf indices based on largest abs. da sum
        leaf_das = [np.sum(np.abs(da[list(leaf_docs[leaf_idx])])) for leaf_idx in range(leaf_count)]
        top_leaf_ids = np.argsort(leaf_das)[-update_set:]
        
        # return remove_idx + document indices for the top `k` leaves
        result = {remove_idx}
        for leaf_idx in top_leaf_ids:
            result |= leaf_docs[leaf_idx]

    return result
