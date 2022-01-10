import time
import joblib

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class LeafRefit(Explainer):
    """
    LeafRefit: Leave-one-out (LOO) keeping the structure fixed.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y, F_{w/o x_i}(x_t)) - L(y, F(x_t))
        - Pos. value means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
        - Neg. value means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Note
        - Does NOT take class or instance weight into account.
        - Assumes leaf values are estimated using a single Newton step.

    Reference
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/influence/leaf_refit.py

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

    def fit(self, model, X, y):
        """
        - Compute and save gradient and hessian sums for each leaf.
            * Compute leaf values using sums.

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
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        assert self.model_.tree_type != 'rf', 'RF not supported for LeafRefit'

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
        original_approx = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)
        gradients = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)
        hessians = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)

        sum_gradients = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        sum_hessians_l2 = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        leaf_values = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)

        n_not_close = 0
        max_diff = 0

        # save gradient information of leaf values for each tree
        for boost_idx in range(n_boost):
            doc_preds = np.zeros((X.shape[0], n_class), dtype=util.dtype_t)

            # precompute gradient statistics
            gradients[:, boost_idx, :] = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)
            hessians[:, boost_idx, :] = self.loss_fn_.hessian(y, current_approx)  # shape=(X.shape[0], no. class)

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
                    sum_gradient = np.sum(gradients[doc_ids, boost_idx, class_idx])
                    sum_hessian_l2 = np.sum(hessians[doc_ids, boost_idx, class_idx]) + l2_leaf_reg
                    leaf_value = -sum_gradient / sum_hessian_l2 * learning_rate

                    # compare leaf values to actual leaf values
                    if not np.isclose(leaf_value, leaf_vals[leaf_idx], atol=1e-5):
                        n_not_close += 1
                        max_diff = max(max_diff, abs(leaf_value - leaf_vals[leaf_idx]))

                    # store statistics
                    sum_gradients[n_prev_leaves + leaf_idx] = sum_gradient
                    sum_hessians_l2[n_prev_leaves + leaf_idx] = sum_hessian_l2
                    leaf_values[n_prev_leaves + leaf_idx] = leaf_value

                n_prev_leaves += leaf_count  # move to next set of tree leaves
                leaf2docs.append(leaf2doc)  # list of dicts, one per tree

            current_approx += doc_preds  # update approximation
            original_approx[:, boost_idx, :] = current_approx.copy()

        # copy and compute new leaf values resulting from the removal of each x in X.
        start = time.time()
        if self.logger:
            self.logger.info(f'\n[INFO] no. leaf vals not within 1e-5 tol.: {n_not_close:,}, '
                             f'max. diff.: {max_diff:.5f}')
            self.logger.info(f'\n[INFO] computing alternate leaf values...')

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
            self.logger.info(f'[INFO] no. cpus: {n_jobs:,}...')

        # process each training example removal in parallel
        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            # result container
            new_leaf_values = np.zeros((0, np.sum(leaf_counts)), dtype=util.dtype_t)

            # trackers
            n_completed = 0
            n_remaining = X.shape[0]

            # get number of fits to perform for this iteration
            while n_remaining > 0:
                n = min(100, n_remaining)

                results = parallel(joblib.delayed(_compute_new_leaf_values)
                                                 (train_idx, leaf_counts, leaf2docs, gradients, hessians,
                                                  sum_gradients, sum_hessians_l2, leaf_values, original_approx, y,
                                                  n_boost, n_class, X.shape[0], learning_rate, self.update_set,
                                                  self.loss_fn_) for train_idx in range(n_completed,
                                                                                        n_completed + n))

                # synchronization barrier
                results = np.vstack(results)  # shape=(n, 1 or X_test.shape[0])
                new_leaf_values = np.vstack([new_leaf_values, results])

                n_completed += n
                n_remaining -= n

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO - LR] {n_completed:,} / {X.shape[0]:,}, cum. time: {cum_time:.3f}s')

        # save results of this method
        self.leaf_values_ = leaf_values  # shape=(total no. leaves,)
        self.new_leaf_values_ = new_leaf_values  # shape=(no. train, total no. leaves)
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

        influence = np.zeros((self.n_train_, X.shape[0]), dtype=util.dtype_t)

        if self.logger:
            self.logger.info('\n[INFO] computing influence for each test example...')

        # compute influence of each training example on the test example
        for remove_idx in range(self.n_train_):
            influence[remove_idx] = self._loss_delta(X, y, remove_idx)  # shape=(X.shape[0],)

        return influence

    # private
    def _loss_delta(self, X, y, remove_idx):
        """
        Compute the influence on the set of examples (X, y) using the updated
            set of leaf values resulting from removing `remove_idx`.

        Input
            X: 2d array of test examples
            y: 1d array of test targets.
            remove_idx: index of removed train instance

        Return
            - Array of test loss differences of shape=(X.shape[0],).
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        og_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        new_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)

        # get prediction of each test example using the original and new leaf values
        n_prev_leaves = 0

        for boost_idx in range(self.n_boost_):  # per boosting iteration
            for class_idx in range(self.n_class_):  # per class

                for test_idx in range(X.shape[0]):  # per test example
                    leaf_idx = doc2leaf[test_idx][boost_idx][class_idx]
                    og_pred[test_idx, class_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                    new_pred[test_idx, class_idx] += self.new_leaf_values_[remove_idx][n_prev_leaves + leaf_idx]

                n_prev_leaves += self.leaf_counts_[boost_idx, class_idx]

        return self.loss_fn_(y, new_pred) - self.loss_fn_(y, og_pred)


def _compute_new_leaf_values(remove_idx, leaf_counts, leaf2docs, gradients, hessians,
                             sum_gradients, sum_hessians_l2, leaf_values, original_approx, y,
                             n_boost, n_class, n_train, learning_rate, update_set, loss_fn):
    """
    Compute new leaf values based on the example being removed.

    Return
        - 1d array of new leaf values of shape=(total no. leaves,).

    Note
        - Parallelizable method.
    """
    new_leaf_values = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)

    # intermediate containers
    doc_deltas = np.zeros((n_train, n_class), dtype=util.dtype_t)
    tree_idx = 0
    n_prev_leaves = 0

    for boost_idx in range(n_boost):
        update_approx = original_approx[:, boost_idx, :] + doc_deltas

        for class_idx in range(n_class):

            leaf_count = leaf_counts[boost_idx, class_idx]
            update_docs = _get_docs_to_update(update_set, leaf_count, leaf2docs[tree_idx], remove_idx, doc_deltas)

            for leaf_idx in range(leaf_count):

                # get intersection of leaf documents and update documents
                leaf_docs = leaf2docs[tree_idx][leaf_idx]
                update_leaf_docs = update_docs.intersection(leaf_docs)
                update_leaf_docs.discard(remove_idx)
                update_leaf_docs = sorted(update_leaf_docs)

                # update gradients and hessians based on updated predictions
                if len(update_leaf_docs) > 0:
                    update_gradients = loss_fn.gradient(y[update_leaf_docs],
                                                        update_approx[update_leaf_docs])[:, class_idx]
                    update_sum_gradient = np.sum(update_gradients - gradients[update_leaf_docs, boost_idx, class_idx])

                    update_hessians = loss_fn.hessian(y[update_leaf_docs],
                                                      update_approx[update_leaf_docs])[:, class_idx]
                    update_sum_hessian_l2 = np.sum(update_hessians - hessians[update_leaf_docs, boost_idx, class_idx])

                # no other training examples affected
                else:
                    update_sum_gradient = 0
                    update_sum_hessian_l2 = 0

                # remove effect of target training example
                if remove_idx in leaf_docs:
                    update_sum_gradient -= gradients[remove_idx, boost_idx, class_idx]
                    update_sum_hessian_l2 -= hessians[remove_idx, boost_idx, class_idx]

                # compute new leaf value and leaf value delta
                new_sum_gradient = sum_gradients[n_prev_leaves + leaf_idx] + update_sum_gradient
                new_sum_hessian_l2 = sum_hessians_l2[n_prev_leaves + leaf_idx] + update_sum_hessian_l2

                new_leaf_value = -new_sum_gradient / new_sum_hessian_l2 * learning_rate
                leaf_value_delta = new_leaf_value - leaf_values[n_prev_leaves + leaf_idx]

                # update prediction deltas
                doc_deltas[update_leaf_docs, class_idx] += leaf_value_delta

                # save
                new_leaf_values[n_prev_leaves + leaf_idx] = new_leaf_value

            n_prev_leaves += leaf_count
            tree_idx += 1

    return new_leaf_values


def _get_docs_to_update(update_set, leaf_count, leaf_docs, remove_idx, deltas):
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
        result = set(np.arange(deltas.shape[0], dtype=np.int32))  # shape=(no. train,)

    # update examples for the top leaves
    else:

        # sort leaf indices based on largest abs. deltas sum
        leaf_deltas = [np.sum(np.abs(deltas[list(leaf_docs[leaf_idx])])) for leaf_idx in range(leaf_count)]
        top_leaf_ids = np.argsort(leaf_deltas)[-update_set:]
        
        # return remove_idx + document indices for the top `k` leaves
        result = {remove_idx}
        for leaf_idx in top_leaf_ids:
            result |= leaf_docs[leaf_idx]

    return result
