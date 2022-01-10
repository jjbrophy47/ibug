import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from .base import Explainer
from .parsers import util


class Trex(Explainer):
    """
    Tree-Ensemble Representer Examples: Explainer that adapts the
    Representer-point method for deep-learning models to tree ensembles.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(phi(sum_j * (alpha_{j=i}=0) K(x_i, x_t)) - L(phi(sum_i alpha_i k(x_i, x_t))).
        - Phi is the activation of the pre-activation prediction.
        - Alpha_i has y_i baked into it if the kernelized model uses actual labels, i.e. alpha_i = alpha_i * y_i.
        - Pos. value means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
        - Neg. value means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Reference
         - https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py

    Paper
        - https://arxiv.org/abs/1811.09720

    Note
        - Supports both GBDTs and RFs.
        - Ordering of local influence using loss approx. magnitude is not necessarily the
            same as using the magniutde of representer values since the activation of the
            loss function can cause slightly differing orderings.
    """
    def __init__(self, kernel='lpw', target='actual', lmbd=0.003, n_epoch=3000,
                 random_state=1, logger=None):
        """
        Input
            kernel: str, Transformation of the input using the tree-ensemble structure.
                'to_': Tree output; output of each tree in the ensemble.
                'lp_': Leaf path; one-hot encoding of leaf indices across all trees.
                'lpw': Weighted leaf path; like 'lp' but replaces 1s with 1 / leaf count.
                'lo_': Leaf output; like 'lp' but replaces 1s with leaf values.
                'low': Weighted leaf otput; like 'lo' but replace leaf value with 1 / leaf value.
                'fp_': Feature path; one-hot encoding of node indices across all trees.
                'fpw': Weighted feature path; like 'fp' but replaces 1s with 1 / node count.
                'fo_': Feature output; like 'fp' but replaces leaf 1s with leaf values.
                'fow': Weighted feature output; like 'fo' but replaces leaf 1s with 1 / leaf values.
            target: str, Targets for the linear model to train on.
                'actual': Ground-truth targets.
                'predicted': Predicted targets from the tree-ensemble.
            lmbd: float, Regularizer for the linear model; necessary for the Representer decomposition.
            n_epoch: int, Max. no. epochs to train the linear model.
            random_state: int, Random state seed to generate reproducible results.
            logger: object, If not None, output to logger.
        """
        assert kernel in ['to_', 'lp_', 'lpw', 'lo_', 'low', 'fp_', 'fpw', 'fo_', 'fow']
        assert target in ['actual', 'predicted']
        assert isinstance(lmbd, float)
        self.kernel = kernel
        self.target = target
        self.lmbd = lmbd
        self.n_epoch = n_epoch
        self.random_state = random_state
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Transform the input data using the specified tree kernel.
        - Fit linear model and compute train instance weights (alpha).

        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.model_.update_node_count(X)

        self.n_class_ = self.model_.n_class_
        self.X_train_ = self._kernel_transform(X)
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        # select target
        if self.target == 'actual':

            if self.model_.objective == 'regression':  # shape=(no. train,)
                self.y_train_ = y

            elif self.model_.objective == 'binary':  # shape=(no. train,)
                self.y_train_ = y

            elif self.model_.objective == 'multiclass':  # shape=(no. train, no. class)
                self.y_train_ = LabelBinarizer().fit_transform(y)

        elif self.target == 'predicted':

            if self.model_.objective == 'regression':  # shape=(no. train,)
                self.y_train_ = model.predict(X)

            elif self.model_.objective == 'binary':  # shape=(no. train,)
                self.y_train_ = model.predict_proba(X)[:, 1]

            elif self.model_.objective == 'multiclass':  # shape=(no. train, no. class)
                self.y_train_ = model.predict_proba(X)

        self.alpha_ = self._compute_train_weights(self.X_train_, self.y_train_)

        return self

    def get_local_influence(self, X, y, verbose=1):
        """
        - Compute influence of each train examples on each test example loss.

        Input
            X: 2d array of test data.
            y: 2d array of test targets.
            verbose: int, verbosity.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the traing data.
        """
        start = time.time()

        X, y = util.check_data(X, y, objective=self.model_.objective)

        X_test_ = self._kernel_transform(X)  # shape=(X.shape[0], no. feature)
        sim = np.matmul(self.X_train_, X_test_.T)  # shape=(no. train, X.shape[0])

        # intermediate result, shape=(no. train, X.shape[0], no. class)
        rep_vals = np.zeros((self.X_train_.shape[0], X.shape[0], self.n_class_), dtype=util.dtype_t)

        for class_idx in range(self.n_class_):  # per class
            rep_vals[:, :, class_idx] = sim * self.alpha_[:, class_idx].reshape(-1, 1)

        # compute pre-act. pred. with and without each train example
        rep_vals_sum = np.sum(rep_vals, axis=0)  # sum over train, shape=(X.shape[0], no. class)
        rep_vals_delta = rep_vals_sum - rep_vals  # shape=(no. train, X.shape[0], no. class)

        # compute losses with and then without each train example
        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=util.dtype_t)
        original_losses = self.loss_fn_(y, rep_vals_sum, raw=True)  # shape=(X.shape[0],)

        # compute losses without each train example, and their influences
        for test_idx in range(X.shape[0]):
            y_temp = np.tile(y[test_idx], (self.X_train_.shape[0], 1))  # shape=(no.train, no. class)
            removed_losses = self.loss_fn_(y_temp, rep_vals_delta[:, test_idx, :], raw=True)  # shape=(no. train,)
            influence[:, test_idx] = removed_losses - original_losses[test_idx]  # shape=(no. train,)

            # progress
            if test_idx > 0 and (test_idx + 1) % 100 == 0 and self.logger and verbose:
                self.logger.info(f'[INFO - TREX] No. finished: {test_idx+1:>10,} / {X.shape[0]:>10,}, '
                                 f'cum. time: {time.time() - start:.3f}s')

        return influence

    # private
    def _kernel_transform(self, X):
        """
        Transforms each x in X using the specified tree kernel.

        Return
            - 2d array of shape=(X.shape[0], no. kernel features).
        """
        structure_dict = {'t': 'tree', 'l': 'leaf', 'f': 'feature'}
        output_dict = {'p': 'path', 'o': 'output'}
        weight_dict = {'_': 'unweighted', 'w': 'weighted'}
        
        s1, s2, s3 = list(self.kernel)
        structure = structure_dict[s1]
        output = output_dict[s2]
        weight = weight_dict[s3]

        if structure == 'tree':
            X_ = self._tree_kernel_transform(X)

        elif structure == 'leaf':
            X_ = self._leaf_kernel_transform(X, output=output, weight=weight)

        elif structure == 'feature':
            X_ = self._feature_kernel_transform(X, output=output, weight=weight)

        return X_

    def _tree_kernel_transform(self, X):
        """
        Transform each x in X to be a vector of tree outputs.

        Return
            - Regression and binary: 2d array of shape=(no. train, no. trees).
            - Multiclass: 2d array of shape=(no. train, no. trees * no. class).
        """
        trees = self.model_.trees.flatten()
        X_ = np.zeros((X.shape[0], trees.shape[0]))

        for i, tree in enumerate(trees):
            X_[:, i] = tree.predict(X)

        return X_

    def _leaf_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded leaf paths.
        - The `output` and `weight` parameters control the value of the 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. leaves).
            - Multiclass: 2d array of shape=(no. train, ~total no. leaves * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_leaves = np.sum([tree.leaf_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_leaves))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_leaves = 0
        for tree in trees:
            start = n_prev_leaves
            stop = n_prev_leaves + tree.leaf_count_
            X_[:, start: stop] = tree.leaf_path(X, output=output, weighted=weighted)
            n_prev_leaves += tree.leaf_count_

        return X_

    def _feature_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded feature paths.
        - The `output` parameter controls the value of the leaf 1s.
        - The `weight` parameter controls the value of all 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. nodes).
            - Multiclass: 2d array of shape=(no. train, ~total no. nodes * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_nodes = np.sum([tree.node_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_nodes))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_nodes = 0
        for tree in trees:
            start = n_prev_nodes
            stop = n_prev_nodes + tree.node_count_
            X_[:, start: stop] = tree.feature_path(X, output=output, weighted=weighted)
            n_prev_nodes += tree.node_count_

        return X_

    def _compute_train_weights(self, X, y):
        """
        Fit a linear model to X and y, then extract weights for all
        train instances.
        """
        X = Variable(torch.FloatTensor(X))
        y = Variable(torch.FloatTensor(y))
        N = len(y)

        # randomly initialize weights
        rng = np.random.default_rng(self.random_state)

        # MSE model for regression, Softmax model for classification
        if self.model_.objective == 'regression':
            assert y.ndim == 1
            W = rng.uniform(-1, 1, size=X.shape[1])
            model = MSEModel(W)

        elif self.model_.objective == 'binary':
            assert y.ndim == 1
            W = rng.uniform(-1, 1, size=X.shape[1])
            model = LogisticModel(W)

        else:
            assert self.model_.objective == 'multiclass'
            assert y.ndim == 2
            W = rng.uniform(-1, 1, size=(X.shape[1], y.shape[1]))
            model = SoftmaxModel(W)

        # optimization settings
        min_loss = 10000.0
        optimizer = optim.SGD([model.W], lr=1.0)
        prev_grad_loss = -1.0

        if self.logger:
            self.logger.info(f'\n[INFO] computing alpha values...')
            start = time.time()

        # train
        for epoch in range(self.n_epoch):
            phi_loss = 0

            optimizer.zero_grad()
            (Phi, L2) = model(X, y)
            loss = Phi / N + L2 * self.lmbd

            phi_loss += util.to_np(Phi / N)
            loss.backward()

            temp_W = model.W.data
            grad_loss = util.to_np(torch.mean(torch.abs(model.W.grad)))

            # save the W with lowest loss
            if grad_loss < min_loss:

                if epoch == 0:
                    init_grad = grad_loss

                min_loss = grad_loss
                best_W = temp_W

                if min_loss < init_grad / 200:
                    if self.logger:
                        self.logger.info(f'[INFO] stopping criteria reached in epoch: {epoch}')
                    break

            # stop early if grad. has not changed
            if epoch % 100 == 0:

                if grad_loss == prev_grad_loss:
                    if self.logger:
                        self.logger.info(f'[INFO] Epoch: {epoch:4d}, loss: {util.to_np(loss):.7f}'
                                         f', phi_loss: {phi_loss:.7f}, grad: {grad_loss:.7f}'
                                         f', cum. time: {time.time() - start:.3f}s')
                        self.logger.info(f'[INFO] grad. has not changed in 100 epochs, stopping early...')
                    break

                else:
                    if self.logger:
                        self.logger.info(f'[INFO] Epoch: {epoch:4d}, loss: {util.to_np(loss):.7f}'
                                         f', phi_loss: {phi_loss:.7f}, grad: {grad_loss:.7f}'
                                         f', cum. time: {time.time() - start:.3f}s')
                    prev_grad_loss = grad_loss

            self._backtracking_line_search(model, model.W.grad, X, y, loss)

        # compute alpha based on the representer theorem's decomposition
        output = torch.matmul(X, Variable(best_W))  # shape=(no. train, no. class)

        if self.model_.objective == 'binary':
            output = torch.sigmoid(output)

        elif self.model_.objective in 'multiclass':
            output = torch.softmax(output, axis=1)

        alpha = output - y  # 1/2 grad. of mse; grad. of logistic loss; grad. of softmax cross-entropy
        alpha = torch.div(alpha, (-2.0 * self.lmbd * N))

        # compute W based on the Representer Theorem decomposition
        W = torch.matmul(torch.t(X), alpha)  # shape=(no. features,) or (no. features, no. class)

        output_approx = torch.matmul(X, W)

        if self.model_.objective == 'binary':
            output_approx = torch.sigmoid(output_approx)

        elif self.model_.objective == 'multiclass':
            output_approx = torch.softmax(output_approx, axis=1)

        # compute closeness
        y = util.to_np(y).flatten()
        yp = util.to_np(output_approx).flatten()

        l1_diff = np.mean(np.abs(y - yp))
        p_corr, _ = pearsonr(y, yp)
        s_corr, _ = spearmanr(y, yp)

        if self.logger:
            self.logger.info(f'[INFO] L1 diff.: {l1_diff:.5f}, pearsonr: {p_corr:.5f}, spearmanr: {s_corr:.5f}')

        self.l1_diff_ = l1_diff
        self.p_corr_ = p_corr
        self.s_corr_ = s_corr

        # convert alpha to numpy and reshape if necessary
        alpha = util.to_np(alpha)
        if self.model_.objective in ['regression', 'binary']:
            alpha = alpha.reshape(-1, 1)

        return alpha

    def _backtracking_line_search(self, model, grad, X, y, val, beta=0.5):
        """
        Search for and then take the biggest possible step for gradient descent.
        """
        N = X.shape[0]

        t = 10.0
        W_O = util.to_np(model.W)
        grad_np = util.to_np(grad)

        while(True):
            model.W = Variable(torch.from_numpy(W_O - t * grad_np).type(torch.float32), requires_grad=True)

            val_n = 0.0
            (Phi, L2) = model(X, y)
            val_n = Phi / N + L2 * self.lmbd

            if t < 0.0000000001:
                print("t too small")
                break

            if util.to_np(val_n - val + t * torch.norm(grad) ** 2 / 2) >= 0:
                t = beta * t
            else:
                break


class MSEModel(nn.Module):
    """
    Mean squared error.

    Note
        - This loss function represents "closeness" if y is predicted probabilities.
    """

    def __init__(self, W):
        super(MSEModel, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(torch.float32), requires_grad=True)

    def forward(self, X, y):
        """
        Calculate loss for the loss function and L2 regularizer.

        Note
            - This loss function represents "closeness" if y is predicted values.
        """
        D = torch.matmul(X, self.W)  # raw output, shape=(X.shape[0],)
        Phi = torch.sum(torch.square(D - y))  # MSE loss

        # L2 norm.
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))

        return (Phi, L2)


class LogisticModel(nn.Module):
    """
    Simgoid + logistic loss.

    Note
        - This loss function represents "closeness" if y is predicted probabilities.
    """

    def __init__(self, W):
        super(LogisticModel, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(torch.float32), requires_grad=True)

    def forward(self, X, y):
        """
        Calculate loss for the loss function and L2 regularizer.

        Note
            - This loss function represents "closeness" if y is predicted values.
        """
        eps = 1e-5

        D = torch.matmul(X, self.W)  # raw output, shape=(X.shape[0],)
        D = torch.sigmoid(D)  # normalized prob.
        D = torch.clip(D, eps, 1 - eps)  # prevent log(0)

        Phi = torch.sum(-(y * torch.log(D) + (1 - y) * torch.log(1 - D)))  # log loss

        # L2 norm.
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))

        return (Phi, L2)


class SoftmaxModel(nn.Module):
    """
    Softmax + categorical cross-entropy.
    """

    def __init__(self, W):
        super(SoftmaxModel, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(torch.float32), requires_grad=True)

    def forward(self, X, y):
        """
        Calculate loss for the loss function and L2 regularizer.

        Note
            - This loss function represents "closeness" if y is predicted probabilities.
        """
        D = torch.matmul(X, self.W)  # raw output, shape=(X.shape[0], no. class)
        D = D - torch.logsumexp(D, axis=1).reshape(-1, 1)  # softmax: normalize log probs.
        Phi = torch.sum(-torch.sum(D * y, axis=1))  # cross-entropy loss

        # L2 norm.
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))

        return (Phi, L2)
