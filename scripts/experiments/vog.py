"""
Compute variance of gradients throughout training.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # intent
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import util
from config import exp_args
from intent.explainers.parsers import parse_model
from intent.explainers.parsers import util as util_it


def compute_losses(y, y_hat, objective, eps=1e-5):
    """
    Compute individual losses.

    Input
        y_hat: 2d array of activated predictions;
            shape=(no. train, 1) for regression/binary,
            shape=(no. train, no. class) for multiclass.
        y: 1d array of label data.

    Return 1d array of individual losses of shape=(X.shape[0],)
    """

    if objective == 'regression':  # squared error
        result = 0.5 * (y - y_hat[:, 0]) ** 2  # shape=(X.shape[0],)

    elif objective == 'binary':  # log loss
        y_hat_pos = np.clip(y_hat[:, 0], eps, 1 - eps)  # prevent log(0)
        result = -(y * np.log(y_hat_pos) + (1 - y) * np.log(1 - y_hat_pos))

    else:
        assert objective == 'multiclass'  # cross-entropy loss
        y_hat = np.clip(y_hat, eps, 1 - eps)  # shape=(X.shape[0], no. class)
        y_hat = -np.log(y_hat)  # shape=(X.shape[0], no. class)
        result = np.array([y_hat[i, target] for i, target in enumerate(y)])

    return result


def compute_dynamics(model, X, y):
    """
    Compute gradients, predictions (with activation), and losses after each boosting iteration.

    Input
        model: TreeEnsemble object (parsed GBDT model).
        X: 2d array of train data.

    Returns 2 3d arrays: gradients, predictions with shape=(X.shape[0], no. boost, no. class),
        and one 2d array of losses with shape=(X.shape[0], no. boost).
    """
    trees = model.trees
    n_boost = model.n_boost_
    n_class = model.n_class_
    bias = model.bias
    loss_fn = util_it.get_loss_fn(model.objective, model.n_class_, model.factor)

    current_approx = np.tile(bias, (X.shape[0], 1)).astype(util_it.dtype_t)  # shape=(X.shape[0], no. class)

    predictions = np.zeros((X.shape[0], n_boost, n_class))  # shape=(X.shape[0], no. boost, no. class)
    gradients = np.zeros((X.shape[0], n_boost, n_class))  # shape=(X.shape[0], no. boost, no. class)
    losses = np.zeros((X.shape[0], n_boost))  # shape=(X.shape[0], no. boost)

    # compute gradients for each boosting iteration
    for boost_idx in range(n_boost):

        # activation
        if model.tree_type == 'rf':
            pred = current_approx / boost_idx

        else:  # gbdt

            if model.objective == 'binary':
                pred = util_it.sigmoid(current_approx)

            elif model.objective == 'multiclass':
                pred = util_it.softmax(current_approx)

            else:
                pred = current_approx

        predictions[:, boost_idx, :] = pred  # shape=(X.shape[0], no. class)
        gradients[:, boost_idx, :] = loss_fn.gradient(y, current_approx)  # shape=(X.shape[0], no. class)
        losses[:, boost_idx] = compute_losses(y, pred, model.objective)  # shape=(X.shape[0],)

        # update approximation
        for class_idx in range(n_class):
            current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

    return predictions, gradients, losses


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_train, y_train, logger, prefix='Train')
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # parse tree ensemble
    start = time.time()
    model = parse_model(tree, X_train, y_train)
    logger.info(f'\nparse time: {time.time() - start:.3f}s')

    # sanity checks
    X_train, y_train = util_it.check_data(X_train, y_train, objective=model.objective)
    assert model.tree_type != 'rf', 'RF not supported!'

    # compute variance of gradients throughout training
    predictions, gradients, losses = compute_dynamics(model, X_train, y_train)

    # save results
    result['predictions'] = predictions  # shape=(no. train, no. boost, no. class)
    result['gradients'] = gradients  # shape=(no. train, no. boost, no. class)
    result['losses'] = losses  # shape=(no. train, no. boost)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           args.dataset)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, out_dir)

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_vog_args().parse_args())
