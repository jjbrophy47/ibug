"""
Utility methods.
"""
import os
import sys
import shutil
import logging
import hashlib
import numpy as np

import properscoring as ps
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from pgbm import PGBMRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

# constants
dtype_t = np.float64


# public
def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Clear all of the logger handlers.
    """
    logger.handlers = []


def clear_dir(in_dir):
    """
    Clear contents of directory.
    """
    if not os.path.exists(in_dir):
        return -1

    # remove contents of the directory
    for fn in os.listdir(in_dir):
        fp = os.path.join(in_dir, fn)

        # directory
        if os.path.isdir(fp):
            shutil.rmtree(fp)

        # file
        else:
            os.remove(fp)

    return 0


def stdout_stderr_to_log(filename):
    """
    Log everything printed to stdout or
    stderr to this specified `filename`.
    """
    logfile = open(filename, 'w')

    stderr = sys.stderr
    stdout = sys.stdout

    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = sys.stdout

    return logfile, stdout, stderr


def reset_stdout_stderr(logfile, stdout, stderr):
    """
    Restore original stdout and stderr
    """
    sys.stdout = stdout
    sys.stderr = stderr
    logfile.close()


def get_data(data_dir, dataset, feature=False):
    """
    Return train and test data for the specified dataset.
    """
    datasets = ['ames_housing', 'cal_housing', 'concrete', 'energy', 'heart', 'kin8nm',
                'life', 'msd', 'naval', 'obesity', 'online_news', 'power', 'protein',
                'synth_regression', 'wine', 'yacht']
    assert dataset in datasets

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'), allow_pickle=True)[()]
    X_train, y_train = data['X_train'].astype(dtype_t), data['y_train'].astype(dtype_t)
    X_test, y_test = data['X_test'].astype(dtype_t), data['y_test'].astype(dtype_t)
    feature_names = data['feature']
    objective = 'regression'

    if feature:
        result = X_train, X_test, y_train, y_test, feature_names, objective
    else:
        result = X_train, X_test, y_train, y_test, objective

    return result


def get_model(tree_type='lgb', n_tree=100, max_depth=5, random_state=1, max_bins=255,
              learning_rate=0.3):
    """
    Return the ensemble object from the specified framework and objective.
    """
    if tree_type == 'cb':
        tree = CatBoostRegressor(n_estimators=n_tree, max_depth=max_depth,
                                 leaf_estimation_iterations=1, random_state=random_state,
                                 logging_level='Silent', learning_rate=learning_rate)

    elif tree_type == 'lgb':
        tree = LGBMRegressor(n_estimators=n_tree, max_depth=max_depth, num_leaves=2 ** max_depth,
                             random_state=random_state)

    elif tree_type == 'sgb':
        tree = HistGradientBoostingRegressor(max_iter=n_tree, max_depth=max_depth, max_leaf_nodes=2 ** max_depth,
                                             random_state=random_state, max_bins=max_bins, early_stopping=False)

    elif tree_type == 'skgbm':
        tree = GradientBoostingRegressor(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

    elif tree_type == 'skrf':
        tree = RandomForestRegressor(n_estimators=n_tree, max_depth=max_depth,
                                     random_state=random_state, bootstrap=False)

    elif tree_type == 'xgb':
        tree = XGBRegressor(n_estimators=n_tree, max_depth=max_depth,
                            random_state=random_state, tree_method='hist')

    elif tree_type == 'pgbm':
        tree = PGBMRegressor(n_estimators=n_tree, random_state=random_state, max_leaves=2 ** max_depth,
                             verbose=0)

    elif tree_type == 'ngboost':
        tree = NGBRegressor(n_estimators=n_tree, random_state=random_state)

    else:
        raise ValueError(f'Unknown tree_type {tree_type}')

    return tree


def eval_pred(y, yhat=None, model=None, X=None, logger=None, prefix=''):
    """
    Evaluate the predictive performance of the model on X and y.
    """
    if yhat is None:
        assert model is not None and X is not None
        yhat = model.predict(X)

    rmse = mean_squared_error(y, yhat, squared=False)  # RMSE
    mae = mean_absolute_error(y, yhat)

    if logger:
        logger.info(f'[{prefix}] RMSE: {rmse:.5f}, MAE: {mae:.5f}, ')

    return rmse, mae


def eval_normal(y, loc, scale, nll=True, crps=False):
    """
    Evaluate each predicted normal distribution.

    Input
        X: 2d array of data.
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


def eval_loss(objective, model, X, y, logger=None, prefix='', eps=1e-5):
    """
    Return individual losses.
    """
    assert X.shape[0] == y.shape[0] == 1

    result = {}

    if objective == 'regression':
        y_hat = model.predict(X)  # shape=(X.shape[0],)
        losses = 0.5 * (y - y_hat) ** 2
        result['pred'] = y_hat
        result['pred_label'] = y_hat[0]
        result['loss'] = losses[0]
        loss_type = 'squared_loss'

    elif objective == 'binary':
        y_hat = model.predict_proba(X)  # shape=(X.shape[0], 2)
        y_hat_pos = np.clip(y_hat[:, 1], eps, 1 - eps)  # prevent log(0)
        losses = -(y * np.log(y_hat_pos) + (1 - y) * np.log(1 - y_hat_pos))
        result['pred'] = y_hat
        result['pred_label'] = np.argmax(y_hat.flatten())
        result['loss'] = losses[0]
        loss_type = 'logloss'

    else:
        assert objective == 'multiclass'
        target = y[0]
        y_hat = model.predict_proba(X)[0]  # shape=(X.shape[0], no. class)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        result['pred'] = y_hat
        result['pred_label'] = np.argmax(y_hat.flatten())
        result['loss'] = -np.log(y_hat)[target]
        loss_type = 'cross_entropy_loss'

    result['keys'] = ['loss']

    if logger:
        with np.printoptions(formatter={'float': '{:0.5f}'.format}):
            logger.info(f"[{prefix}] prediction: {result['pred']}, "
                        f"pred. label: {result['pred_label']:>10.3f}, "
                        f"{loss_type}: {result['loss']:>10.3f}, ")

    return result


def dict_to_hash(my_dict, skip=[]):
    """
    Convert to string and concatenate the desired values
    in `my_dict` and return the hashed string.
    """
    d = my_dict.copy()

    # remove keys not desired in the hash string
    for key in skip:
        if key in d:
            del d[key]

    s = ''.join(str(v) for k, v in sorted(d.items()))  # alphabetical key sort

    result = hashlib.md5(s.encode('utf-8')).hexdigest() if s != '' else ''

    return result


def explainer_params_to_dict(explainer, exp_params):
    """
    Return dict of explainer hyperparameters.
    """
    params = {}

    if explainer in ['boostin', 'boostinW1', 'boostinW2']:
        pass

    elif explainer in ['boostinLE', 'boostinLEW1', 'boostinLEW2']:
        pass

    elif explainer in ['leaf_inf', 'leaf_infLE']:
        params['update_set'] = exp_params['leaf_inf_update_set']
        params['atol'] = exp_params['leaf_inf_atol']
        params['n_jobs'] = exp_params['n_jobs']

    elif explainer in ['leaf_refit', 'leaf_refitLE']:
        params['update_set'] = exp_params['leaf_inf_update_set']
        params['atol'] = exp_params['leaf_inf_atol']
        params['n_jobs'] = exp_params['n_jobs']

    elif explainer in ['leaf_infSP', 'leaf_infSPLE']:
        pass

    elif explainer == 'trex':
        params['kernel'] = exp_params['tree_kernel']
        params['target'] = exp_params['trex_target']
        params['lmbd'] = exp_params['trex_lmbd']
        params['n_epoch'] = exp_params['trex_n_epoch']
        params['random_state'] = exp_params['random_state']

    elif explainer in ['loo', 'looLE']:
        params['n_jobs'] = exp_params['n_jobs']

    elif explainer == 'dshap':
        params['trunc_frac'] = exp_params['dshap_trunc_frac']
        params['check_every'] = exp_params['dshap_check_every']
        params['n_jobs'] = exp_params['n_jobs']
        params['random_state'] = exp_params['random_state']

    elif explainer == 'subsample':
        params['sub_frac'] = exp_params['subsample_sub_frac']
        params['n_iter'] = exp_params['subsample_n_iter']
        params['n_jobs'] = exp_params['n_jobs']
        params['random_state'] = exp_params['random_state']

    elif explainer == 'random':
        params['random_state'] = exp_params['random_state']

    elif explainer == 'input_sim':
        params['measure'] = exp_params['input_sim_measure']

    elif explainer == 'tree_sim':
        params['measure'] = exp_params['tree_sim_measure']
        params['kernel'] = exp_params['tree_kernel']

    elif explainer == 'leaf_sim':
        pass

    # create hash string based on the chosen hyperparameters
    hash_str = dict_to_hash(params, skip=['n_jobs', 'random_state', 'atol'])

    return params, hash_str


def get_hyperparams(tree_type, dataset):
    """
    Input
        tree_type: str, Tree-ensemble type.
        dataset: str, dataset.

    Return
        - Dict of selected hyperparameters for the inputs.
    """
    lgb = {}
    lgb['adult'] = {'n_estimators': 100, 'num_leaves': 31, 'max_depth': -1}
    lgb['bank_marketing'] = {'n_estimators': 50, 'num_leaves': 31, 'max_depth': -1}
    lgb['bean'] = {'n_estimators': 25, 'num_leaves': 15, 'max_depth': -1}
    lgb['compas'] = {'n_estimators': 25, 'num_leaves': 91, 'max_depth': -1}
    lgb['concrete'] = {'n_estimators': 200, 'num_leaves': 15, 'max_depth': -1}
    lgb['credit_card'] = {'n_estimators': 50, 'num_leaves': 15, 'max_depth': -1}
    lgb['diabetes'] = {'n_estimators': 200, 'num_leaves': 31, 'max_depth': -1}
    lgb['energy'] = {'n_estimators': 200, 'num_leaves': 15, 'max_depth': -1}
    lgb['flight_delays'] = {'n_estimators': 100, 'num_leaves': 91, 'max_depth': -1}
    lgb['german_credit'] = {'n_estimators': 25, 'num_leaves': 15, 'max_depth': -1}
    lgb['htru2'] = {'n_estimators': 100, 'num_leaves': 15, 'max_depth': -1}
    lgb['life'] = {'n_estimators': 200, 'num_leaves': 61, 'max_depth': -1}
    lgb['naval'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['no_show'] = {'n_estimators': 50, 'num_leaves': 61, 'max_depth': -1}
    lgb['obesity'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['power'] = {'n_estimators': 200, 'num_leaves': 61, 'max_depth': -1}
    lgb['protein'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['spambase'] = {'n_estimators': 200, 'num_leaves': 31, 'max_depth': -1}
    lgb['surgical'] = {'n_estimators': 200, 'num_leaves': 15, 'max_depth': -1}
    lgb['twitter'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['vaccine'] = {'n_estimators': 100, 'num_leaves': 15, 'max_depth': -1}
    lgb['wine'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}

    sgb = {}
    sgb['adult'] = {'max_iter': 200, 'max_leaf_nodes': 15, 'max_bins': 250, 'max_depth': None}
    sgb['bank_marketing'] = {'max_iter': 50, 'max_leaf_nodes': 31, 'max_bins': 100, 'max_depth': None}
    sgb['bean'] = {'max_iter': 25, 'max_leaf_nodes': 15, 'max_bins': 50, 'max_depth': None}
    sgb['compas'] = {'max_iter': 50, 'max_leaf_nodes': 15, 'max_bins': 50, 'max_depth': None}
    sgb['concrete'] = {'max_iter': 200, 'max_leaf_nodes': 15, 'max_bins': 250, 'max_depth': None}
    sgb['credit_card'] = {'max_iter': 25, 'max_leaf_nodes': 15, 'max_bins': 100, 'max_depth': None}
    sgb['diabetes'] = {'max_iter': 200, 'max_leaf_nodes': 31, 'max_bins': 100, 'max_depth': None}
    sgb['energy'] = {'max_iter': 200, 'max_leaf_nodes': 15, 'max_bins': 50, 'max_depth': None}
    sgb['flight_delays'] = {'max_iter': 200, 'max_leaf_nodes': 61, 'max_bins': 250, 'max_depth': None}
    sgb['german_credit'] = {'max_iter': 25, 'max_leaf_nodes': 15, 'max_bins': 100, 'max_depth': None}
    sgb['htru2'] = {'max_iter': 50, 'max_leaf_nodes': 15, 'max_bins': 50, 'max_depth': None}
    sgb['life'] = {'max_iter': 200, 'max_leaf_nodes': 31, 'max_bins': 250, 'max_depth': None}
    sgb['naval'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 250, 'max_depth': None}
    sgb['no_show'] = {'max_iter': 50, 'max_leaf_nodes': 61, 'max_bins': 100, 'max_depth': None}
    sgb['obesity'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 250, 'max_depth': None}
    sgb['power'] = {'max_iter': 200, 'max_leaf_nodes': 61, 'max_bins': 250, 'max_depth': None}
    sgb['protein'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 250, 'max_depth': None}
    sgb['spambase'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 250, 'max_depth': None}
    sgb['surgical'] = {'max_iter': 100, 'max_leaf_nodes': 31, 'max_bins': 250, 'max_depth': None}
    sgb['twitter'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 250, 'max_depth': None}
    sgb['vaccine'] = {'max_iter': 100, 'max_leaf_nodes': 15, 'max_bins': 50, 'max_depth': None}
    sgb['wine'] = {'max_iter': 200, 'max_leaf_nodes': 91, 'max_bins': 100, 'max_depth': None}

    cb = {}
    cb['adult'] = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.6}
    cb['bank_marketing'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1}
    cb['bean'] = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.3}
    cb['compas'] = {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.3}
    cb['concrete'] = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.3}
    cb['credit_card'] = {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1}
    cb['diabetes'] = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.3}
    cb['energy'] = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.9}
    cb['flight_delays'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.3}
    cb['german_credit'] = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    cb['htru2'] = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.3}
    cb['life'] = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.3}
    cb['naval'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.6}
    cb['no_show'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.3}
    cb['obesity'] = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.6}
    cb['power'] = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.6}
    cb['protein'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.3}
    cb['spambase'] = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.3}
    cb['surgical'] = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}
    cb['twitter'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.6}
    cb['vaccine'] = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}
    cb['wine'] = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.3}

    xgb = {}
    xgb['adult'] = {'n_estimators': 200, 'max_depth': 3}
    xgb['bank_marketing'] = {'n_estimators': 100, 'max_depth': 4}
    xgb['bean'] = {'n_estimators': 25, 'max_depth': 5}
    xgb['compas'] = {'n_estimators': 50, 'max_depth': 3}
    xgb['concrete'] = {'n_estimators': 200, 'max_depth': 4}
    xgb['credit_card'] = {'n_estimators': 10, 'max_depth': 3}
    xgb['diabetes'] = {'n_estimators': 200, 'max_depth': 3}
    xgb['energy'] = {'n_estimators': 200, 'max_depth': 5}
    xgb['flight_delays'] = {'n_estimators': 200, 'max_depth': 7}
    xgb['german_credit'] = {'n_estimators': 10, 'max_depth': 4}
    xgb['htru2'] = {'n_estimators': 100, 'max_depth': 2}
    xgb['life'] = {'n_estimators': 200, 'max_depth': 5}
    xgb['naval'] = {'n_estimators': 100, 'max_depth': 7}
    xgb['no_show'] = {'n_estimators': 100, 'max_depth': 5}
    xgb['obesity'] = {'n_estimators': 200, 'max_depth': 7}
    xgb['power'] = {'n_estimators': 200, 'max_depth': 7}
    xgb['protein'] = {'n_estimators': 200, 'max_depth': 7}
    xgb['spambase'] = {'n_estimators': 200, 'max_depth': 4}
    xgb['surgical'] = {'n_estimators': 50, 'max_depth': 5}
    xgb['twitter'] = {'n_estimators': 200, 'max_depth': 7}
    xgb['vaccine'] = {'n_estimators': 100, 'max_depth': 3}
    xgb['wine'] = {'n_estimators': 100, 'max_depth': 7}

    skrf = {}
    skrf['adult'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['bank_marketing'] = {'n_estimators': 100, 'max_depth': 7}
    skrf['bean'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['compas'] = {'n_estimators': 100, 'max_depth': 7}
    skrf['concrete'] = {'n_estimators': 100, 'max_depth': 7}
    skrf['credit_card'] = {'n_estimators': 50, 'max_depth': 7}
    skrf['diabetes'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['energy'] = {'n_estimators': 10, 'max_depth': 7}
    skrf['flight_delays'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['german_credit'] = {'n_estimators': 25, 'max_depth': 5}
    skrf['htru2'] = {'n_estimators': 50, 'max_depth': 7}
    skrf['life'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['naval'] = {'n_estimators': 100, 'max_depth': 7}
    skrf['no_show'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['obesity'] = {'n_estimators': 50, 'max_depth': 7}
    skrf['power'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['protein'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['spambase'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['surgical'] = {'n_estimators': 100, 'max_depth': 7}
    skrf['twitter'] = {'n_estimators': 50, 'max_depth': 7}
    skrf['vaccine'] = {'n_estimators': 200, 'max_depth': 7}
    skrf['wine'] = {'n_estimators': 200, 'max_depth': 5}

    hp = {}
    hp['lgb'] = lgb
    hp['sgb'] = sgb
    hp['cb'] = cb
    hp['xgb'] = xgb
    hp['skrf'] = skrf

    return hp[tree_type][dataset]


# private
class Tee(object):
    """
    Class to control where output is printed to.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()   # output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()
