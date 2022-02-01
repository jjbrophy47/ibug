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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats

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


def get_data(data_dir, dataset, fold=1, feature=False):
    """
    Return train and test data for the specified dataset.
    """
    datasets = ['ames', 'bike', 'california',
                'communities', 'concrete', 'energy',
                'facebook', 'heart', 'kin8nm', 'life', 'meps',
                'msd', 'naval', 'news', 'obesity',
                'power', 'protein', 'star', 'superconductor',
                'synthetic', 'wave', 'wine', 'yacht']
    assert dataset in datasets

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'), allow_pickle=True)[()]

    if dataset != 'heart':
        data = data[fold]

    X_train, y_train = data['X_train'].astype(dtype_t), data['y_train'].astype(dtype_t)
    X_test, y_test = data['X_test'].astype(dtype_t), data['y_test'].astype(dtype_t)
    feature_names = data['feature']
    objective = 'regression'

    if feature:
        result = X_train, X_test, y_train, y_test, feature_names, objective
    else:
        result = X_train, X_test, y_train, y_test, objective

    return result


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
        result += (np.mean([-stats.norm.logpdf(y[i], loc=loc[i], scale=scale[i]) for i in range(len(y))]),)
    if crps:
        result += (np.mean([ps.crps_gaussian(y[i], mu=loc[i], sig=scale[i]) for i in range(len(y))]),)

    if len(result) == 1:
        result = result[0]

    return result


def eval_dist(y, samples, dist='normal', nll=True, crps=False,
              min_scale=None, random_state=1, rng=None,
              loc=None, scale=None):
    """
    Evaluate each predicted normal distribution.

    Input
        y: np.ndarray, 1d array of targets, shape=(no. instances,).
        X: np.ndarray, 2d array of samples, shape=(no. instances, no. neighbors).
        dist: str, distribution to model.
        nll: bool, If True, return the avg. neg. log likelihood.
        crps: bool, If True, return the avg. CRPS score.
        random_state: int, Random seed to enhance reproducibility.
        loc: np.ndarray, 1d array of location values, shape=(no. instances,).
        scale: np.ndarray, 1d array of scale values, shape=(no. instances,).

    Return
        Tuple of scores.
    """
    assert nll or crps
    assert y.ndim == 1
    assert samples.ndim == 2
    assert y.shape[0] == samples.shape[0]
    if loc is not None:
        assert loc.shape == y.shape
    if scale is not None:
        assert loc is not None
        assert scale.shape == loc.shape

    # pseudo-random number generator
    if rng is None:
        rng = np.random.default_rng(random_state)

    # get distribution
    if dist == 'normal':
        D = stats.norm
    elif dist == 'skewnormal':
        D = stats.skewnorm
    elif dist == 'lognormal':
        D = stats.lognorm
    elif dist == 'laplace':
        D = stats.laplace
    elif dist == 'student_t':
        D = stats.t
    elif dist == 'logistic':
        D = stats.logistic
    elif dist == 'gumbel':
        D = stats.gumbel_r
    elif dist == 'weibull':
        D = stats.weibull_min
    elif dist == 'kde':
        D = stats.gaussian_kde
    else:
        raise ValueError(f'Distribution {dist} unknown!')

    # evaluate output samples
    nll_list = []
    crps_list = []

    for i in range(y.shape[0]):

        # handle extremely small scale value
        if min_scale is not None and np.std(samples[i]) < min_scale:
            noise = rng.normal(loc=0.0, scale=min_scale, size=samples.shape[1])
            samples[i] += noise

        # fit distribution
        if dist == 'kde':
            D_obj = D(samples[i], bw_method='scott')
        else:
            # keep location and scale fixed
            if loc is not None and scale is not None:
                try:  # fit shape parameter
                    if dist in ['skewnormal', 'lognormal', 'student_t', 'weibull']:
                        params = D.fit(samples[i], floc=loc[i], fscale=scale[i])
                    else:
                        params = (loc[i], scale[i])
                except:
                    print('Failed using fixed loc. and scale, trying w/o fixed...')
                    params = D.fit(samples[i])

            # keep location fixed
            elif loc is not None:
                try:  # fit scale and shape parameters
                    if dist in ['skewnormal', 'lognormal', 'student_t', 'weibull']:
                        params = D.fit(samples[i], floc=loc[i])
                    elif dist in ['normal', 'laplce', 'logistic', 'gumbel']:  # fit scale parameter
                        params = D.fit(samples[i], floc=loc[i])
                    else:
                        params = (loc[i], scale[i])
                except:
                    print('Failed using fixed loc. and scale, trying w/o fixed...')
                    params = D.fit(samples[i])
            else:
                params = D.fit(samples[i])

            # fixes odd error where scale is negative
            if dist == 'logistic' and params[1] < 0:
                params = (params[0], np.abs(params[1]))

            D_obj = D(*params)

        # evaluate
        if nll:
            if dist == 'kde':
                nll_list.append(-D_obj.logpdf(x=y[i])[0])
            else:
                nll_list.append(-D_obj.logpdf(x=y[i]))

        if crps:
            if dist == 'kde':
                samples_crps = D_obj.resample(size=1000, seed=random_state)[0]
            else:
                samples_crps = D_obj.rvs(size=1000, random_state=random_state)
            crps_list.append(ps.crps_ensemble(y[i], samples_crps))

    # assemble output
    result = ()
    if nll:
        result += (np.mean(nll_list),)
    if crps:
        result += (np.mean(crps_list),)
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

    s = ''.join(k + str(v) for k, v in sorted(d.items()))  # alphabetical key sort

    result = hashlib.md5(s.encode('utf-8')).hexdigest() if s != '' else ''

    return result


def get_method_identifier(model, exp_params):
    """
    Return model name concatenated with a hash
    unqiuely identifying this model with the chosen settings.

    Input
        model: str, model name.
        exp_params: dict, method parameters.
    """
    settings = {}

    if model == 'constant':
        settings['tree_type'] = exp_params['tree_type']

    elif model == 'kgbm':
        settings['tree_frac'] = exp_params['tree_frac']
        settings['affinity'] = exp_params['affinity']
        settings['tree_type'] = exp_params['tree_type']

    elif model == 'knn':
        settings['min_scale_pct'] = exp_params['min_scale_pct']

    if exp_params['delta']:
        settings['delta'] = exp_params['delta']

    if exp_params['gridsearch'] and model in ['constant', 'kgbm', 'pgbm']:
        settings['gridsearch'] = exp_params['gridsearch']

    if len(settings) > 0:
        hash_str = dict_to_hash(settings)
        method_name = f'{model}_{hash_str}'
    else:
        method_name = model

    return method_name


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
