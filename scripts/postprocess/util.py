import os
import sys
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


# public
def get_results(args, exp_dir, logger=None, progress_bar=True):
    """
    Retrieve results for the multiple methods.
    """

    if logger and progress_bar:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.model, args.tree_type, args.affinity]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):
        model, tree_type, affinity = items

        template = {'tree_type': tree_type,
                    'affinity': affinity,
                    'delta': args.delta}

        method_id = exp_util.get_method_identifier(model, template)
        method_dir = os.path.join(exp_dir, method_id)

        # skip empty experiments
        if not os.path.exists(method_dir) or method_id in visited:
            continue

        # add results to result dict
        else:
            visited.add(method_id)
            result = _get_result(method_dir)
            if result is not None:
                results.append((method_id, result))

    return results


def filter_results(results, skip_list):
    """
    Remove results for methods on the skip list.
    """
    result = []

    for method, res in results:

        include = True
        for skip in skip_list:

            if skip in method:
                include = False
                break

        if include:
            result.append((method, res))

    return result


def get_plot_dicts(markers=False):
    """
    Return dict for color, line, and labels for each method.
    """
    color = {}
    color['constant_fd48c03eaa6f667804f917b37f89aa30'] = 'blue'
    color['constant_8958a70621bd987675ee4563f7381154'] = 'blue'
    color['constant_e996a4dd6a3156fa10b9dd0883ca8c4a'] = 'blue'
    color['kgbm_e1475e027e7f1c9398825dee1b291cf5'] = 'cyan'
    color['kgbm_83fdcc238907ad1712b0c2dc89de489a'] = 'cyan'
    color['kgbm_3b909679514d273bb41f91a7f01426ad'] = 'cyan'
    color['knn_fd48c03eaa6f667804f917b37f89'] = 'orange'
    color['knn_8958a70621bd987675ee4563f7381154'] = 'orange'
    color['knn_e996a4dd6a3156fa10b9dd0883ca8c4a'] = 'orange'
    color['ngboost'] = 'green'
    color['ngboost_34ec78fcc91ffb1e54cd85e4a0924332'] = 'green'
    color['ngboost_0f9f2d92c2583ef952556e1f382d0974'] = 'green'
    color['pgbm'] = 'brown'

    line = {}
    line['constant_fd48c03eaa6f667804f917b37f89aa30'] = '-'
    line['constant_8958a70621bd987675ee4563f7381154'] = '-'
    line['constant_e996a4dd6a3156fa10b9dd0883ca8c4a'] = '-'
    line['kgbm_e1475e027e7f1c9398825dee1b291cf5'] = '-'
    line['kgbm_83fdcc238907ad1712b0c2dc89de489a'] = '-'
    line['kgbm_3b909679514d273bb41f91a7f01426ad'] = '-'
    line['knn_fd48c03eaa6f667804f917b37f89'] = '-'
    line['knn_8958a70621bd987675ee4563f7381154'] = '-'
    line['knn_e996a4dd6a3156fa10b9dd0883ca8c4a'] = '-'
    line['ngboost'] = '-'
    line['ngboost_34ec78fcc91ffb1e54cd85e4a0924332'] = '-'
    line['ngboost_0f9f2d92c2583ef952556e1f382d0974'] = '-'
    line['pgbm'] = '-'

    label = {}
    label['constant_fd48c03eaa6f667804f917b37f89aa30'] = 'Constant (LGB)'
    label['constant_8958a70621bd987675ee4563f7381154'] = 'Constant (LGB)'
    label['constant_e996a4dd6a3156fa10b9dd0883ca8c4a'] = 'Constant (LGB)'
    label['kgbm_e1475e027e7f1c9398825dee1b291cf5'] = r'KGBM (LGB)'
    label['kgbm_83fdcc238907ad1712b0c2dc89de489a'] = r'KGBM (LGB)'
    label['kgbm_3b909679514d273bb41f91a7f01426ad'] = r'KGBM (LGB)'
    label['knn_fd48c03eaa6f667804f917b37f89aa30'] = 'KNN'
    label['knn_8958a70621bd987675ee4563f7381154'] = 'KNN'
    label['knn_e996a4dd6a3156fa10b9dd0883ca8c4a'] = 'KNN'
    label['ngboost'] = 'NGBoost'
    label['ngboost_34ec78fcc91ffb1e54cd85e4a0924332'] = 'NGBoost'
    label['ngboost_0f9f2d92c2583ef952556e1f382d0974'] = 'NGBoost'
    label['pgbm'] = 'PGBM'

    marker = {}
    color['constant_fd48c03eaa6f667804f917b37f89aa30'] = 'o'
    color['constant_8958a70621bd987675ee4563f7381154'] = 'o'
    color['constant_e996a4dd6a3156fa10b9dd0883ca8c4a'] = 'o'
    color['kgbm_e1475e027e7f1c9398825dee1b291cf5'] = 'd'
    color['kgbm_83fdcc238907ad1712b0c2dc89de489a'] = 'd'
    color['kgbm_3b909679514d273bb41f91a7f01426ad'] = 'd'
    color['knn_fd48c03eaa6f667804f917b37f89'] = '1'
    color['knn_8958a70621bd987675ee4563f7381154'] = '1'
    color['knn_e996a4dd6a3156fa10b9dd0883ca8c4a'] = '1'
    color['ngboost'] = '^'
    color['ngboost_34ec78fcc91ffb1e54cd85e4a0924332'] = '^'
    color['ngboost_0f9f2d92c2583ef952556e1f382d0974'] = '^'
    color['pgbm'] = '+'

    result = (color, line, label)

    if markers:
        result += (marker,)

    return result


def plot_settings(family='serif', fontsize=11,
                  markersize=5, linewidth=None):
    """
    Matplotlib settings.
    """
    plt.rc('font', family=family)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('legend', title_fontsize=fontsize)
    plt.rc('lines', markersize=markersize)
    if linewidth is not None:
        plt.rc('lines', linewidth=linewidth)


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


# private
def _get_result(in_dir):
    """
    Obtain the results for this baseline method.
    """

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        result = np.load(fp, allow_pickle=True)[()]

    return result
