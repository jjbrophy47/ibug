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

    experiment_settings = list(product(*[args.model, args.tree_type, args.affinity, args.scale_bias]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):
        model, tree_type, affinity, scale_bias = items

        template = {'tree_type': tree_type,
                    'affinity': affinity,
                    'scale_bias': scale_bias}

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
    color['constant'] = 'blue'
    color['kgbm_6bb0cf1f98de921b0e92b3360b9a259b'] = 'cyan'
    color['knn_fd48c03eaa6f667804f917b37f89'] = 'orange'
    color['ngboost'] = 'green'
    color['pgbm'] = 'brown'

    line = {}
    line['constant'] = '-'
    line['kgbm_6bb0cf1f98de921b0e92b3360b9a259b'] = '-'
    line['knn_fd48c03eaa6f667804f917b37f89'] = '-'
    line['ngboost'] = '-'
    line['pgbm'] = '-'

    label = {}
    label['constant'] = 'Constant (LGB)'
    label['kgbm_6bb0cf1f98de921b0e92b3360b9a259b'] = r'KGBM (LGB)'
    label['knn_fd48c03eaa6f667804f917b37f89aa30'] = 'KNN'
    label['ngboost'] = 'NGBoost'
    label['pgbm'] = 'PGBM'

    marker = {}
    color['constant'] = 'o'
    color['kgbm_6bb0cf1f98de921b0e92b3360b9a259b'] = 'd'
    color['knn_fd48c03eaa6f667804f917b37f89'] = '1'
    color['ngboost'] = '^'
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
