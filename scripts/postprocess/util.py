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

    experiment_settings = list(product(*[args.model, args.tree_type, args.affinity,
                                         args.min_scale_pct]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):
        model, tree_type, affinity, min_scale_pct = items

        template = {'tree_type': tree_type,
                    'affinity': affinity,
                    'delta': args.delta,
                    'min_scale_pct': min_scale_pct}

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
    color['constant_334858faf273fcedf1f1a954626ea3c5'] = 'blue'
    color['kgbm_663cc48c2d48513d9b906dee97427c37'] = 'cyan'  # tf: 1.0, del: 1
    color['kgbm_ea165ac8a817022c2ac013f13f83e81d'] = 'cyan'  # tf: 1.0, del: 0
    color['kgbm_73c21532a539b7baa4be98e716a91d18'] = 'cyan'  # tf: 0.5, del: 0
    color['kgbm_2641e1b8f198bdc326366826f3c37d5e'] = 'cyan'  # tf: 0.1, del: 0
    color['knn_19f65e07ad4a0f21e2a3b3488e53c947'] = 'orange'
    color['knn_816312457bd29f7ba532915a9f39aa29'] = 'orange'
    color['ngboost'] = 'green'
    color['ngboost_c4ca4238a0b923820dcc509a6f75849b'] = 'green'
    color['pgbm'] = 'brown'
    color['pgbm_c4ca4238a0b923820dcc509a6f75849b'] = 'brown'

    line = {}
    line['constant_fd48c03eaa6f667804f917b37f89aa30'] = '-'
    line['constant_334858faf273fcedf1f1a954626ea3c5'] = '-'
    line['kgbm_663cc48c2d48513d9b906dee97427c37'] = '-'  # tf: 1.0, del: 1
    line['kgbm_ea165ac8a817022c2ac013f13f83e81d'] = '-'  # tf: 1.0, del: 0
    line['kgbm_73c21532a539b7baa4be98e716a91d18'] = '-'  # tf: 0.5, del: 0
    line['kgbm_2641e1b8f198bdc326366826f3c37d5e'] = '-'  # tf: 0.1, del: 0
    line['knn_19f65e07ad4a0f21e2a3b3488e53c947'] = '-'
    line['knn_816312457bd29f7ba532915a9f39aa29'] = '-'
    line['ngboost'] = '-'
    line['ngboost_c4ca4238a0b923820dcc509a6f75849b'] = '-'
    line['pgbm'] = '-'
    line['pgbm_c4ca4238a0b923820dcc509a6f75849b'] = '-'

    label = {}
    label['constant_fd48c03eaa6f667804f917b37f89aa30'] = 'Constant (LGB)'
    label['constant_334858faf273fcedf1f1a954626ea3c5'] = 'Constant (LGB)'
    label['kgbm_663cc48c2d48513d9b906dee97427c37'] = 'KGBM (LGB)'  # tf: 1.0, del: 1
    label['kgbm_ea165ac8a817022c2ac013f13f83e81d'] = 'KGBM (LGB)'  # tf: 1.0, del: 0
    label['kgbm_73c21532a539b7baa4be98e716a91d18'] = 'KGBM (LGB)'  # tf: 0.5, del: 0
    label['kgbm_2641e1b8f198bdc326366826f3c37d5e'] = 'KGBM (LGB)'  # tf: 0.1, del: 0
    label['knn_19f65e07ad4a0f21e2a3b3488e53c947'] = 'KNN'
    label['knn_816312457bd29f7ba532915a9f39aa29'] = 'KNN'
    label['ngboost'] = 'NGBoost'
    label['ngboost_c4ca4238a0b923820dcc509a6f75849b'] = 'NGBoost'
    label['pgbm'] = 'PGBM'
    label['pgbm_c4ca4238a0b923820dcc509a6f75849b'] = 'PGBM'

    marker = {}
    marker['constant_fd48c03eaa6f667804f917b37f89aa30'] = 'o'
    marker['constant_334858faf273fcedf1f1a954626ea3c5'] = 'o'
    marker['kgbm_663cc48c2d48513d9b906dee97427c37'] = 'd'  # tf: 1.0, del: 1
    marker['kgbm_ea165ac8a817022c2ac013f13f83e81d'] = 'd'  # tf: 1.0, del: 0
    marker['kgbm_73c21532a539b7baa4be98e716a91d18'] = 'd'  # tf: 0.5, del: 0
    marker['kgbm_2641e1b8f198bdc326366826f3c37d5e'] = 'd'  # tf: 0.1, del: 0
    marker['knn_19f65e07ad4a0f21e2a3b3488e53c947'] = '1'
    marker['knn_816312457bd29f7ba532915a9f39aa29'] = '1'
    marker['ngboost'] = '^'
    marker['ngboost_c4ca4238a0b923820dcc509a6f75849b'] = '^'
    marker['pgbm'] = '+'
    marker['pgbm_c4ca4238a0b923820dcc509a6f75849b'] = 'brown'

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
