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
def get_results(args, exp_dir, logger=None, progress_bar=False):
    """
    Retrieve results for the multiple methods.
    """
    if logger and progress_bar:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.model_type,
                                         args.tree_type,
                                         args.affinity,
                                         args.tree_subsample_frac,
                                         args.tree_subsample_order,
                                         args.instance_subsample_frac,
                                         args.cond_mean_type]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):
        model_type, tree_type, affinity, tree_subsample_frac, tree_subsample_order,\
            instance_subsample_frac, cond_mean_type = items

        template = {'tree_type': tree_type,
                    'affinity': affinity,
                    'tree_subsample_frac': tree_subsample_frac,
                    'tree_subsample_order': tree_subsample_order,
                    'instance_subsample_frac': instance_subsample_frac,
                    'gridsearch': args.gridsearch,
                    'cond_mean_type': cond_mean_type}

        method_id = exp_util.get_method_identifier(model_type, template)
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


def plot_settings(family='serif', fontsize=11,
                  markersize=5, linewidth=None, libertine=False):
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
    if libertine:
        assert family == 'serif'
        plt.rc('text.latex', preamble=r"""\usepackage{libertine}""")
        plt.rc('text.latex', preamble=r"""
                                      \usepackage{libertine}
                                      \usepackage[libertine]{newtxmath}
                                       """)


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
