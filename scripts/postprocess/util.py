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

    results = []

    for method in tqdm(args.method, disable=not progress_bar):

        method_dir = os.path.join(exp_dir, method)

        # skip empty experiments
        if not os.path.exists(method_dir):
            continue

        # add results to result dict
        else:
            result = _get_result(method_dir)

            if result is not None:
                results.append((method, result))

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
    color['random'] = 'blue'
    color['vog_asc'] = 'purple'
    color['vog_desc'] = 'green'

    line = {}
    line['random'] = '-'
    line['vog_asc'] = '-'
    line['vog_desc'] = '-'

    label = {}
    label['random'] = 'Random'
    label['vog_asc'] = 'VOG (asc.)'
    label['vog_desc'] = 'VOG (desc.)'

    marker = {}
    marker['random_'] = 'o'
    marker['vog_asc'] = '^'
    marker['vog_desc'] = 'd'

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


def get_method_color(method):
    """
    Return color given the method name.
    """
    color = {}
    color['Random'] = 'blue'
    color['Target'] = 'cyan'
    color['Minority'] = 'cyan'
    color['Loss'] = 'yellow'
    color['BoostIn'] = 'orange'
    color['LeafInfSP'] = 'brown'
    color['TREX'] = 'green'
    color['TreeSim'] = 'mediumseagreen'
    color['InputSim'] = 'gray'
    color['LOO'] = 'red'
    color['SubSample'] = 'rebeccapurple'
    color['LeafInfluence'] = 'brown'
    color['LeafRefit'] = 'gray'

    assert method in color, f'{method} not in color dict'
    return color[method]


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
