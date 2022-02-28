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
def get_results(args, exp_dir, logger=None, progress_bar=False, remove_neighbors=False):
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
                                         args.instance_subsample_frac]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):
        model_type, tree_type, affinity, tree_subsample_frac, tree_subsample_order, instance_subsample_frac = items

        template = {'tree_type': tree_type,
                    'affinity': affinity,
                    'tree_subsample_frac': tree_subsample_frac,
                    'tree_subsample_order': tree_subsample_order,
                    'instance_subsample_frac': tree_subsample_frac,
                    'gridsearch': args.gridsearch}

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
                if remove_neighbors:
                    if 'neighbor_idxs' in result:
                        del result['neighbor_idxs']
                    if 'neighbor_vals' in result:
                        del result['neighbor_vals']
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
    color['constant_7dce9f275106683dabaf1f08543e1590'] = 'blue'  # del=0, gs=0, tt=lgb
    color['constant_aca6c15e53adb4da131f8e19c52d4338'] = 'blue'  # del=1, gs=0, tt=lgb
    color['constant_2f29848944ee19df0034c5b92858317b'] = 'blue'  # del=0, gs=1, tt=lgb
    color['constant_27ebb0dfd1660c84cea9593d646a7175'] = 'blue'  # del=1, gs=1, tt=lgb
    color['kgbm_a3c550c0f0bbbbb3b1cfb318075c64b8'] = 'cyan'  # L other=def.
    color['kgbm_9e6706ce251feef3fe80af643dba21cd'] = 'cyan'  # LD other=def.
    color['kgbm_55182dc99cac0d1db2b9406a6c35adcb'] = 'cyan'  # LG other=def.
    color['kgbm_7009ad3af77a41a50b20b4a5616a97a2'] = 'cyan'  # LGD other=def.
    color['kgbm_257690bac9ad94b554382703ef083b7c'] = 'cyan'  # X other=def.
    color['kgbm_d043ef8d524cc7e4c54f6ec0a951e968'] = 'cyan'  # XD other=def.
    color['kgbm_6b60eb609547c1d00acf5647cc3ac706'] = 'cyan'  # XG other=def.
    color['kgbm_8437b091947fba51cc8ed2bf9c1d52cc'] = 'cyan'  # XGD other=def.
    color['kgbm_10d39697496e3417832ee82591e665df'] = 'cyan'  # C other=def.
    color['kgbm_eeb56073a3edce5d6308bc4059a11ddd'] = 'cyan'  # CD other=def.
    color['kgbm_bab933a5e5b09c9da6e49a6e064af7cf'] = 'cyan'  # CG other=def.
    color['kgbm_39005ff08c2fc46ba3d0dd1db1085ad4'] = 'cyan'  # CGD other=def.
    color['knn_eb1618b43c0dbc5c6ae7216686c1c398'] = 'orange'  # del=0, ms=def.
    color['knn_fbf8f87f41a9d607aa7c4993190b1b13'] = 'orange'  # del=1, ms=def.
    color['ngboost'] = 'green'  # del=0
    color['ngboost_acff7ecceeb4c799d280f7252a2b3585'] = 'green'  # del=1
    color['pgbm'] = 'brown'  # del=0, gs=0
    color['pgbm_acff7ecceeb4c799d280f7252a2b3585'] = 'brown'  # del=1, gs=0
    color['pgbm_5e13b5ea212546260ba205c54e1d9559'] = 'brown'  # del=0, gs=1
    color['pgbm_992d84ddec6882788a2969fca8876a52'] = 'brown'  # del=1, gs=1

    line = {}
    line['constant_7dce9f275106683dabaf1f08543e1590'] = '-'  # del=0, gs=0, tt=lgb
    line['constant_aca6c15e53adb4da131f8e19c52d4338'] = '-'  # del=1, gs=0, tt=lgb
    line['constant_2f29848944ee19df0034c5b92858317b'] = '-'  # del=0, gs=1, tt=lgb
    line['constant_27ebb0dfd1660c84cea9593d646a7175'] = '-'  # del=1, gs=1, tt=lgb
    line['kgbm_a3c550c0f0bbbbb3b1cfb318075c64b8'] = '-'  # del=0, gs=0, other=def.
    line['kgbm_9e6706ce251feef3fe80af643dba21cd'] = '-'  # del=1, gs=0, other=def.
    line['kgbm_55182dc99cac0d1db2b9406a6c35adcb'] = '-'  # del=0, gs=1, other=def.
    line['kgbm_7009ad3af77a41a50b20b4a5616a97a2'] = '-'  # del=1, gs=1, other=def.
    line['kgbm_257690bac9ad94b554382703ef083b7c'] = '-'  # X other=def.
    line['kgbm_d043ef8d524cc7e4c54f6ec0a951e968'] = '-'  # XD other=def.
    line['kgbm_6b60eb609547c1d00acf5647cc3ac706'] = '-'  # XG other=def.
    line['kgbm_8437b091947fba51cc8ed2bf9c1d52cc'] = '-'  # XGD other=def.
    line['kgbm_10d39697496e3417832ee82591e665df'] = '-'  # C other=def.
    line['kgbm_eeb56073a3edce5d6308bc4059a11ddd'] = '-'  # CD other=def.
    line['kgbm_bab933a5e5b09c9da6e49a6e064af7cf'] = '-'  # CG other=def.
    line['kgbm_39005ff08c2fc46ba3d0dd1db1085ad4'] = '-'  # CGD other=def.
    line['knn_eb1618b43c0dbc5c6ae7216686c1c398'] = '-'  # del=0, ms=def.
    line['knn_fbf8f87f41a9d607aa7c4993190b1b13'] = '-'  # del=1, ms=def.
    line['ngboost'] = '-'  # del=0
    line['ngboost_acff7ecceeb4c799d280f7252a2b3585'] = '-'  # del=1
    line['pgbm'] = '-'  # del=0, gs=0
    line['pgbm_acff7ecceeb4c799d280f7252a2b3585'] = '-'  # del=1, gs=0
    line['pgbm_5e13b5ea212546260ba205c54e1d9559'] = '-'  # del=0, gs=1
    line['pgbm_992d84ddec6882788a2969fca8876a52'] = '-'  # del=1, gs=1

    label = {}
    label['constant_7dce9f275106683dabaf1f08543e1590'] = 'Constant-L'  # del=0, gs=0, tt=lgb
    label['constant_aca6c15e53adb4da131f8e19c52d4338'] = 'Constant-LD'  # del=1, gs=0, tt=lgb
    label['constant_2f29848944ee19df0034c5b92858317b'] = 'Constant-LG'  # del=0, gs=1, tt=lgb
    label['constant_27ebb0dfd1660c84cea9593d646a7175'] = 'Constant-LDG'  # del=1, gs=1, tt=lgb
    label['kgbm_a3c550c0f0bbbbb3b1cfb318075c64b8'] = 'KGBM-L'  # del=0, gs=0, other=def.
    label['kgbm_9e6706ce251feef3fe80af643dba21cd'] = 'KGBM-LD'  # del=1, gs=0, other=def.
    label['kgbm_55182dc99cac0d1db2b9406a6c35adcb'] = 'KGBM-LG'  # del=0, gs=1, other=def.
    label['kgbm_7009ad3af77a41a50b20b4a5616a97a2'] = 'KGBM-LDG'  # del=1, gs=1, other=def.
    label['kgbm_257690bac9ad94b554382703ef083b7c'] = 'KGBM-X'  # X other=def.
    label['kgbm_d043ef8d524cc7e4c54f6ec0a951e968'] = 'KGBM-XD'  # XD other=def.
    label['kgbm_6b60eb609547c1d00acf5647cc3ac706'] = 'KGBM-XG'  # XG other=def.
    label['kgbm_8437b091947fba51cc8ed2bf9c1d52cc'] = 'KGBM-XDG'  # XGD other=def.
    label['kgbm_10d39697496e3417832ee82591e665df'] = 'KGBM-C'  # C other=def.
    label['kgbm_eeb56073a3edce5d6308bc4059a11ddd'] = 'KGBM-CD'  # CD other=def.
    label['kgbm_bab933a5e5b09c9da6e49a6e064af7cf'] = 'KGBM-CG'  # CG other=def.
    label['kgbm_39005ff08c2fc46ba3d0dd1db1085ad4'] = 'KGBM-CDG'  # CGD other=def.
    label['knn_eb1618b43c0dbc5c6ae7216686c1c398'] = 'KNN'  # del=0, ms=def.
    label['knn_fbf8f87f41a9d607aa7c4993190b1b13'] = 'KNN-D'  # del=1, ms=def.
    label['ngboost'] = 'NGBoost'  # del=0
    label['ngboost_acff7ecceeb4c799d280f7252a2b3585'] = 'NGBoost-D'  # del=1
    label['pgbm'] = 'PGBM'  # del=0, gs=0
    label['pgbm_acff7ecceeb4c799d280f7252a2b3585'] = 'PGBM-D'  # del=1, gs=0
    label['pgbm_5e13b5ea212546260ba205c54e1d9559'] = 'PGBM-G'  # del=0, gs=1
    label['pgbm_992d84ddec6882788a2969fca8876a52'] = 'PGBM-DG'  # del=1, gs=1

    marker = {}
    marker['constant_7dce9f275106683dabaf1f08543e1590'] = 'o'  # del=0, gs=0, tt=lgb
    marker['constant_aca6c15e53adb4da131f8e19c52d4338'] = 'o'  # del=1, gs=0, tt=lgb
    marker['constant_2f29848944ee19df0034c5b92858317b'] = 'o'  # del=0, gs=1, tt=lgb
    marker['constant_27ebb0dfd1660c84cea9593d646a7175'] = 'o'  # del=1, gs=1, tt=lgb
    marker['kgbm_a3c550c0f0bbbbb3b1cfb318075c64b8'] = 'd'  # del=0, gs=0, other=def.
    marker['kgbm_9e6706ce251feef3fe80af643dba21cd'] = 'd'  # del=1, gs=0, other=def.
    marker['kgbm_55182dc99cac0d1db2b9406a6c35adcb'] = 'd'  # del=0, gs=1, other=def.
    marker['kgbm_7009ad3af77a41a50b20b4a5616a97a2'] = 'd'  # del=1, gs=1, other=def.
    marker['kgbm_257690bac9ad94b554382703ef083b7c'] = 'd'  # X other=def.
    marker['kgbm_d043ef8d524cc7e4c54f6ec0a951e968'] = 'd'  # XD other=def.
    marker['kgbm_6b60eb609547c1d00acf5647cc3ac706'] = 'd'  # XG other=def.
    marker['kgbm_8437b091947fba51cc8ed2bf9c1d52cc'] = 'd'  # XGD other=def.
    marker['kgbm_10d39697496e3417832ee82591e665df'] = 'd'  # C other=def.
    marker['kgbm_eeb56073a3edce5d6308bc4059a11ddd'] = 'd'  # CD other=def.
    marker['kgbm_bab933a5e5b09c9da6e49a6e064af7cf'] = 'd'  # CG other=def.
    marker['kgbm_39005ff08c2fc46ba3d0dd1db1085ad4'] = 'd'  # CGD other=def.
    marker['knn_eb1618b43c0dbc5c6ae7216686c1c398'] = '1'  # del=0, ms=def.
    marker['knn_fbf8f87f41a9d607aa7c4993190b1b13'] = '1'  # del=1, ms=def.
    marker['ngboost'] = '^'  # del=0
    marker['ngboost_acff7ecceeb4c799d280f7252a2b3585'] = '^'  # del=1
    marker['pgbm'] = '+'  # del=0, gs=0
    marker['pgbm_acff7ecceeb4c799d280f7252a2b3585'] = '+'  # del=1, gs=0
    marker['pgbm_5e13b5ea212546260ba205c54e1d9559'] = '+'  # del=0, gs=1
    marker['pgbm_992d84ddec6882788a2969fca8876a52'] = '+'  # del=1, gs=1

    result = (color, line, label)

    if markers:
        result += (marker,)

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
