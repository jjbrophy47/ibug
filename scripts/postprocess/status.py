"""
Organize results.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


def get_status(args, settings, method_id, in_dir):
    """
    Compute status dataframe for the specified method.

    Input
        args: argparse, Experiment arguments.
        settings: dict, Contains lists of experiment settings.
        method_id: str, Method identifier.
        in_dir: str, Input directory.

    Return
        pd.dataframe of statuses for all experiments.
    """
    results = []
    for setting in list(exp_util.product_dict(**settings)):

        missing_list = []
        for fold in args.fold:
            method_dir = os.path.join(in_dir,
                                      setting['dataset'],
                                      setting['metric'],
                                      f'fold{fold}',
                                      method_id)

            for fn in args.fn:
                fp = os.path.join(method_dir, fn)
                if not os.path.exists(fp):
                    missing_list.append(fold)
                    break

        setting['method'] = method_id
        setting['incomplete_folds'] = 'None!' if len(missing_list) == 0 else ','.join([str(x) for x in missing_list])

        results.append(setting)

    df = pd.DataFrame(results).sort_values(['metric', 'dataset'])
    return df


def process(args, in_dir, out_dir, logger):

    settings = {'dataset': args.dataset, 'metric': args.metric}

    # get method identifiers
    for model_type in args.model_type:
        if model_type == 'constant':
            method_args_lists = {'tree_type': args.tree_type,
                                 'gridsearch': args.gridsearch}
        
        elif model_type == 'knn':
            method_args_lists = {
                'tree_type': [t for t in args.tree_type if t in ['knn', 'lgb']],
                'gridsearch': args.gridsearch,
                'cond_mean_type': args.cond_mean_type,
            }

        elif model_type == 'ibug':
            method_args_lists = {'tree_type': [t for t in args.tree_type if t != 'knn'],
                                 'tree_subsample_frac': args.tree_subsample_frac,
                                 'tree_subsample_order': args.tree_subsample_order,
                                 'instance_subsample_frac': args.instance_subsample_frac,
                                 'affinity': args.affinity,
                                 'gridsearch': args.gridsearch,
                                 'cond_mean_type': args.cond_mean_type}
        else:
            method_args_lists = {'gridsearch': args.gridsearch}

        # get status updates for each method identifier
        for method_args in list(exp_util.product_dict(**method_args_lists)):
            method_id = exp_util.get_method_identifier(model_type, method_args)
            df = get_status(args, settings, method_id, in_dir)
            df.to_csv(os.path.join(out_dir, f'{method_id}.csv'), index=False)
            logger.info(f'\n{method_id}\n{df}')


def main(args):

    # setup I/O
    in_dir = os.path.join(args.in_dir, args.exp, args.custom_dir)
    out_dir = os.path.join(args.out_dir, args.exp, args.custom_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    exp_util.clear_dir(out_dir)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, in_dir, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/experiments/')
    parser.add_argument('--out_dir', type=str, default='output/status/experiments/')
    parser.add_argument('--exp', type=str, default='train')
    parser.add_argument('--custom_dir', type=str, default='default')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['ames', 'bike', 'california', 'communities', 'concrete',
                                 'energy', 'facebook', 'kin8nm', 'life', 'meps',
                                 'msd', 'naval', 'obesity', 'news', 'power', 'protein',
                                 'star', 'superconductor', 'synthetic', 'wave',
                                 'wine', 'yacht'])
    parser.add_argument('--metric', type=str, nargs='+', default=['nll', 'crps'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))

    # Method identifiers
    parser.add_argument('--model_type', type=str, nargs='+', default=['knn', 'ngboost', 'pgbm', 'ibug', 'cbu', 'bart'])
    parser.add_argument('--tree_type', type=str, nargs='+', default=['lgb', 'xgb', 'cb', 'knn'])
    parser.add_argument('--tree_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tree_subsample_order', type=str, nargs='+', default=['random'])
    parser.add_argument('--instance_subsample_frac', type=float, nargs='+', default=[1.0])
    parser.add_argument('--affinity', type=str, nargs='+', default=['unweighted'])
    parser.add_argument('--gridsearch', type=int, nargs='+', default=[1])
    parser.add_argument('--cond_mean_type', type=str, nargs='+', default=['base', 'neighbors'])

    # Additional settings
    parser.add_argument('--fn', type=str, nargs='+', default=['results.npy'])

    args = parser.parse_args()
    main(args)
