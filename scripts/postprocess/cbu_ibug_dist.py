"""
Compare probabilistic performance between CBU normal distribution
and IBUG higher-order or non-parametric distribution.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
from scipy.stats import sem

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for ibug
from experiments import util as exp_util


# constants
distributions = {
    'meps': 'weibull',
    'wine': 'kde',
}


def experiment(args, method_name1, method_name2):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    print(f'scoring: {args.scoring}')

    # load cbu and ibug predictions
    for dataset in args.dataset:
        print(f'\n{dataset}')

        m1_scores = []
        m2_scores = []
        for fold in args.fold:
            in_dir1 = os.path.join(args.in_dir,
                args.custom_in_dir1,
                dataset,
                args.scoring,
                f'fold{fold}',
                method_name1)

            in_dir2 = os.path.join(args.in_dir,
                args.custom_in_dir2,
                dataset,
                args.scoring,
                f'fold{fold}',
                method_name2)

            if not os.path.exists(in_dir1) or not os.path.exists(in_dir2):
                print('one or more input directories do not exist, skipping fold...')
                continue

            r1 = np.load(os.path.join(in_dir1, 'results.npy'), allow_pickle=True)[()]
            r2 = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]

            m1_scores.append(r1['metrics']['test_delta']['scoring_rule'][args.scoring])
            m2_scores.append(r2['test_posterior'][args.scoring][distributions[dataset]])
        
        m1_mean, m1_sem = np.mean(m1_scores), sem(m1_scores)
        m2_mean, m2_sem = np.mean(m2_scores), sem(m2_scores)
        print(f'{method_name1}: {m1_mean:.3f} +/- {m1_sem:.3f}')
        print(f'{method_name2}: {m2_mean:.3f} +/- {m2_sem:.3f}')


def main(args):

    # get method names
    train_args = vars(args).copy()
    train_args['tree_subsample_frac'] = 1.0
    train_args['tree_subsample_order'] = 'random'
    train_args['instance_subsample_frac'] = 1.0
    method_name1 = exp_util.get_method_identifier(args.model_type1, train_args)
    method_name2 = exp_util.get_method_identifier(args.model_type2, train_args)

    experiment(args, method_name1, method_name2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='output/talapas/experiments/predict/')
    parser.add_argument('--custom_in_dir1', type=str, default='default')
    parser.add_argument('--custom_in_dir2', type=str, default='dist')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+', default=['meps', 'wine'])
    parser.add_argument('--fold', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--model_type1', type=str, default='cbu')
    parser.add_argument('--model_type2', type=str, default='ibug')

    # Method settings
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, IBUG, PGBM
    parser.add_argument('--tree_type', type=str, default='lgb')  # IBUG, constant
    parser.add_argument('--tree_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--tree_subsample_order', type=str, default='random')  # IBUG
    parser.add_argument('--instance_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--affinity', type=str, default='unweighted')  # IBUG

    # Default settings
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--scoring', type=str, default='nll')

    args = parser.parse_args()
    main(args)
