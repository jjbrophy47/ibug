"""
Commandline arguments for status scripts.
"""
from . import exp_args


def get_experiments_args():
    """
    Add arguments specific to the "Experiments" status script.

    Return ArgParser object.
    """
    cmd = exp_args.get_general_args()
    cmd = exp_args.get_explainer_args(cmd)

    cmd.add('--method_list', type=str, nargs='+', default=['random', 'target', 'leaf_sim', 'boostin',
                                                           'leaf_infSP', 'trex', 'subsample', 'loo', 'leaf_inf',
                                                           'leaf_refit', 'boostinW1', 'boostinW2'])

    cmd.add('--exp', type=str, default='influence/')
    cmd.add('--in_dir', type=str, default='temp_influence/')
    cmd.add('--out_dir', type=str, default='output/status/')

    # single-test experiment args
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--edit_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--targeted_edit_frac', type=float, nargs='+',
            default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])

    # multi-test experiment args
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--remove_frac_set', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--edit_frac_set', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--poison_frac_set', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

    # noise only
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    cmd.add('--agg_type', type=int, nargs='+', default=['self', 'test_sum'])

    # other
    cmd.add('--status_type', type=str, default='time')

    return cmd
