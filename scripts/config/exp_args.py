"""
Common commandline arguments.
"""
import configargparse


def get_general_args(cmd=None):
    """
    Create an ArgParser object and add general arguments to it.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--data_dir', type=str, default='data/')
    cmd.add('--dataset', type=str, default='surgical')
    cmd.add('--tree_type', type=str, default='lgb')
    return cmd


def get_explainer_args(cmd=None):
    """
    Add arguments used by the explainers.

    Input
        cmd: ArgParser, object to add commandline arguments to.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--method', type=str, default='random')
    cmd.add('--leaf_inf_update_set', type=int, default=-1)  # LeafInfluence
    cmd.add('--leaf_inf_atol', type=int, default=1e-5)  # LeafInfluence
    cmd.add('--input_sim_measure', type=str, default='euclidean')  # InputSim
    cmd.add('--tree_sim_measure', type=str, default='dot_prod')  # TreeSim
    cmd.add('--tree_kernel', type=str, default='lpw')  # Trex, TreeSim
    cmd.add('--trex_target', type=str, default='actual')  # Trex
    cmd.add('--trex_lmbd', type=float, default=0.003)  # Trex
    cmd.add('--trex_n_epoch', type=str, default=3000)  # Trex
    cmd.add('--dshap_trunc_frac', type=float, default=0.25)  # DShap
    cmd.add('--dshap_check_every', type=int, default=100)  # DShap
    cmd.add('--subsample_sub_frac', type=float, default=0.7)  # SubSample
    cmd.add('--subsample_n_iter', type=int, default=4000)  # SubSample
    cmd.add('--n_jobs', type=int, default=-1)  # LOO, DShap, SubSample, LeafInf, LeafRefit
    cmd.add('--random_state', type=int, default=1)  # DShap, LOO, Minority, Random, SubSample, Target, Trex
    return cmd


# experiments

def get_vog_args():
    """
    Add arguments specific to the "VOG" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/experiments/vog/')
    return cmd


def get_compression_args():
    """
    Add arguments specific to the "VOG" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/experiments/compression/')
    return cmd
