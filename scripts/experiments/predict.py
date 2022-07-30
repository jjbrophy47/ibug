"""
Model performance.
"""
import os
import sys
import time
import resource
import argparse
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for ibug
import util
from train import get_loc_scale
from train import tune_delta


# constants
METRIC_NAMES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mdae": "MDAE",
    "marpd": "MARPD",
    "r2": "R2",
    "corr": "Correlation",
    "rms_cal": "Root-mean-squared Calibration Error",
    "ma_cal": "Mean-absolute Calibration Error",
    "miscal_area": "Miscalibration Area",
    "sharp": "Sharpness",
    "nll": "Negative-log-likelihood",
    "crps": "CRPS",
    "check": "Check Score",
    "interval": "Interval Score",
    "rms_adv_group_cal": ("Root-mean-squared Adversarial Group " "Calibration Error"),
    "ma_adv_group_cal": "Mean-absolute Adversarial Group Calibration Error",
}


def _display_adv_group_cal(adv_group_metric_dict, logger, print_group_num=3):
    """
    Display and write results to logger.
    """
    for a_group_cali_type, a_group_cali_dic in adv_group_metric_dict.items():
        num_groups = a_group_cali_dic["group_sizes"].shape[0]
        print_idxs = [int(x) for x in np.linspace(1, num_groups - 1, print_group_num)]

        logger.info("\t{}".format(METRIC_NAMES[a_group_cali_type]))

        for idx in print_idxs:
            logger.info(
                "\t\tGroup Size: {:.2f} -- Calibration Error: {:.3f}".format(
                    a_group_cali_dic["group_sizes"][idx],
                    a_group_cali_dic["adv_group_cali_mean"][idx],
                )
            )


def display_results(res_dict, logger, title=''):
    """
    Display and write results to logger.

    Reference:
    https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/uncertainty_toolbox/metrics.py
    """
    if title != '':
        logger.info(f'\n{title}:')

    logger.info(" ====================================================== \n Accuracy Metrics ")
    for acc_metric, acc_val in res_dict['accuracy'].items():
        logger.info("\t{:<13} {:.3f}".format(METRIC_NAMES[acc_metric], acc_val))

    logger.info(" ====================================================== \n Average Calibration Metrics ")
    for cali_metric, cali_val in res_dict['avg_calibration'].items():
        logger.info("\t{:<37} {:.3f}".format(METRIC_NAMES[cali_metric], cali_val))

    logger.info(" ====================================================== \n Adversarial Group Calibration Metrics ")
    _display_adv_group_cal(res_dict['adv_group_calibration'], logger)

    logger.info(" ====================================================== \n Sharpness Metrics ")
    for sharp_metric, sharp_val in res_dict['sharpness'].items():
        logger.info("\t{:}   {:.3f}".format(METRIC_NAMES[sharp_metric], sharp_val))

    logger.info(" ====================================================== \n Scoring Rule Metrics ")
    for sr_metric, sr_val in res_dict['scoring_rule'].items():
        logger.info("\t{:<25} {:.3f}".format(METRIC_NAMES[sr_metric], sr_val))
    logger.info(" ====================================================== \n")


def eval_posterior(model, X, y, min_scale, loc, scale, distribution_list,
                   fixed_params, random_state=1, logger=None, prefix=''):
    """
    Model the posterior using different parametric and non-parametric
        distributions and evaluate their predictive performance.

    Input
        model: IBUG, Fitted model.
        X: np.ndarray, 2d array of input data.
        y: np.ndarray, 1d array of ground-truth targets.
        loc: np.ndarray, 1d array of location values.
        scale: np.ndarray, 1d array of scale values.
        min_scale: float, Minimum scale value.
        distribution_list: list, List of distributions to try.
        fixed_params: str, Selects which parameters of the distribution to keep fixed.
        random_state: int, Random seed.
        logger: logging object, Log updates.
        prefix: str, Used to identify updates.

    Return
        3-tuple including a dict of CRPS and NLL results, and 2 2d
            arrays of neighbor indices and target values, shape=(no. test, k).
    """
    assert 'ibug' in str(model).lower()
    assert X.ndim == 2 and y.ndim == 1 and loc.ndim == 1 and scale.ndim == 1
    assert X.shape[0] == len(y) == len(loc) == len(scale)

    if logger:
        logger.info(f'\n[{prefix}] Computing k-nearest neighbors...')

    start = time.time()
    loc, scale, neighbor_idxs, neighbor_vals = model.pred_dist(X, return_kneighbors=True)
    if logger:
        logger.info(f'time: {time.time() - start:.3f}s')
        logger.info(f'\n[{prefix}] Evaluating distributions...')

    start = time.time()
    dist_res = {'nll': {}, 'crps': {}}

    for dist in distribution_list:

        # select fixed parameters
        if fixed_params == 'dist_fl':
            loc_copy, scale_copy = loc.copy(), None

        elif fixed_params == 'dist_fls':
            loc_copy, scale_copy = loc.copy(), scale.copy()

        else:
            loc_copy, scale_copy = None, None

        nll, crps = util.eval_dist(y=y, samples=neighbor_vals.copy(), dist=dist, nll=True,
                                   crps=True, min_scale=min_scale, random_state=random_state,
                                   loc=loc_copy, scale=scale_copy)

        if logger:
            logger.info(f'[{prefix} - {dist}] CRPS: {crps:.5f}, NLL: {nll:.5f}')

        dist_res['nll'][dist] = nll
        dist_res['crps'][dist] = crps

    if logger:
        logger.info(f'[{prefix} time: {time.time() - start:.3f}s')

    # assemble output
    result = {'performance': dist_res, 'neighbors': {'idxs': neighbor_idxs, 'y_vals': neighbor_vals}}
    return result


def calibrate_variance(scale_arr, delta, op):
    """
    Add or multiply all values in `scale_arr` by `delta`.

    Input
        scale_arr: np.ndarray, 1d array of scale values.
        delta: float, Value to add or multiply.
        op: str, Operation to perform on `scale_arr`.

    Return
        1d array of calibrated scale values.
    """
    assert scale_arr.ndim == 1
    assert op in ['add', 'mult']
    result = np.array(scale_arr)
    if op == 'add':
        result = result + delta
    else:
        result = result * delta
    return result


def experiment(args, in_dir, out_dir, logger):
    """
    Main method comparing performance of tree ensembles and svm models.
    """
    begin = time.time()  # experiment timer
    rng = np.random.default_rng(args.random_state)  # pseduo-random number generator

    # load results and models from training
    result = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
    model_val = util.load_model(model_type=args.model_type, fp=result['saved_models']['model_val'])
    model_test = util.load_model(model_type=args.model_type, fp=result['saved_models']['model_test'])

    # additional settings
    if args.model_type == 'ibug':
        model_val.set_n_jobs(n_jobs=args.n_jobs)
        model_test.set_n_jobs(n_jobs=args.n_jobs)
        model_test.set_tree_subsampling(frac=args.tree_subsample_frac, order=args.tree_subsample_order)
        model_test.set_instance_subsampling(frac=args.instance_subsample_frac)

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset, args.fold)
    val_idxs = result['data']['val_idxs']
    tune_idxs = result['data']['tune_idxs']
    X_val, y_val = X_train[val_idxs].copy(), y_train[val_idxs].copy()
    X_tune, y_tune = X_train[tune_idxs].copy(), y_train[tune_idxs].copy()

    logger.info('\nno. train: {:,}'.format(X_train.shape[0]))
    logger.info('  -> no. val.: {:,}'.format(X_val.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))
    logger.info(f"\ndelta: {result['delta']['best_delta']}, op: {result['delta']['best_op']}")

    # validation: predict
    logger.info(f'\nVALIDATION SET\n')
    start = time.time()
    loc_val, scale_val = get_loc_scale(model=model_val, model_type=args.model_type, X=X_val, y_train=y_tune)
    val_pred_time = time.time() - start
    logger.info(f'predicting...{time.time() - start:.3f}s')

    # validation: calibrate variance
    if args.tune_delta:
        assert args.model_type in ['constant', 'pgbm']
        logger.info(f'\nTuning delta for {args.out_scoring.upper()} scoring...')
        start = time.time()
        best_delta, best_op = tune_delta(loc=loc_val, scale=scale_val, y=y_val,
                                         scoring=args.out_scoring, verbose=args.verbose, logger=logger)
        tune_time_delta = time.time() - start
        logger.info(f'\nbest delta: {best_delta}, op: {best_op}')
        logger.info(f'tune time (delta): {tune_time_delta:.3f}s')
    else:
        best_delta, best_op = result['delta']['best_delta'], result['delta']['best_op']
    scale_val_delta = calibrate_variance(scale_val, delta=best_delta, op=best_op)

    # validation: evaluate performance (w/ and w/o delta)
    start = time.time()
    val_res = uct.metrics.get_all_metrics(y_pred=loc_val, y_std=scale_val, y_true=y_val, verbose=False)
    val_res_delta = uct.metrics.get_all_metrics(y_pred=loc_val, y_std=scale_val_delta, y_true=y_val, verbose=False)
    val_eval_time = time.time() - start
    logger.info(f'evaluating performance...{time.time() - start:.3f}s')

    # validation: plot intervals, miscalibration area, adversarial group calibration error
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    uct.viz.plot_intervals_ordered(y_pred=loc_val, y_std=scale_val, y_true=y_val, ax=axs[0][0])
    uct.viz.plot_calibration(y_pred=loc_val, y_std=scale_val, y_true=y_val, ax=axs[0][1])
    uct.viz.plot_adversarial_group_calibration(y_pred=loc_val, y_std=scale_val, y_true=y_val, ax=axs[0][2])
    uct.viz.plot_intervals_ordered(y_pred=loc_val, y_std=scale_val_delta, y_true=y_val, ax=axs[1][0])
    uct.viz.plot_calibration(y_pred=loc_val, y_std=scale_val_delta, y_true=y_val, ax=axs[1][1])
    uct.viz.plot_adversarial_group_calibration(y_pred=loc_val, y_std=scale_val_delta, y_true=y_val, ax=axs[1][2])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'val_pred.png'), bbox_inches='tight')

    # test: predict
    logger.info(f'\nTEST SET\n')
    start = time.time()
    loc_test, scale_test = get_loc_scale(model=model_test, model_type=args.model_type, X=X_test, y_train=y_train)
    test_pred_time = time.time() - start
    logger.info(f'predicting...{time.time() - start:.3f}s')

    # test: calibrate variance
    scale_test_delta = calibrate_variance(scale_test, delta=best_delta, op=best_op)

    # test: evaluate performance (w/ and w/o delta)
    start = time.time()
    test_res = uct.metrics.get_all_metrics(y_pred=loc_test, y_std=scale_test, y_true=y_test, verbose=False)
    test_res_delta = uct.metrics.get_all_metrics(y_pred=loc_test, y_std=scale_test_delta, y_true=y_test, verbose=False)
    test_eval_time = time.time() - start
    logger.info(f'evaluating performance...{time.time() - start:.3f}s')

    # show results
    logger.info(f'\nRESULTS')
    display_results(val_res, logger, title='Validation')
    display_results(val_res_delta, logger, title='Validation (w/ delta)')
    display_results(test_res, logger, title='Test')
    display_results(test_res_delta, logger, title='Test (w/ delta)')

    # test: plot intervals, miscalibration area, adversarial group calibration error
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    uct.viz.plot_intervals_ordered(y_pred=loc_test, y_std=scale_test, y_true=y_test, ax=axs[0][0])
    uct.viz.plot_calibration(y_pred=loc_test, y_std=scale_test, y_true=y_test, ax=axs[0][1])
    uct.viz.plot_adversarial_group_calibration(y_pred=loc_test, y_std=scale_test, y_true=y_test, ax=axs[0][2])
    uct.viz.plot_intervals_ordered(y_pred=loc_test, y_std=scale_test_delta, y_true=y_test, ax=axs[1][0])
    uct.viz.plot_calibration(y_pred=loc_test, y_std=scale_test_delta, y_true=y_test, ax=axs[1][1])
    uct.viz.plot_adversarial_group_calibration(y_pred=loc_test, y_std=scale_test_delta, y_true=y_test, ax=axs[1][2])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'test_pred.png'), bbox_inches='tight')

    # save results
    del result['data']['tune_idxs']
    del result['data']['val_idxs']
    result['predict_args'] = vars(args)
    result['preds'] = {
        'val': {'loc': loc_val, 'scale': scale_val, 'scale_delta': scale_val_delta},
        'test': {'loc': loc_test, 'scale': scale_test, 'scale_delta': scale_test_delta}
    }
    result['timing'].update({'val_pred_time': val_pred_time,
                             'val_eval_time': val_eval_time,
                             'test_pred_time': test_pred_time,
                             'test_eval_time': test_eval_time}),
    result['metrics'] = {
        'val': val_res, 'val_delta': val_res_delta,
        'test': test_res, 'test_delta': test_res_delta
    }
    result['misc'] = {
        'max_RSS': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,  # MB if OSX, GB if Linux
        'total_experiment_time': time.time() - begin
    }
    if args.tune_delta:
        result['timing']['tune_delta'] = tune_time_delta
        result['delta'] = {'best_delta': best_delta, 'best_op': best_op}
    if args.model_type == 'ibug':

        # collecting statistics
        logger.info('\n\nLEAF STATISTICS')
        start = time.time()
        result['affinity_count'] = model_test.get_affinity_stats(X_test)  # dict with 2 1d arrays of len=n_boost
        result['leaf_density'] = model_test.get_leaf_stats()  # dict with 1 1d array of len=n_boost
        logger.info(f'time: {time.time() - start:.3f}s')

        # posterior modeling
        if args.custom_out_dir in ['dist', 'dist_fl', 'dist_fls']:
            logger.info('\n\nPOSTERIOR MODELING')
            val_res = eval_posterior(model=model_val, X=X_val, y=y_val, min_scale=model_val.min_scale_,
                                     loc=loc_val, scale=scale_val_delta, distribution_list=args.distribution,
                                     fixed_params=args.custom_out_dir, random_state=args.random_state,
                                     logger=logger, prefix='VAL')
            test_res = eval_posterior(model=model_test, X=X_test, y=y_test, min_scale=model_test.min_scale_,
                                      loc=loc_test, scale=scale_test_delta, distribution_list=args.distribution,
                                      fixed_params=args.custom_out_dir, random_state=args.random_state,
                                      logger=logger, prefix='TEST')
            result['val_posterior'] = val_res['performance']
            result['test_posterior'] = test_res['performance']
            if args.fold == 1:
                result['neighbors'] = test_res['neighbors']

    # Macs show this in bytes, unix machines show this in KB
    logger.info(f"\ntotal experiment time: {result['misc']['total_experiment_time']:.3f}s")
    logger.info(f"max_rss (MB): {result['misc']['max_RSS']:.1f}")
    logger.info(f"\nresults:\n{result}")
    logger.info(f"\nsaving results and models to {os.path.join(out_dir, 'results.npy')}")

    # save results/models
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # define input directory
    train_args = vars(args).copy()
    train_args['tree_subsample_frac'] = 1.0
    train_args['tree_subsample_order'] = 'random'
    train_args['instance_subsample_frac'] = 1.0
    method_name = util.get_method_identifier(args.model_type, train_args)

    in_dir = os.path.join(args.in_dir,
                          args.custom_in_dir,
                          args.dataset,
                          args.in_scoring,
                          f'fold{args.fold}',
                          method_name)

    # define output directory
    method_name = util.get_method_identifier(args.model_type, vars(args))

    out_dir = os.path.join(args.out_dir,
                           args.custom_out_dir,
                           args.dataset,
                           args.out_scoring,
                           f'fold{args.fold}',
                           method_name)

    # create outut directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    experiment(args, in_dir, out_dir, logger)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='output/experiments/train/')
    parser.add_argument('--out_dir', type=str, default='output/experiments/predict/')
    parser.add_argument('--custom_in_dir', type=str, default='default')
    parser.add_argument('--custom_out_dir', type=str, default='default')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='concrete')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='ibug')

    # Method settings
    parser.add_argument('--gridsearch', type=int, default=1)  # affects constant, IBUG, PGBM
    parser.add_argument('--tree_type', type=str, default='lgb')  # IBUG, constant
    parser.add_argument('--tree_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--tree_subsample_order', type=str, default='random')  # IBUG
    parser.add_argument('--instance_subsample_frac', type=float, default=1.0)  # IBUG
    parser.add_argument('--affinity', type=str, default='unweighted')  # IBUG

    # Default settings
    parser.add_argument('--tune_delta', type=int, default=0)  # Constant and PGBM
    parser.add_argument('--tune_frac', type=float, default=1.0)  # ALL
    parser.add_argument('--val_frac', type=float, default=0.2)  # ALL
    parser.add_argument('--random_state', type=int, default=1)  # ALL
    parser.add_argument('--verbose', type=int, default=2)  # ALL
    parser.add_argument('--n_stopping_rounds', type=int, default=25)  # NGBoost, PGBM, IBUG, constant
    parser.add_argument('--in_scoring', type=str, default='nll')
    parser.add_argument('--out_scoring', type=str, default='nll')
    parser.add_argument('--distribution', type=str, nargs='+',
                        default=['normal', 'skewnormal', 'lognormal', 'laplace',
                                 'student_t', 'logistic', 'gumbel', 'weibull', 'kde'])

    # Extra settings
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()
    main(args)
