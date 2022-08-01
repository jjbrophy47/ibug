from typing import Tuple, Any

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from uncertainty_toolbox.utils import assert_is_flat_same_shape, assert_is_positive

def mean_absolute_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    """Mean absolute calibration error; identical to ECE.
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        vectorized: whether to vectorize computation for observed proportions.
                    (while setting to True is faster, it has much higher memory requirements
                    and may fail to run for larger datasets).
        recal_model: an sklearn isotonoic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
    Returns:
        A single scalar which calculates the mean absolute calibration error.
    """
    assert_is_flat_same_shape(y_pred, y_std, y_true)  # Check that input arrays are flat
    assert_is_positive(y_std)  # Check that input std is positive
    assert prop_type in ["interval", "quantile"]  # Check that prop_type is one of 'interval' or 'quantile'

    # Get lists of expected and observed proportions for a range of quantiles
    exp_proportions, obs_proportions = get_proportion_lists(y_pred, y_std, y_true, num_bins, recal_model, prop_type)

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace

def get_proportion_lists(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: Any = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions
    Returns the expected proportions and observed proportion of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are vectorized for faster execution, but this function is
    not suited when there are memory constraints.
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonoic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions
    """
    assert_is_flat_same_shape(y_pred, y_std, y_true)  # Check that input arrays are flat
    assert_is_positive(y_std)  # Check that input std is positive
    assert prop_type in ["interval", "quantile"]  # Check that prop_type is one of 'interval' or 'quantile'

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)

    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    print(y_true)
    print(y_pred)
    print(y_std)
    residuals = y_pred - y_true
    print(residuals, residuals.shape)

    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    print(normalized_residuals, normalized_residuals.shape)

    norm = stats.norm(loc=0, scale=1)  # standard normal distribution

    if prop_type == "interval":
        print(0.5 - in_exp_proportions / 2.0)
        print(0.5 + in_exp_proportions / 2.0)

        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)  # left-side gaussian
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)  # right

        print(gaussian_lower_bound)
        print(gaussian_upper_bound)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        print(above_lower[4])
        print(below_upper[4])

        within_quantile = above_lower * below_upper
        print(within_quantile[4])

        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
        print(np.sum(within_quantile, axis=0))
        print(obs_proportions)

    elif prop_type == "quantile":
        gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        print(gaussian_quantile_bound)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        print(below_quantile)
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)
        print(np.sum(below_quantile, axis=0))

    return exp_proportions, obs_proportions


def main():

    rng = np.random.default_rng(1)

    y_pred = np.array([0.2, 0.2, 0.2, 0.5, 0.5])
    y_true = np.array([0.2, 0.2, 0.2, 0.5, 0.5])
    y_std = np.array([0.001, 0.001, 0.001, 0.001, 0.001])

    # y_pred = rng.normal(loc=0, scale=1.0, size=5)
    # y_std = rng.uniform(low=0, high=0.25, size=5)
    # y_true = y_pred + rng.normal(loc=0, scale=0.25, size=5)

    mace = mean_absolute_calibration_error(y_pred=y_pred, y_std=y_std, y_true=y_true, prop_type='quantile')
    print(mace)


if __name__ == '__main__':
    main()
