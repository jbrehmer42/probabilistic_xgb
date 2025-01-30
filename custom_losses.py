import numpy as np
import xgboost as xgb
from scipy.special import polygamma, digamma

a = 0.008


def _get_predictions_and_observations(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute target and predictions from xgboost inputs. Only select first target
    column since we assume a duplicated target
    """
    y = dtrain.get_label().reshape(predt.shape)[:, 0]
    return y, predt[:, 0], predt[:, 1]


def mean_var_loss(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and second derivatives of the 'mean-variance loss' at the given
    predictions. The loss is given by

    L(x_1, x_2, y) = (x_1 - y)^2 + a * (x_2 + x_1^2 - y^2)^2

    i.e. the squared error for both the first and second moment, with the second
    squared error scaled down by a constant.
    """
    y, x1, x2 = _get_predictions_and_observations(predt, dtrain)
    grad1 = x1 - y - 4 * a * x1 * (np.square(y) - (x2 + np.square(x1)))
    grad2 = 2 * a * (x2 + np.square(x1) - np.square(y))
    h11 = 1 - 4 * a * (np.square(y) - x2 - 3 * a * np.power(x1, 3))
    h22 = 2 * a * np.ones(predt.shape[0])
    return np.stack((grad1, grad2), axis=1), np.stack((h11, h22), axis=1)


def dawid_sebastiani_loss(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and second derivatives of the Dawid-Sebastiani loss at the given
    predictions. The loss is given by

    L(x_1, x_2, y) = log(x_2) + (x_1 - y)^2 / x_2

    i.e. the logarithmic loss of a standard normal distribution (up to
    irrelevant constants).
    """
    y, x1, x2 = _get_predictions_and_observations(predt, dtrain)
    grad1 = 2 * (x1 - y) / x2
    grad2 = 1 / x2 - np.square((y - x1) / x2)
    h11 = 2 / x2
    h22 = 2 * np.square(y - x1) / np.power(x2, 3) - 1 / np.square(x2)
    return np.stack((grad1, grad2), axis=1), np.stack((h11, h22), axis=1)


def dawid_sebastiani_loss_sd(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and second derivatives of the Dawid-Sebastiani loss at the given
    predictions. Treat second component of prediction as **standard deviation** here!
    The loss is given by

    L(x_1, x_2, y) = 2 log(x_2) + (x_1 - y)^2 / x_2^2

    i.e. the logarithmic loss of a standard normal distribution (up to irrelevant
    constants).
    """
    y, x1, x2 = _get_predictions_and_observations(predt, dtrain)
    grad1 = 2 * (x1 - y) / np.square(x2)
    grad2 = (2 / x2) * (1 - np.square((y - x1) / x2))
    h11 = 2 / np.square(x2)
    h22 = (2 / np.square(x2)) * (3 * np.square((y - x1) / x2) - 1)
    return np.stack((grad1, grad2), axis=1), np.stack((h11, h22), axis=1)


def poisson_loss(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and second derivatives of the Poisson loss at the given
    predictions. The loss is given by

    L(x, y) = x - y log(x)

    i.e. the logarithmic loss of a Poisson distribution (up to irrelevant
    constants)
    """
    y = dtrain.get_label().reshape(predt.shape)
    grad = 1 - y / predt
    hess = y / np.square(predt)
    return grad, hess


def negative_binomial_loss(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and second derivatives of the Negative Binomial loss at the
    given predictions. The loss is given by

    L(x_1, x_2, y) = - log( \Gamma (y + x_2) ) + log(y!) + log( \Gamma(x_2) )
                     - x_2 log(x_2) + x_2 log(x_2 + x_1) - y log(x_1) + y log(x_2 + x_1)

    i.e. the logarithmic loss of a Negative Binomial distribution (up to irrelevant
    constants). The first parameter is the mean m of the distribution, the second one
    is the positive number of successes (often called r) such that var = m + 1/r m^2.
    \Gamma represents the gamma function
    """
    y, x1, x2 = _get_predictions_and_observations(predt, dtrain)
    qq = (x2 + y) / (x2 + x1)
    grad1 = (x1 - y) / (x1 + x1**2 / x2)
    grad2 = digamma(x2) - digamma(y + x2) + np.log(1 + x1 / x2) + qq - 1
    h11 = y / x1**2 - qq / (x2 + x1)
    h22 = polygamma(1, x2) - polygamma(1, y + x2) + 1 / (x2 + x1) * (1 - qq - x1 / x2)
    return np.stack((grad1, grad2), axis=1), np.stack((h11, h22), axis=1)
