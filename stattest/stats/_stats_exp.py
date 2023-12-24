from scipy.stats import norm
import numpy as np
import numpy.typing as npt
from statistics import NormalDist


def eptest_exp(x):
    """
    Perform the Epps and Pulley test for normality.

    The Epps and Pulley test tests the null hypothesis that the
    data was drawn from an exponential distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.

    Returns
    -------
    statistic : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.
    """

    n = len(x)
    if n < 3:
        raise ValueError("Data must be at least length 3.")

    x_avg = sum(x) / n

    for i in range(n):
        x[i] = np.exp(-x[i] / x_avg)


    EP = ((48 * n) ** (0.5)) * ((sum(x) / n) - 0.5)

    pv_right = 1 - norm.cdf(EP)
    pv_left = norm.cdf(EP)
    pv_common = min(pv_right, pv_left) * 2

    return EP, (pv_common, pv_left, pv_right)


xo = np.random.exponential(scale=1, size=1000)
