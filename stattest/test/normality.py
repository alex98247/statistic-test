from stattest.test.AbstractTest import AbstractTest
from stattest.core import norm
import numpy as np
import scipy.stats as scipy_stats

cache = {}


class AbstractNormalityTest(AbstractTest):

    def __init__(self):
        self.mean = 0
        self.var = 1

    def calculate_critical_value(self, rvs_size, alpha, count=500_000):
        global cache
        if self.code() not in cache.keys():
            cache[self.code()] = {}
        if rvs_size in cache[self.code()].keys():
            return cache[self.code()][rvs_size]

        result = np.zeros(count)

        for i in range(count):
            x = self.generate(size=rvs_size, mean=self.mean, var=self.var)
            result[i] = self.execute_statistic(x)

        result.sort()

        ecdf = scipy_stats.ecdf(result)
        x_cr = np.quantile(ecdf.cdf.quantiles, q=1 - alpha)
        cache[self.code()][rvs_size] = x_cr
        return x_cr

    def test(self, rvs, alpha):
        x_cr = self.calculate_critical_value(len(rvs), alpha)
        statistic = self.execute_statistic(rvs)

        return False if statistic > x_cr else True

    def generate(self, size, mean=0, var=1):
        return norm.generate_norm(size, mean, var)


class KSTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'KS'

    def execute_statistic(self, rvs):
        """

        :param rvs: unsorted vector
        :return:
        """
        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.norm.cdf(rvs)
        d_plus, _ = KSTest.__compute_dplus(cdf_vals, rvs)
        d_minus, _ = KSTest.__compute_dminus(cdf_vals, rvs)

        return d_plus if d_plus > d_minus else d_minus

    @staticmethod
    def __compute_dplus(cdf_vals, rvs):
        n = len(cdf_vals)
        d_plus = (np.arange(1.0, n + 1) / n - cdf_vals)
        a_max = d_plus.argmax()
        loc_max = rvs[a_max]
        return d_plus[a_max], loc_max

    @staticmethod
    def __compute_dminus(cdf_vals, rvs):
        n = len(cdf_vals)
        d_minus = (cdf_vals - np.arange(0.0, n) / n)
        a_max = d_minus.argmax()
        loc_max = rvs[a_max]
        return d_minus[a_max], loc_max


class ChiSquareTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'ChiSquare'

    def execute_statistic(self, rvs):
        f_obs = np.asanyarray(rvs)
        f_obs_float = f_obs.astype(np.float64)
        f_exp = [16, 16, 16, 16, 16, 8]
        terms = (f_obs_float - f_exp) ** 2 / f_exp
        return terms.sum(axis=0)


class ADTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'AD'

    def execute_statistic(self, rvs):
        n = len(rvs)

        s = np.std(rvs, ddof=1, axis=0)
        y = np.sort(rvs)
        xbar = np.mean(rvs, axis=0)
        w = (y - xbar) / s

        # TODO: add mean and var
        log_cdf = scipy_stats.distributions.norm.logcdf(w)
        log_sf = scipy_stats.distributions.norm.logsf(w)

        i = np.arange(1, n + 1)
        a_2 = -n - np.sum((2 * i - 1.0) / n * (log_cdf + log_sf[::-1]), axis=0)
        return a_2


class SWTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'SW'

    def execute_statistic(self, rvs):
        f_obs = np.asanyarray(rvs)
        f_obs_sorted = np.sort(f_obs)
        x_mean = np.mean(f_obs)

        denominator = (f_obs - x_mean) ** 2
        denominator = denominator.sum()

        a = self.ordered_statistic(len(f_obs))
        terms = a * f_obs_sorted
        return (terms.sum() ** 2) / denominator

    @staticmethod
    def ordered_statistic(n):
        if n == 3:
            sqrt = np.sqrt(0.5)
            return np.array([sqrt, 0, -sqrt])

        m = np.array([scipy_stats.norm.ppf((i - 3 / 8) / (n + 0.25)) for i in range(1, n + 1)])

        m2 = m ** 2
        term = np.sqrt(m2.sum())
        cn = m[-1] / term
        cn1 = m[-2] / term

        p1 = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, cn]
        u = 1 / np.sqrt(n)

        wn = np.polyval(p1, u)
        # wn = np.array([p1[0] * (u ** 5), p1[1] * (u ** 4), p1[2] * (u ** 3), p1[3] * (u ** 2), p1[4] * (u ** 1), p1[5]]).sum()
        w1 = -wn

        if n == 4 or n == 5:
            phi = (m2.sum() - 2 * m[-1] ** 2) / (1 - 2 * wn ** 2)
            phi_sqrt = np.sqrt(phi)
            result = np.array([m[k] / phi_sqrt for k in range(1, n - 1)])
            return np.concatenate([[w1], result, [wn]])

        p2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, cn1]

        if n > 5:
            wn1 = np.polyval(p2, u)
            w2 = -wn1
            phi = (m2.sum() - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * wn ** 2 - 2 * wn1 ** 2)
            phi_sqrt = np.sqrt(phi)
            result = np.array([m[k] / phi_sqrt for k in range(2, n - 2)])
            return np.concatenate([[w1, w2], result, [wn1, wn]])