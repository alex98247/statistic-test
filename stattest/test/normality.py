import math

from stattest.core.norm import pdf_norm
from stattest.core.sample import central_moment
from stattest.test.AbstractTest import AbstractTest
from stattest.core import norm
import numpy as np
import scipy.stats as scipy_stats
from scipy.stats.stats import normaltest
import pandas as pd

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
        return 'CHI2'

    def execute_statistic(self, rvs):
        rvs = np.sort(rvs)

        f_obs = np.asanyarray(rvs)
        f_obs_float = f_obs.astype(np.float64)
        f_exp = pdf_norm(rvs)
        terms = (f_obs_float - f_exp) ** 2 / f_exp
        return terms.sum(axis=0)


class ADTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'AD'

    def execute_statistic(self, rvs):
        n = len(rvs)

        s = np.std(rvs)
        y = np.sort(rvs)
        xbar = np.mean(rvs)
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


class SWMTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'SWM'

    def execute_statistic(self, rvs):
        n = len(rvs)

        rvs = np.sort(rvs)
        vals = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.norm.cdf(vals)

        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        CM = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)
        return CM


class LillieforsTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'Lilliefors'

    def execute_statistic(self, rvs):
        n = len(rvs)
        vals = np.sort(np.asarray(rvs))
        s = np.arange(0, n) / n
        cdf_vals = scipy_stats.norm.cdf(vals)
        return np.max(np.abs(cdf_vals - s))


class DATest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'DA'

    def execute_statistic(self, rvs):
        x = np.asanyarray(rvs)
        y = np.sort(x)
        n = len(x)

        x_mean = np.mean(x)
        m2 = np.sum((x - x_mean) ** 2) / n
        i = np.arange(1, n + 1)
        c = (n + 1) / 2
        terms = (i - c) * y
        stat = terms.sum() / (n ** 2 * np.sqrt(m2))
        return stat


class JBTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'JB'

    def execute_statistic(self, rvs):
        x = np.asarray(rvs)
        x = x.ravel()
        axis = 0

        n = x.shape[axis]
        if n == 0:
            raise ValueError('At least one observation is required.')

        mu = x.mean(axis=axis, keepdims=True)
        diffx = x - mu
        s = scipy_stats.skew(diffx, axis=axis, _no_deco=True)
        k = scipy_stats.kurtosis(diffx, axis=axis, _no_deco=True)
        statistic = n / 6 * (s ** 2 + k ** 2 / 4)
        return statistic


class SkewTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'SKEW'

    def execute_statistic(self, rvs):
        x = np.asanyarray(rvs)
        y = np.sort(x)

        return self.skew_test(y)

    @staticmethod
    def skew_test(a):
        n = len(a)
        if n < 8:
            raise ValueError(
                "skew test is not valid with less than 8 samples; %i samples"
                " were given." % int(n))
        b2 = scipy_stats.skew(a, axis=0)
        y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
        beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) /
                 ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)))
        W2 = -1 + math.sqrt(2 * (beta2 - 1))
        delta = 1 / math.sqrt(0.5 * math.log(W2))
        alpha = math.sqrt(2.0 / (W2 - 1))
        y = np.where(y == 0, 1, y)
        Z = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))

        return Z


class KurtosisTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'KURTOSIS'

    def execute_statistic(self, rvs):
        x = np.asanyarray(rvs)
        y = np.sort(x)

        return self.kurtosis_test(y)

    @staticmethod
    def kurtosis_test(a):
        n = len(a)
        if n < 5:
            raise ValueError(
                "kurtosistest requires at least 5 observations; %i observations"
                " were given." % int(n))
        # if n < 20:
        #    warnings.warn("kurtosistest only valid for n>=20 ... continuing "
        #                  "anyway, n=%i" % int(n),
        #                  stacklevel=2)
        b2 = scipy_stats.kurtosis(a, axis=0, fisher=False)

        E = 3.0 * (n - 1) / (n + 1)
        varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5))  # [1]_ Eq. 1
        x = (b2 - E) / np.sqrt(varb2)  # [1]_ Eq. 4
        # [1]_ Eq. 2:
        sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * np.sqrt((6.0 * (n + 3) * (n + 5)) /
                                                                              (n * (n - 2) * (n - 3)))
        # [1]_ Eq. 3:
        A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + np.sqrt(1 + 4.0 / (sqrtbeta1 ** 2)))
        term1 = 1 - 2 / (9.0 * A)
        denom = 1 + x * np.sqrt(2 / (A - 4.0))
        term2 = np.sign(denom) * np.where(denom == 0.0, np.nan,
                                          np.power((1 - 2.0 / A) / np.abs(denom), 1 / 3.0))
        # if np.any(denom == 0):
        #    msg = ("Test statistic not defined in some cases due to division by "
        #           "zero. Return nan in that case...")
        #    warnings.warn(msg, RuntimeWarning, stacklevel=2)

        Z = (term1 - term2) / np.sqrt(2 / (9.0 * A))  # [1]_ Eq. 5

        return Z


class DAPTest(SkewTest, KurtosisTest):

    @staticmethod
    def code():
        return 'DAP'

    def execute_statistic(self, rvs):
        x = np.asanyarray(rvs)
        y = np.sort(x)

        s = self.skew_test(y)
        k = self.kurtosis_test(y)
        k2 = s * s + k * k
        return k2


# https://github.com/puzzle-in-a-mug/normtest
class FilliTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'Filli'

    def execute_statistic(self, rvs):
        uniform_order = self._uniform_order_medians(len(rvs))
        zi = self._normal_order_medians(uniform_order)
        x_data = np.sort(rvs)
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    @staticmethod
    def _uniform_order_medians(sample_size):
        i = np.arange(1, sample_size + 1)
        mi = (i - 0.3175) / (sample_size + 0.365)
        mi[0] = 1 - 0.5 ** (1 / sample_size)
        mi[-1] = 0.5 ** (1 / sample_size)

        return mi

    @staticmethod
    def _normal_order_medians(mi):
        normal_ordered = scipy_stats.norm.ppf(mi)
        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        correl = scipy_stats.pearsonr(x_data, zi)[0]
        return correl


# https://github.com/puzzle-in-a-mug/normtest
class LooneyGulledgeTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'LG'

    def execute_statistic(self, rvs):
        # ordering
        x_data = np.sort(rvs)

        # zi
        zi = self._normal_order_statistic(
            x_data=x_data,
            weighted=False,  # TODO: False or True
        )

        # calculating the stats
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    @staticmethod
    def _normal_order_statistic(x_data, weighted=False):
        # ordering
        x_data = np.sort(x_data)
        if weighted:
            df = pd.DataFrame({"x_data": x_data})
            # getting mi values
            df["Rank"] = np.arange(1, df.shape[0] + 1)
            df["Ui"] = LooneyGulledgeTest._order_statistic(
                sample_size=x_data.size,
            )
            df["Mi"] = df.groupby(["x_data"])["Ui"].transform("mean")
            normal_ordered = scipy_stats.norm.ppf(df["Mi"])
        else:
            ordered = LooneyGulledgeTest._order_statistic(
                sample_size=x_data.size,
            )
            normal_ordered = scipy_stats.norm.ppf(ordered)

        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        correl = scipy_stats.pearsonr(zi, x_data)[0]
        return correl

    @staticmethod
    def _order_statistic(sample_size):
        i = np.arange(1, sample_size + 1)
        cte_alpha = 3 / 8
        return (i - cte_alpha) / (sample_size - 2 * cte_alpha + 1)


# https://github.com/puzzle-in-a-mug/normtest
class RyanJoinerTest(AbstractNormalityTest):
    def __init__(self, weighted=False, cte_alpha="3/8"):
        super().__init__()
        self.weighted = weighted
        self.cte_alpha = cte_alpha

    @staticmethod
    def code():
        return 'RJ'

    def execute_statistic(self, rvs):
        # ordering
        x_data = np.sort(rvs)

        # zi
        zi = self._normal_order_statistic(
            x_data=x_data,
            weighted=self.weighted,
            cte_alpha=self.cte_alpha,
        )

        # calculating the stats
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    def _normal_order_statistic(self, x_data, weighted=False, cte_alpha="3/8"):
        # ordering
        x_data = np.sort(x_data)
        if weighted:
            df = pd.DataFrame({"x_data": x_data})
            # getting mi values
            df["Rank"] = np.arange(1, df.shape[0] + 1)
            df["Ui"] = self._order_statistic(
                sample_size=x_data.size,
                cte_alpha=cte_alpha,
            )
            df["Mi"] = df.groupby(["x_data"])["Ui"].transform("mean")
            normal_ordered = scipy_stats.norm.ppf(df["Mi"])
        else:
            ordered = self._order_statistic(
                sample_size=x_data.size,
                cte_alpha=cte_alpha,
            )
            normal_ordered = scipy_stats.norm.ppf(ordered)

        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        return scipy_stats.pearsonr(zi, x_data)[0]

    @staticmethod
    def _order_statistic(sample_size, cte_alpha="3/8"):
        i = np.arange(1, sample_size + 1)
        if cte_alpha == "1/2":
            cte_alpha = 0.5
        elif cte_alpha == "0":
            cte_alpha = 0
        else:
            cte_alpha = 3 / 8

        return (i - cte_alpha) / (sample_size - 2 * cte_alpha + 1)


class SFTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'SF'

    def execute_statistic(self, rvs):
        n = len(rvs)
        rvs = np.sort(rvs)

        x_mean = np.mean(rvs)
        alpha = 0.375
        terms = (np.arange(1, n + 1) - alpha) / (n - 2 * alpha + 1)
        e = -scipy_stats.norm.ppf(terms)

        w = np.sum(e * rvs) ** 2 / (np.sum((rvs - x_mean) ** 2) * np.sum(e ** 2))
        return w


# https://habr.com/ru/articles/685582/
class EPTest(AbstractNormalityTest):

    @staticmethod
    def code():
        return 'EP'

    def execute_statistic(self, rvs):
        n = len(rvs)
        X = np.sort(rvs)
        X_mean = np.mean(X)
        m2 = np.var(X, ddof=0)

        A = np.sqrt(2) * np.sum([np.exp(-(X[i] - X_mean) ** 2 / (4 * m2)) for i in range(n)])
        B = 2 / n * np.sum(
            [np.sum([np.exp(-(X[j] - X[k]) ** 2 / (2 * m2)) for j in range(0, k)])
             for k in range(1, n)])
        t = 1 + n / np.sqrt(3) + B - A
        return t
