from stattest.test.AbstractTest import AbstractTest
from stattest.core.distribution import expon
import numpy as np
import scipy.stats as scipy_stats
import scipy.special as scipy_special

from stattest.test.cache import MonteCarloCacheService


class AbstractExponentialityTest(AbstractTest):

    def __init__(self, cache=MonteCarloCacheService()):
        self.lam = 1
        self.cache = cache

    def calculate_critical_value(self, rvs_size, alpha, count=1_000_000):
        keys_cr = [self.code(), str(rvs_size), str(alpha)]
        x_cr = self.cache.get_with_level(keys_cr)
        if x_cr is not None:
            return x_cr

        d = self.cache.get_distribution(self.code(), rvs_size)
        if d is not None:
            ecdf = scipy_stats.ecdf(d)
            x_cr = np.quantile(ecdf.cdf.quantiles, q=1 - alpha)
            self.cache.put_with_level(keys_cr, x_cr)
            self.cache.flush()
            return x_cr

        result = np.zeros(count)

        for i in range(count):
            x = self.generate(size=rvs_size, lam=1)
            result[i] = self.execute_statistic(x)

        result.sort()

        ecdf = scipy_stats.ecdf(result)
        x_cr = np.quantile(ecdf.cdf.quantiles, q=1 - alpha)
        self.cache.put_with_level(keys_cr, x_cr)
        self.cache.put_distribution(self.code(), rvs_size, result)
        self.cache.flush()
        return x_cr

    def test(self, rvs, alpha):
        x_cr = self.calculate_critical_value(len(rvs), alpha)
        statistic = self.execute_statistic(rvs)

        return False if statistic > x_cr else True

    def generate(self, size, lam=1):
        return expon.generate_expon(size, lam)



class EPTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'EP_exp'

    def execute_statistic(self, rvs):
        """
        Epps and Pulley test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ep : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        ep = np.sqrt(48 * n) * np.sum(np.exp(-y) - 1 / 2) / n

        return ep


class KSTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'KS_exp'

    def execute_statistic(self, rvs):
        """
        Kolmogorov and Smirnov test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ks : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        z = np.sort(1 - np.exp(-y))
        j1 = np.arange(1, n + 1) / n
        m1 = np.max(j1 - z)
        j2 = (np.arange(0, n) + 1) / n
        m2 = np.max(z - j2)
        ks = max(m1, m2)

        return ks

'''
class AhsanullahTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'Ahsanullah_exp'

    def execute_statistic(self, rvs):
        """
        Statistic of the exponentiality test based on Ahsanullah characterization.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        a : float
            The test statistic.
        """

        n = len(rvs)
        h = 0
        g = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if abs(rvs[i] - rvs[j]) < rvs[k]:
                        h += 1
                    if 2 * min(rvs[i], rvs[j]) < rvs[k]:
                        g += 1
        a = (h - g) / (n ** 3)

        return a
'''

class AtkinsonTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'Atkinson_exp'

    def execute_statistic(self, rvs, p=0.001):
        """
        Atkinson test statistic for exponentiality.

        Parameters
        ----------
        p : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        atk : float
            The test statistic.
        """

        n = len(rvs)
        y = np.mean(rvs)
        m = np.mean(np.power(rvs, p))
        r = (m ** (1 / p)) / y
        atk = np.sqrt(n) * np.abs(r - scipy_special.gamma(1 + p) ** (1 / p))

        return atk


class COTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'CO_exp'

    def execute_statistic(self, rvs):
        """
        Cox and Oakes test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        co : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        y = np.log(y) * (1 - y)
        co = np.sum(y) + n

        return co


class CVMTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'CVM_exp'

    def execute_statistic(self, rvs):
        """
        Cramer-von Mises test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        cvm : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        z = np.sort(1 - np.exp(-y))
        c = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        z = (z - c) ** 2
        cvm = 1 / (12 * n) + np.sum(z)

        return cvm
