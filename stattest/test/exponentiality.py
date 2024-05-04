import math

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
class AHSTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'AHS_exp'

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


class ATKTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'ATK_exp'

    def execute_statistic(self, rvs, p=0.99):
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


class DeshpandeTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'Deshpande_exp'

    def execute_statistic(self, rvs, b=0.44):
        """
        Deshpande test statistic for exponentiality.

        Parameters
        ----------
        b : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        des : float
            The test statistic.
        """

        n = len(rvs)
        des = 0
        for i in range(n):
            for k in range(n):
                if (i != k) and (rvs[i] > b * rvs[k]):
                    des += 1
        des /= (n * (n - 1))

        return des


class EPSTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'EPS_exp'

    def execute_statistic(self, rvs):
        """
        Epstein test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        eps : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        x = np.concatenate(([0], rvs))
        d = (np.arange(n, 0, -1)) * (x[1:n + 1] - x[0:n])
        eps = 2 * n * (np.log(np.sum(d) / n) - (np.sum(np.log(d))) / n) / (1 + (n + 1) / (6 * n))

        return eps


class FroziniTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'Frozini_exp'

    def execute_statistic(self, rvs):
        """
        Frozini test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        froz : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        froz = (1 / np.sqrt(n)) * np.sum(np.abs(1 - np.exp(-rvs / y) - (np.arange(1, n + 1) - 0.5) / n))

        return froz


class GiniTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'Gini_exp'

    def execute_statistic(self, rvs):
        """
        Gini test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        gini : float
            The test statistic.
        """

        n = len(rvs)
        a = np.arange(1, n)
        b = np.arange(n - 1, 0, -1)
        a = a * b
        x = np.sort(rvs)
        k = x[1:] - x[:-1]
        gini = np.sum(k * a) / ((n - 1) * np.sum(x))

        return gini


class GDTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'GD_exp'

    def execute_statistic(self, rvs, r=None):
        """
        Gnedenko F-test statistic for exponentiality.

        Parameters
        ----------
        r : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        gd : float
            The test statistic.
        """

        if r is None:
            r = round(len(rvs) / 2)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        D = (np.arange(n, 0, -1)) * (x[1:n + 1] - x[0:n])
        gd = (sum(D[:r]) / r) / (sum(D[r:]) / (n - r))

        return gd


class HMTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'HM_exp'

    def execute_statistic(self, rvs, r=None):
        """
        Harris' modification of Gnedenko F-test.

        Parameters
        ----------
        r : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hm : float
            The test statistic.
        """

        if r is None:
            r = round(len(rvs) / 4)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        D = (np.arange(n, 0, -1)) * (x[1:n + 1] - x[:n])
        hm = ((np.sum(D[:r]) + np.sum(D[-r:])) / (2 * r)) / ((np.sum(D[r:-r])) / (n - 2 * r))

        return hm


class HG1TestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'HG1_exp'

    def execute_statistic(self, rvs):
        """
        Hegazy-Green 1 test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hg : float
            The test statistic.
        """

        n = len(rvs)
        x = np.sort(rvs)
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum(np.abs(x - b))

        return hg

'''
class HPTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'HP_exp'

    def execute_statistic(self, rvs):
        """
        Hollander-Proshan test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hp : float
            The test statistic.
        """

        n = len(rvs)
        t = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i != j) and (i != k) and (j < k) and (rvs[i] > rvs[j] + rvs[k]):
                        t += 1
        hp = (2 / (n * (n - 1) * (n - 2))) * t

        return hp
'''


class KMTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'KM_exp'

    def execute_statistic(self, rvs):
        """
        Kimber-Michael test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        km : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        s = (2 / np.pi) * np.arcsin(np.sqrt(1 - np.exp(-(rvs / y))))
        r = (2 / np.pi) * np.arcsin(np.sqrt((np.arange(1, n + 1) - 0.5) / n))
        km = max(abs(r - s))

        return km


class KCTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'KC_exp'

    def execute_statistic(self, rvs):
        """
        Kochar test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        kc : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        u = np.array([(i + 1) / (n + 1) for i in range(n)])
        j = 2 * (1 - u) * (1 - np.log(1 - u)) - 1
        kc = np.sqrt(108 * n / 17) * (np.sum(j * rvs)) / np.sum(rvs)

        return kc


class LZTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'LZ_exp'

    def execute_statistic(self, rvs, p=0.5):
        """
        Lorenz test statistic for exponentiality.

        Parameters
        ----------
        p : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        lz : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        lz = sum(rvs[:int(n * p)]) / sum(rvs)

        return lz


class MNTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'MN_exp'

    def execute_statistic(self, rvs):
        """
        Moran test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        mn : float
            The test statistic.
        """

        n = len(rvs)
        y = np.mean(rvs)
        mn = -scipy_special.digamma(1) + np.mean(np.log(rvs / y))

        return mn


class PTTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'PT_exp'

    def execute_statistic(self, rvs):
        """
        Pietra test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        pt : float
            The test statistic.
        """

        n = len(rvs)
        xm = np.mean(rvs)
        pt = np.sum(np.abs(rvs - xm)) / (2 * n * xm)

        return pt


class SWTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'SW_exp'

    def execute_statistic(self, rvs):
        """
        Shapiro-Wilk test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        sw : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        sw = n * (y - rvs[0]) ** 2 / ((n - 1) * np.sum((rvs - y) ** 2))

        return sw

'''
class RSTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'RS_exp'

    def execute_statistic(self, rvs):
        """
        Statistic of the exponentiality test based on Rossberg characterization.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        rs : float
            The test statistic.
        """

        n = len(rvs)
        sh = 0
        sg = 0
        for m in range(n):
            h = 0
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if (rvs[i] + rvs[j] + rvs[k] - 2 * min(rvs[i], rvs[j], rvs[k]) - max(rvs[i], rvs[j], rvs[k]) < rvs[m]):
                            h += 1
            h = ((6 * math.factorial(n - 3)) / math.factorial(n)) * h
            sh += h
        for m in range(n):
            g = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if min(rvs[i], rvs[j]) < rvs[m]:
                        g += 1
            g = ((2 * math.factorial(n - 2)) / math.factorial(n)) * g
            sg += g
        rs = sh - sg
        rs /= n

        return rs
'''

class WETestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'WE_exp'

    def execute_statistic(self, rvs):
        """
        WE test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        we : float
            The test statistic.
        """

        n = len(rvs)
        m = np.mean(rvs)
        v = np.var(rvs)
        we = (n - 1) * v / (n ** 2 * m ** 2)

        return we


class WWTestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'WW_exp'

    def execute_statistic(self, rvs):
        """
        Wong and Wong test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ww : float
            The test statistic.
        """

        n = len(rvs)
        ww = max(rvs) / min(rvs)

        return ww


class HG2TestExp(AbstractExponentialityTest):

    @staticmethod
    def code():
        return 'HG2_exp'

    def execute_statistic(self, rvs):
        """
        Hegazy-Green 2 test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hg : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum((rvs - b) ** 2)

        return hg
