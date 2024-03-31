import scipy.stats as sts
import numpy as np
import unittest
import stattest.stats as stats
from stattest.test import KSTest


class TestStatMethods(unittest.TestCase):
    def test_chisquare(self):
        ch2 = stats.chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8])
        self.assertEqual(3.5, ch2)

    def test_kstest(self):
        ks_test = KSTest()
        x = sts.norm.rvs(size=10)

        x1 = sts.norm.rvs(size=10, loc=11)
        print(x1, sts.kstest(x1, 'norm').statistic)

        x1 = sts.norm.rvs(size=10, scale=5)
        print(x1, sts.kstest(x1, 'norm').statistic)

        x1 = sts.norm.rvs(size=10, loc=11, scale=5)
        print(x1, sts.kstest(x1, 'norm').statistic)

        ch2 = ks_test.execute_statistic(x)
#[ 0.38323312 -1.10386561  0.75226465 -2.23024566 -0.27247827  0.95926434,  0.42329541 -0.11820711  0.90892169 -0.29045373]
# 0.18573457378941832
        self.assertEqual(sts.kstest(x, 'norm').statistic, ch2)

    def test_swstest(self):
        ch2 = stats.swstest([16, 18, 16])
        self.assertAlmostEqual(0.75, ch2, 5)

        ch2 = stats.swstest([16, 18, 16, 14])
        self.assertAlmostEqual(0.9446643, ch2, 5)

        ch2 = stats.swstest([16, 18, 16, 14, 15])
        self.assertAlmostEqual(0.955627, ch2, 5)

        ch2 = stats.swstest([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0])
        self.assertAlmostEqual(0.872973, ch2, 5)

    def test_adstest(self):
        ch2 = stats.adstest([16, 18, 16, 14, 12, 12, 16, 18, 16, 14, 12, 12])
        self.assertAlmostEqual(0.6822883, ch2, 5)

    def test_cvmtest(self):
        data = np.array([12, 12, 12, 12, 14, 14, 16, 16, 16, 16, 18])
        ch2 = stats.cvmtest(data)
        x = sts.norm.rvs(size=10)
        print(x, sts.cramervonmises(x, 'norm'))
        self.assertAlmostEqual(3.66666666, ch2, 5)

    def test_jbtest(self):
        data = np.array([12, 12, 12, 12, 14, 14, 16, 16, 16, 16, 18])
        ch2 = stats.cvmtest(data)
        x = sts.norm.rvs(size=10)
        print(x, sts.jarque_bera(x))
        self.assertAlmostEqual(3.66666666, ch2, 5)

    def test_lilliefors(self):
        data = np.array([-1, 0, 1])
        ch2 = stats.lilliefors(data)
        self.assertAlmostEqual(0.17467808, ch2, 5)

    def test_dastest(self):
        data = np.array([-1, 0, 1])
        ch2 = stats.dastest(data)
        self.assertAlmostEqual(0, ch2, 5)

    def test_dapstest(self):
        data = np.array([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0])
        ch2 = stats.dapstest(data)
        self.assertAlmostEqual(0, ch2, 5)

    def test_sfstest(self):
        data = np.array(
            [-1.5461357, 0.8049704, -1.2676556, 0.1912453, 1.4391551, 0.5352138, -1.6212611, 0.1015035, -0.2571793,
             0.8756286])
        data = np.sort(data)
        print(data)
        ch2 = stats.sfstest(data)
        self.assertAlmostEqual(0.93569, ch2, 5)

    def test_filli_test(self):
        data = np.array(
            [-1.5461357, 0.8049704, -1.2676556, 0.1912453, 1.4391551, 0.5352138, -1.6212611, 0.1015035, -0.2571793,
             0.8756286])
        data = np.sort(data)
        print(data)
        ch2 = stats.filli_test(data)
        self.assertAlmostEqual(0, ch2, 5)

    def test_mi_test(self):
        data = np.array(
            [-1.5461357, 0.8049704, -1.2676556, 0.1912453, 1.4391551, 0.5352138, -1.6212611, 0.1015035, -0.2571793,
             0.8756286])
        data = np.sort(data)
        ch2 = stats.mi_test(data)
        self.assertAlmostEqual(0, ch2, 5)


if __name__ == '__main__':
    unittest.main()
