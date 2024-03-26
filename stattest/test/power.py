from stattest.test import AbstractTest
from tqdm import tqdm

from stattest.test.generator import AbstractRVSGenerator


def calculate_test_power(test: AbstractTest, rvs_generator: AbstractRVSGenerator, alpha=0.05, rvs_size=15,
                         count=1_000_000):
    k = 0
    for i in tqdm(range(count)):
        x = test.test(rvs_generator.generate(rvs_size), alpha=alpha)
        if x is False:
            k = k + 1
    return k / count


def calculate_mean_test_power(test: AbstractTest, rvs_generators: [AbstractRVSGenerator], alpha=0.05, rvs_size=15,
                              count=1_000_000):
    k = 0
    for generator in rvs_generators:
        power = calculate_test_power(test, generator, alpha=alpha, rvs_size=rvs_size, count=count)
        print('power', power)
        k = k + power
    return k / len(rvs_generators)
