import multiprocessing
import timeit
from itertools import repeat

import numpy as np

from stattest.test.cache import MonteCarloCacheService, ThreadSafeMonteCarloCacheService
from stattest.test.exponentiality import AbstractExponentialityTest

sizes = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def run(tests_to_run: [AbstractExponentialityTest], sizes):
    for test in tests_to_run:
        for size in sizes:
            print('Calculating...', test.code(), size)
            test.calculate_critical_value(size, 0.05, 1000)
            print('Critical value calculated', test.code(), size)


if __name__ == '__main__':
    cpu_count = 4  # multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    cache = ThreadSafeMonteCarloCacheService(lock=lock)
    tests = [cls(cache) for cls in AbstractExponentialityTest.__subclasses__()]
    tests_chunks = np.array_split(np.array(tests), cpu_count)
    with multiprocessing.Pool(cpu_count) as pool:
        pool.starmap(run, zip(tests_chunks, repeat(sizes)))
