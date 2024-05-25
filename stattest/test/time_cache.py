import csv
import os
import timeit

from stattest.core.store import FastJsonStoreService, write_json


class TimeCacheService(FastJsonStoreService):

    def __init__(self, filename='time_cache.json', separator=':', csv_delimiter=';', dir_path='execution_time'):
        super().__init__(filename, separator)
        self.csv_delimiter = csv_delimiter
        self.dir_path = dir_path

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def count_time():
        """
        Returns the default timer.
        """

        return timeit.default_timer()

    def __build_file_path(self, test_code: str, size: int):
        file_name = 'time_' + test_code + '_' + str(size) + '.csv'
        return os.path.join(self.dir_path, file_name)

    def put_time(self, test_code: str, size: int, time: []):
        """
        Add calculation time to csv file. Name generated by time_{test_code}_{size}.csv

        :param test_code: statistic test code
        :param size: sample size
        :param time: distribution data to save
        """

        file_path = self.__build_file_path(test_code, size)
        with open(file_path, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.csv_delimiter, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(time)

    def get_time(self, test_code: str, size: int) -> [float]:
        """
        Return time cached values or None.

        :param test_code: statistic test code
        :param size: sample size
        """

        file_path = self.__build_file_path(test_code, size)
        if os.path.exists(file_path):
            with open(file_path) as f:
                reader = csv.reader(f, delimiter=self.csv_delimiter, quoting=csv.QUOTE_NONNUMERIC)
                return list(reader)
        else:
            return None


class ThreadSafeTimeCacheService(TimeCacheService):

    def __init__(self, lock, filename='time_cache.json', separator=':', csv_delimiter=';', dir_path='execution_time', cache=None):
        super().__init__(filename, separator, csv_delimiter, dir_path)
        self.lock = lock
        self.cache = cache

    def put_time(self, test_code: str, size: int, time: []):
        """
        Add calculation time to csv file. Name generated by time_{test_code}_{size}.csv.
        Thread-safe.

        :param test_code: statistic test code
        :param size: sample size
        :param time: distribution data to save
        """

        with self.lock:
            super().put_time(test_code, size, time)

    def get_time(self, test_code: str, size: int) -> [float]:
        """
        Return time cached values or None.
        Thread-safe.

        :param test_code: statistic test code
        :param size: sample size
        """

        with self.lock:
            super().get_time(test_code, size)

    def flush(self):
        """
        Flush data to persisted store.
        """

        with self.lock:
            cache_dict = dict(self.cache)
            write_json(self.filename, cache_dict)

    def put(self, key: str, value):
        """
        Put object to cache.

        :param key: cache key
        :param value: cache value
        """
        with self.lock:
            self.cache[key] = value

    def put_with_level(self, keys: [str], value):
        """
        Put JSON value by keys chain in 'keys' param.

        :param value: value to put
        :param keys: keys chain param
        """

        key = self._create_key(keys)
        with self.lock:
            self.cache[key] = value
