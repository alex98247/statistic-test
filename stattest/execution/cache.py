from stattest.core.store import FastJsonStoreService


class CacheResultService(FastJsonStoreService):

    def get_with_prefix(self, keys: [str]) -> {}:
        """
        Get JSON value by prefix chain in 'keys' param.

        :param keys: keys chain prefix param
        """

        key_prefix = self._create_key(keys)
        result = {}
        for key in self.cache:
            if key.startswith(key_prefix):
                result[key] = self.get(key)
        return result

    def get_level_prefixes(self, keys: [str], level: int) -> set:
        """
        Get JSON value by prefix chain in 'keys' param.

        :param keys:
        :param level:
        """

        key_prefix = self._create_key(keys)
        result = []
        for key in self.cache:
            split = key.split(self.separator)
            if len(split) > level and key.startswith(key_prefix):
                result.append(split[level])
        return set(result)
