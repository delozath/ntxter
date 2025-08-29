import numpy as np
from functools import reduce

def nested_keys(d, keys, default=None):
    try:
        return reduce(lambda d, key: d[key], keys, d)
    except (KeyError, TypeError):
        return default
#
#
class ColnameIndexing:
    def __init__(self, cols) -> None:
        self.cols = cols
        self._index = np.arange(len(cols))
    #
    @property
    def index(self):
        return self._index
    #
    @index.setter
    def index(self, _):
        raise ValueError("Index value is not mutable")
    #
    def get(self, columns):
        cols = np.array(self.cols)[:, None]
        mask = cols == columns
        found = mask.any(axis=0)
        index = mask.argmax(axis=0)
        #
        if found.all():
            return index, 0
        else:
            return index[found], np.array(columns)[~found]