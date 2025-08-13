from functools import reduce

def nested_keys(d, keys, default=None):
    try:
        return reduce(lambda d, key: d[key], keys, d)
    except (KeyError, TypeError):
        return default