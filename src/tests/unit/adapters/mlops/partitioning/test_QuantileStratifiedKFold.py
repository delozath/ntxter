import pytest


import numpy as np
import pandas as pd

from sklearn.metrics import recall_score


from ntxter.adapters.mlops.partitioning import QuantileStratifiedKFold

N_SPLITS = 8
N_ROWS = 400
N_COLS = 4
N_TOL = 3

def gen_data():
    """
    Returns
    -------
    y: y[mask] enhanced outliers for testing purposes
    """
    y = np.random.normal(0, 1, N_ROWS)
    mask = (np.quantile(y, 0.025) < y) & (y <  np.quantile(y, 0.975))
    y[~mask] = 3 * y[~mask]

    M = np.random.uniform(-5, 5, N_COLS)
    S = np.identity(N_COLS)
    X = np.random.multivariate_normal(M, S, N_ROWS)

    return X, y, mask


#[TEST]
#[PASSED]
# - assert splits correctness
# - tt and tn have no common indices
# - tt and tn combined have all indices
def test_BaseQuantileStratifiedKFold_split_no_clip_outliers():
    X, y, _ = gen_data()
    instance = QuantileStratifiedKFold(n_splits=N_SPLITS)

    assert instance.get_n_splits() == N_SPLITS
    for tn, tt in instance.split(X, y):
        assert len(set(tt).intersection(set(tn))) == 0
        assert np.concatenate((tn, tt)).shape[0] == y.shape[0]

#[TEST]
#[PASSED]
# - assert splits correctness with outliers clipping
# - tn splits are different among iterations
# - tt and tn have no common indices
# - tt and tn combined have all non-outlier indices
def test_BaseQuantileStratifiedKFold_split_clip_outliers():
    X, y, mask = gen_data()
    index = np.arange(y.shape[0])
    instance = QuantileStratifiedKFold(n_splits=N_SPLITS, clip_outliers='tukey')

    tmp = None
    for tn, tt, outlier in instance.split_outliers_clipping(X, y):
        if tmp is not None:
           assert np.setdiff1d(tn, tmp).shape[0] != 0

        assert len(np.intersect1d(tn, tt)) == 0

        missing_index = np.setdiff1d(index, np.concatenate((tn, tt)))
        np.testing.assert_array_equal(index[~mask], missing_index)
        
        assert np.concatenate((tn, tt)).shape[0] < index.shape[0]

        tmp = tn