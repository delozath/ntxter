import pytest

import numpy as np
import pandas as pd

from pathlib import Path

from ntxter.core.data.types import Bundles, BundleTrainTest, BundleMultSplit


data = np.arange(50)[:, None]
data = pd.DataFrame(
    np.concatenate([data, -2*data, data/4], axis=1),
    columns=['normal', 'double', 'fourth']
)
y = (np.ones(25)[:, None] @ np.array([[1, -1]])).flatten('F')


def compare_arrays(a0, a1):
    return (a0 != a1).sum()

#[TEST]
#[PASSED]
# Bundles
def test_Bundles_init():
    # - bundle.X == data given
    # - bundle.y == y given
    # - bundle.X_names == column names
    # - bundle.y_names == zero, is a np.ndarray
    # - test if index sequence
    # - test auxiliar _X, _y were set to None
    # - test split Dummy implementation

    class Dummy(Bundles):
        def __init__(self, X, y):
            super().__init__(X, y)

        def split(self, *args, **kwargs):
             self.test = True
    
    bundle = Dummy(data, y)

    assert (bundle.X != data).sum().sum() == 0
    assert (bundle.y != y).sum().sum() == 0
    assert all(bundle.X_names == data.columns)
    assert bundle.y_names[0] == 0
    assert all(bundle.index == bundle.X[:, 0])
    assert bundle._X is None
    assert bundle._y is None
    
    bundle.split()
    assert bundle.test == True

#[TEST]
#[PASSED]
# BundleTrainTest
def test_BundleTrainTest():
    # - test bundle train x, y equal original data[train], data[train]
    # - test bundle test x, y equal original data[test], data[test]

    N = y.shape[0]
    perc = np.random.uniform(0.5, 0.8)
    K = int(np.ceil(perc*N))
    bundle = BundleTrainTest(data, y)

    index_ref = np.arange(N)
    np.random.shuffle(index_ref)

    n_train = index_ref[:K]
    n_test = index_ref[K:]

    bundle.split(n_train, n_test)

    assert compare_arrays(bundle.X_train,  data.iloc[n_train].values) == 0
    assert compare_arrays(bundle.X_test,   data.iloc[n_test] .values) == 0
    assert compare_arrays(bundle.y_train,  y[n_train]) == 0
    assert compare_arrays(bundle.y_test,   y[n_test ]) == 0


#[TEST]
#[]
# BundleMultSplit
def split_len(bundle, perc_range: tuple) -> tuple:
    perc = np.random.uniform(*perc_range)
    N = bundle.y.shape[0]
    K = int(np.ceil(N * perc))

    return N, K

def split_index(bundle, perc_range):
    N, K = split_len(bundle, perc_range)
    index_ref = np.arange(N)
    np.random.shuffle(index_ref)
    index_left = index_ref[:K]
    index_right = index_ref[K:]

    return index_left, index_right

def test_BundleMultSplit():
    # - store unseen data, reduce X
    # - store additional partition, reduce X
    # - split train and test, testing if all reassembled splits are equal to original data
    bundle = BundleMultSplit(data, y)
    X_full, y_full = bundle.X.copy(), bundle.y.copy()

    index_unseen, index_split1 = split_index(bundle, (0.1, 0.2))
    bundle.split_unseen(index_unseen)

    assert compare_arrays(bundle.X_unseen, X_full[index_unseen]) == 0
    assert bundle.X.shape[0] + len(index_unseen) == X_full.shape[0]

    X_split_1 = bundle.X.copy()
    index_split2, index_conformal = split_index(bundle, (0.6, 0.8))
    bundle.reserve(index_conformal, 'conf')

    assert compare_arrays(bundle.X_conf, X_split_1[index_conformal]) == 0
    assert bundle.X.shape[0] + len(index_conformal) == X_split_1.shape[0]

    X_split_2 = bundle.X.copy()
    n_train, n_test = split_index(bundle, (0.6, 0.8))
    bundle.split(n_train, n_test)

    df_X_test = pd.DataFrame(
        np.concatenate((bundle.X_train, bundle.X_test, bundle.X_conf, bundle.X_unseen))
    )
    df_X_test.columns = data.columns
    df_X_test = df_X_test.assign(y=np.concatenate((bundle.y_train, bundle.y_test, bundle.y_conf, bundle.y_unseen)))
    df_X_test = df_X_test.sort_values(by='normal')
    
    assert compare_arrays(df_X_test[data.columns].values, data.values) == 0
    assert compare_arrays(df_X_test['y'].values, y) == 0