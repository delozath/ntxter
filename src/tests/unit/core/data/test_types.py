import pytest

import numpy as np
import pandas as pd

from pathlib import Path

from ntxter.core.data.types import Bundles, BundleTrainTest


data = np.arange(30)[:, None]
data = pd.DataFrame(
    np.concatenate([data, -2*data, data/4], axis=1),
    columns=['normal', 'double', 'fourth']
)
y = (np.ones(15)[:, None] @ np.array([[1, -1]])).flatten('F')


def test_Bundles_init():
    #[TEST]
    #[PASSED]
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

def test_BundleTrainTest():
    #[TEST]
    #[PASSED]
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

    assert (bundle.X_train != data.iloc[n_train].values).sum() == 0
    assert (bundle.X_test  != data.iloc[n_test] .values).sum() == 0
    assert (bundle.y_train != y[n_train]).sum() == 0
    assert (bundle.y_test  != y[n_test ]).sum() == 0