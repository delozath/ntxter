import pytest

import numpy as np
import pandas as pd

from pathlib import Path

from ntxter.core.data.types import Bundles


data = np.arange(30)[:, None]
data = pd.DataFrame(
    np.concatenate([data, -2*data, data/4], axis=1),
    columns=['normal', 'double', 'fourth']
)
y = (np.ones(15)[:, None] @ np.array([[1, -1]])).flatten('F')


@pytest.fixture
def bundle():
    class Dummy(Bundles):
        def __init__(self, X, y):
            super().__init__(X, y)

        def split(self, *args, **kwargs):
             self.test = True
    instance = Dummy(data, y)
    return instance

def test_Bundles_init(bundle):
    #[TEST]
    # - bundle.X == data given
    # - bundle.y == y given
    # - bundle.X_names == column names
    # - bundle.y_names == zero, is a np.ndarray
    # - test if index sequence
    # - test auxiliar _X, _y were set to None
    # - test split Dummy implementation

    assert (bundle.X != data).sum().sum() == 0
    assert (bundle.y != y).sum().sum() == 0
    assert all(bundle.X_names == data.columns)
    assert bundle.y_names[0] == 0
    assert all(bundle.index == bundle.X[:, 0])
    assert bundle._X is None
    assert bundle._y is None
    
    bundle.split()
    assert bundle.test == True