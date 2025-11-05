import warnings
from abc import ABC, abstractmethod
import pickle

import numpy as np
from pathlib import Path


from ntxter.core.base.descriptors import UnpackDataAndCols, ArrayIndexSlice


class Bundles(ABC):
    _X = UnpackDataAndCols()
    _y = UnpackDataAndCols()

    X_train = ArrayIndexSlice()
    X_test  = ArrayIndexSlice()
    y_train = ArrayIndexSlice()
    y_test  = ArrayIndexSlice()

    def __init__(self, X, y):
        self._X = X
        self._y = y
        
        self.X, self.X_names = self._X
        self.y, self.y_names = self._y
        
        self._X, self._y = None, None
        
        self.index = np.arange(self.X.shape[0])
    
    @abstractmethod
    def split(self, *args, **kwargs):
        pass
    
    def to_pickle(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class BundleTrainTest(Bundles):
    def __init__(self, X, y):
        super().__init__(X, y)
    
    def split(self, n_train, n_test):
        self.X_train = self.X, n_train
        self.X_test  = self.X, n_test
        self.y_train = self.y, n_train
        self.y_test  = self.y, n_test


class BundleMultSplit(BundleTrainTest):
    def __init__(self, X, y) -> None:
        super().__init__(X, y)
        self._n_splits_count = 0

    def reserve(self, index, name):
        X_tmp = self.X[index].copy()
        y_tmp = self.y[index].copy()
        setattr(self, f'X_{name}', X_tmp)
        setattr(self, f'y_{name}', y_tmp)
        
        self.X = np.delete(self.X, index, axis=0).copy()
        self.y = np.delete(self.y, index).copy()
        
        self._n_splits_count += 1
    
    def split_unseen(self, index):
        self.reserve(index, 'unseen')
    
    def split(self, n_train, n_test):
        super().split(n_train, n_test)
        if self._n_splits_count == 0:
            warnings.warn("No splits have been set, train/test partitions is performed in the whole dataset")

