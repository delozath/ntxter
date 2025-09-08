import numpy as np
from ntxter.validation import UnpackDataAndCols, ArrayIndexSlice

from abc import ABC, abstractmethod

import warnings

class _Bundle(ABC):
    _X = UnpackDataAndCols()
    _y = UnpackDataAndCols()
    def __init__(self, X, y):
        self._X = X
        self._y = y
        #
        self.X, self.X_names = self._X
        self.y, self.y_names = self._y
        #
        self._X, self._y = None, None
        #
        self.index = np.arange(self.X.shape[0])
    #
    @abstractmethod
    def split(self, *args, **kwargs):
        pass
#
#
class BundleTrainTestSplit(_Bundle):
    X_train = ArrayIndexSlice()
    X_test  = ArrayIndexSlice()
    y_train = ArrayIndexSlice()
    y_test  = ArrayIndexSlice()
    #
    def __init__(self, X, y):
        super().__init__(X, y)
    #
    def split(self, n_train, n_test):
        self.X_train = self.X, n_train
        self.X_test  = self.X, n_test
        self.y_train = self.y, n_train
        self.y_test  = self.y, n_test
#
#
class BundleTrainTestUnseenSplit(BundleTrainTestSplit):
    #
    def __init__(self, X, y) -> None:
        super().__init__(X, y)
        self._unseen_flag = False
    #
    def set_unseen(self, idx):
        mask = np.array([True if i in idx else False for i in self.index])
        self.X_unseen = self.X[mask].copy()
        self.y_unseen = self.y[mask].copy()
        #
        self.X = self.X[~mask].copy()
        self.y = self.y[~mask].copy()
        self.index = np.arange(self.X.shape[0])
        #
        self._unseen_flag = True
    #
    def split(self, n_train, n_test):
        if self._unseen_flag:
            return super().split(n_train, n_test)
        else:
            raise RuntimeError("Unseed data indexes have to be assigned before performing the split method")