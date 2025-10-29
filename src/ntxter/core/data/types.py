from abc import ABC, abstractmethod


import numpy as np


from ntxter.core.base.descriptors import UnpackDataAndCols, ArrayIndexSlice


class Bundles(ABC):
    _X = UnpackDataAndCols()
    _y = UnpackDataAndCols()
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

class BundleTrainTest(Bundles):
    X_train = ArrayIndexSlice()
    X_test  = ArrayIndexSlice()
    y_train = ArrayIndexSlice()
    y_test  = ArrayIndexSlice()
    
    def __init__(self, X, y):
        super().__init__(X, y)
    
    def split(self, n_train, n_test):
        self.X_train = self.X, n_train
        self.X_test  = self.X, n_test
        self.y_train = self.y, n_train
        self.y_test  = self.y, n_test