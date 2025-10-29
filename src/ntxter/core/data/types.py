from abc import ABC, abstractmethod


import numpy as np


from ntxter.core.base.descriptors import UnpackDataAndCols


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