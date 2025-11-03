from abc import ABC, abstractmethod
from typing import Generator

import numpy as np

from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import check_array

#from ntxter.core.data.types import ArrayIndexSlice


class BaseQuantileStratifiedKFold(ABC, BaseCrossValidator):
    EPS = 1E-4
    #X_train   = ArrayIndexSlice()
    #y_train   = ArrayIndexSlice()
    #X_test    = ArrayIndexSlice()
    #y_test    = ArrayIndexSlice()
    
    @abstractmethod
    def _iter_test_masks(self, X, y, **kwargs) -> Generator:
        ...

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        X, y = self._validations(X, y)
        
        for tn_idx, tt_idx in super().split(X, y, groups):
            yield tn_idx, tt_idx
    
    def _validations(self, X, y):
        X = check_array(X, dtype=float)
        y = self._y_validation(X, y)

        return X, y
    
    def _y_validation(self, X, y):
        y = check_array(y, ensure_2d=False, dtype=float)
        n_samples, _ = X.shape
        
        if y is None:
            raise ValueError(f"y parameter is required for {self.__class__.__name__}")
        
        if len(y) != n_samples:
            raise ValueError("X and y must have the same lengths")
        
        return y
    
    def quantiles(self, y, n_bins, clip_outliers, k_outlier):       
        percentiles = np.linspace(0, 1, n_bins + 1)
        percentiles = np.quantile(y, percentiles)
        percentiles[0] -= BaseQuantileStratifiedKFold.EPS
        group = np.searchsorted(percentiles, y)
        
        if clip_outliers=='tukey':
            qn = np.array([np.quantile(y, q) for q in [0.25, 0.75]])
            irq = np.diff(qn)

            outliers = (k_outlier * irq) * [-1, 1] + qn #Tukey rule
            mask_outliers = (y>=outliers[0]) & (y<=outliers[1])
        elif clip_outliers=='skip':
                    mask_outliers = np.ones_like(y).astype(bool)
        else:
            raise NotImplementedError(f"clip method `{clip_outliers}` is not implemented yet")
        
        return group, mask_outliers