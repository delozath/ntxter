from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import check_array


class BaseQuantileStratifiedKFold(ABC, BaseCrossValidator):
    """Shared scikit-learn implementation for quantile-based splitters."""

    EPS = 1E-4

    @abstractmethod
    def _iter_test_masks(self, X, y, **kwargs) -> Generator:
        ...

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        X, y = self._validations(X, y)
        yield from super().split(X, y, groups)

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

    def quantiles(self, y, n_bins):
        percentiles = np.linspace(0, 1, n_bins + 1)
        percentiles = np.quantile(y, percentiles)
        percentiles[0] -= self.EPS
        return np.searchsorted(percentiles, y)
