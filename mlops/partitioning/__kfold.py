import numpy as np

from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedKFold
from ntxter.validation import ArrayIndexSlice, UnpackDataAndCols

from typing import Iterator, Union, Tuple
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state

class _StratifiedKFold(BaseCrossValidator, ABC):
    EPS = 1E-4
    X_train   = ArrayIndexSlice()
    y_train   = ArrayIndexSlice()
    X_test    = ArrayIndexSlice()
    y_test    = ArrayIndexSlice()
    #
    def __init__(self, n_splits):
        self.n_splits = n_splits
    #
    def _X_y_validation(self, X, y):
        X = check_array(X, dtype=float)
        y = self._y_validation(X, y)
        return X, y
    #
    def _y_validation(self, X, y):
        n_samples, _ = X.shape
        if y is None:
            raise ValueError(f"y parameter is required for {self.__class__.__name__}")
        #
        y = check_array(y, ensure_2d=False, dtype=float)
        if len(y) != n_samples:
            raise ValueError("X and y must have the same lengths")
        #
        return y
    #
    @abstractmethod
    def split(self, X, y, groups=None):
        pass
    #
    def _wrap_assignment(self, X, y, tn_idx, tt_idx):
        self.X_train = X, tn_idx
        self.y_train = y, tn_idx
        self.X_test  = X, tt_idx
        self.y_test  = y, tt_idx
#
#
class StratifiedKFoldWrapper(_StratifiedKFold):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Union[int, None] = None
     ):
        super().__init__(n_splits)
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    #
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    #
    def split(self, X, y, groups=None):
        X, y = self._X_y_validation(X, y)
        for tn_idx, tt_idx in self.skf.split(X, y, groups):
            self._wrap_assignment(X, y, tn_idx, tt_idx)
            #
            yield tn_idx, tt_idx
#
#
class QuantileStratifiedKFold(_StratifiedKFold):
    X_train   = ArrayIndexSlice()
    y_train   = ArrayIndexSlice()
    X_test    = ArrayIndexSlice()
    y_test    = ArrayIndexSlice()
    #
    def __init__(
        self,
        n_splits: int = 5,
        n_bins: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        outliers: str = 'tukey',
        k_outlier: float = 1.5
     ):
        if n_splits < 2:
            raise ValueError("Number of splits must be at least 2.")
        if n_bins < 2:
            raise ValueError("Number of bins must be at least 2.")
        if n_bins > n_splits:
            raise ValueError("Number of bins must be lower than number of splits")
        #
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state
        self.clip_outliers = outliers
        self.k_outlier = k_outlier
    #
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    #
    def split(self, X, y, groups=None):
        for tn_idx, tt_idx in super().split(X, y, groups):
            self.X_train = X, tn_idx
            self.y_train = y, tn_idx
            self.X_test  = X, tt_idx
            self.y_test  = y, tt_idx
            #
            yield tn_idx, tt_idx
    #
    def _iter_test_masks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X, y = self._X_y_validation(X, y)
        shape = X.shape[0]
        index = np.arange(shape).astype(int)
        groups, mask_outliers = self._quantiles(y) #always overwrite groups
        #
        for _ in range(self.n_splits):
            mask = np.zeros_like(y).astype(bool)
            tt_idx = np.array([], dtype=int)
            for grp in np.unique(groups):
                grp_idx = index[grp == groups]
                #
                n_grp_idx =  grp_idx.shape[0]
                n_tt = np.ceil(n_grp_idx / self.n_splits).astype(int)
                if n_tt < 2:
                    raise ValueError("Too many splits. Balance the trade-off between them and the number of bins")
                #
                np.random.shuffle(grp_idx)
                tt_idx = np.concatenate((tt_idx, grp_idx[-n_tt:]))
            #
            mask[tt_idx] = True 
            mask &= mask_outliers
            #
            yield mask
    #
    def _quantiles(self, y):
        qn = np.array([np.quantile(y, q) for q in [0.25, 0.75]])
        irq = np.diff(qn)
        #
        percentiles = np.linspace(0, 1, self.n_bins + 1)
        percentiles = np.quantile(y, percentiles)
        percentiles[0] -= _StratifiedKFold.EPS
        group = np.searchsorted(percentiles, y)

        if self.clip_outliers=='tukey':
            outliers = (self.k_outlier * irq) * [-1, 1] + qn #Tukey rule
            mask_outliers = (y>=outliers[0]) & (y<=outliers[1])
        else:
            mask_outliers = np.ones_like(y).astype(bool)
        #
        return group, mask_outliers