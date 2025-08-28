import numpy as np

from sklearn.model_selection import StratifiedKFold
from ntxter.validation import ArrayIndexSlice, UnpackDataAndCols

from typing import Iterator, Union, Tuple
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state

class _StratifiedKFold:
    EPS = 1E-4
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
#
class StratifiedKFoldWrapper(StratifiedKFold, _StratifiedKFold):
    X_train   = ArrayIndexSlice()
    y_train   = ArrayIndexSlice()
    X_test    = ArrayIndexSlice()
    y_test    = ArrayIndexSlice()
    #X_unseen  = ArrayIndexSlice()
    #y_unseen  = ArrayIndexSlice()
    #
    def __init__(
        self,
        X: np.ndarray, 
        y: np.ndarray,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Union[int, None] = None
     ):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.X, self.y = self._X_y_validation(X, y)
    #
    def split(self, X, y, groups=None):
        for tn_idx, tt_idx in super().split(self.X, self.y, groups):
            self.X_train = self.X, tn_idx
            self.y_train = self.y, tn_idx
            self.X_test  = self.X, tt_idx
            self.y_test  = self.y, tt_idx
            #
            yield
#
#
#TODO: incorporate StratifiedGroupKFold
class QuantileStratifiedKFold(BaseCrossValidator, _StratifiedKFold):
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
        # TODO: devolver el número de particiones, o calcularlo dinámicamente si aplica
        pass
    #
    def _iter_test_masks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X, y = self._X_y_validation(X, y)
        group, mask_outliers =self._quantiles(y)
        #
        if mask_outliers is not None:
            X = X[mask_outliers]
            y = y[mask_outliers]
            group = group[mask_outliers]
        #
        for _ in range(self.n_splits):
            tn_idx, tt_idx = np.array([]), np.array([])
            for grp in np.unique(group):
                X_ = X[group == grp]
                #
                x_shp =  X_.shape[0]
                n_tt = np.ceil(x_shp / self.n_splits).astype(int)
                if n_tt < 2:
                    raise ValueError("Too many splits. Balance the trade-off between them and the number of bins")
                #
                idx = np.arange(x_shp)
                np.random.shuffle(idx)
                #
                tn_idx = np.concatenate((tn_idx, idx[:-n_tt]))
                tt_idx = np.concatenate((tt_idx, idx[-n_tt:]))
            #
            yield tn_idx, tt_idx
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
            mask_outliers = None
        #
        return group, mask_outliers
    #
    def tmp(self):
        # Build quantile edges; guarantee strictly increasing edges.
        # We handle duplicates by nudging edges minimally.
        qs = np.linspace(0, 1, num=self.n_bins + 1)
        edges = np.quantile(y, qs)

        # Optional clipping of extreme outliers to the interior quantiles
        if self.clip_outliers:
            qmin, qmax = edges[1], edges[-2]  # ignore absolute extremes
            y_clipped = np.clip(y, qmin, qmax)
        else:
            y_clipped = y