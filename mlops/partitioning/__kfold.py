import numpy as np

from sklearn.model_selection import StratifiedKFold
from ntxter.validation import ArrayIndexSlice, UnpackDataAndCols

from typing import Iterator, Union, Tuple
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state

class _AbstractStratified:
    def _X_y_validation(self, X, y):
        n_samples, _ = X.shape
        if y is None:
            raise ValueError(f"y parameter is required for {self.__class__.__name__}")
        #
        y = check_array(y, ensure_2d=False, dtype=float)
        if len(y) != n_samples:
            raise ValueError("X and y must have the same lengths")

class StratifiedKFoldWrapper(StratifiedKFold, _AbstractStratified):
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
        self.X = X
        self.y = y
        #
        self._X_y_validation(X, y)
    #
    def split(self, groups=None):
        for tn_idx, tt_idx in super().split(self.X, self.y, groups):
            self.X_train = self.X, tn_idx
            self.y_train = self.y, tn_idx
            self.X_test  = self.X, tt_idx
            self.y_test  = self.y, tt_idx
            yield
#
#
class QuantileStratifiedKFold(BaseCrossValidator, _AbstractStratified):
    def __init__(
        self,
        n_splits: int = 5,
        n_bins: int = 5,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        clip_outliers: bool = True,
     ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        #
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state
        self.clip_outliers = clip_outliers
    #
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        # TODO: devolver el número de particiones, o calcularlo dinámicamente si aplica
        pass
    #
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self._check_parameters(X, y)
        #    Debe generar pares (train_idx, test_idx) como arrays de enteros.
        
            # TODO: validar entradas (longitud de X, consistencia con y y groups)
            # TODO: generar índice base (np.arange(n_samples))
            # TODO: implementar tu lógica personalizada de división
            # TODO: barajar si shuffle=True usando check_random_state
            # TODO: yield train_idx, test_idx en cada partición
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