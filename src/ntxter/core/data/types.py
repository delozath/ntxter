import warnings
from typing import Dict, List, Any, Protocol, Type, override
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict, is_dataclass

import pickle
import pandas as pd


import numpy as np
import pandas as pd
import polars as pl
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
    
    def to_pickle(self, path: str | Path | None = None):
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            return pickle.dumps(self)


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


class EstimatorProtocol(Protocol):
    def fit(self, X: np.ndarray | pd.DataFrame | pl.DataFrame, y: np.ndarray | pd.Series) -> Any: ...
    def predict(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> np.ndarray | pd.Series | List: ...

class PipelineProtocol(Protocol):
    def fit(self, X: np.ndarray | pd.DataFrame | pl.DataFrame, y: np.ndarray | pd.Series) -> Any: ...
    def predict(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> np.ndarray | pd.Series | List: ...
    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame: ...

@dataclass
class BasePipelineStage[P]:
    """A base class for a pipeline stage. Holds the estimator type and its parameters.
    
    Parameters
    ----------
    stage : str
        The name of the pipeline stage.
    estimator : Type[EstimatorProtocol] | EstimatorProtocol
        The instance of the estimator or a class type that implements the EstimatorProtocol.
    params : Dict | P
        The parameters to initialize the estimator. P is a generic type for specific parameter dataclasses.

    Returns
    -------
    None

    Notes
    -----
    Type[EstimatorProtocol] is an abstraction for any estimator class that implements fit and predict methods. For example, sklearn Pipeline.
    """
    stage: str
    estimator: Type[EstimatorProtocol] | EstimatorProtocol
    params: Dict | P = field(default_factory=dict)

    def __post_init__(self):
        if is_dataclass(self.params):
            self.params = asdict(self.params)
        
        if isinstance(self.estimator, type):
            self.estimator = self.estimator(**self.params)


@dataclass
class TidyDataFrameRetriever[T](ABC):
    data: T
    cols: list[str] | str
    id_vars: list[str] | str
    query: str = ""

    def __post_init__(self):
        self._data_type_check()

        if isinstance(self.cols, str):
            self.cols = [self.cols]
        
        if isinstance(self.id_vars, str):
            self.id_vars = [self.id_vars]
        
        if not isinstance(self.cols, list):
            raise TypeError("cols must be a string or a list of strings.")
        
        if not isinstance(self.id_vars, list):
            raise TypeError("`id_vars` must be a string or a list of strings.")
        
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")

        self._colunm_names_check()

    @abstractmethod
    def _data_type_check(self) -> None:
        raise NotImplementedError("Method `_data_type_check` must be implemented")
    
    @abstractmethod
    def _colunm_names_check(self):
        inters_cols = set(self.cols).intersection(self.id_vars)
        if len(inters_cols)>0:
            raise ValueError("There are common columns between `cols` and `id_vars should be mutually exclusive.")
    
