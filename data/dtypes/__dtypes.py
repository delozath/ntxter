import warnings
from typing import ClassVar


import numpy as np

from abc import ABC, abstractmethod


from ntxter.validation import UnsetAttributeError, ExpectedTypeError
from ntxter.validation import SingleAssignWithType, UnpackDataAndCols, ArrayIndexSlice
from ntxter.mlops.pipeline import AbstractModelPipeline


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
class BundleMultSplitTrainTest(BundleTrainTestSplit):
    #
    def __init__(self, X, y) -> None:
        super().__init__(X, y)
        self._n_splits_count = 0
    #
    def reserve(self, index, name):
        X_tmp = self.X[index].copy()
        y_tmp = self.y[index].copy()
        setattr(self, f'X_{name}', X_tmp)
        setattr(self, f'y_{name}', y_tmp)
        #
        self.X = np.delete(self.X, index, axis=0).copy()
        self.y = np.delete(self.y, index).copy()
        #
        self._n_splits_count += 1
    #
    def set_unseen(self, index):
        self.reserve(index, 'unseen')
    #
    def split(self, n_train, n_test):
        super().split(n_train, n_test)
        if self._n_splits_count == 0:
            warnings.warn("No splits have been set, train/test partitions is performed in the whole dataset")


class ModelDescriptor:
    _SCHEMA: ClassVar[list[str]]
    descriptors = SingleAssignWithType(dict)
    
    def __init__(self, descriptors):
        self._pipeline = None
        if hasattr(self, '_SCHEMA'):
            if isinstance(descriptors, dict):
                if (set(descriptors.keys()) - set(self._SCHEMA)) or (set(self._SCHEMA) - set(descriptors.keys())):
                    raise ValueError("descriptors are not in the correct schema")
                self.descriptors = descriptors
            else:
                raise ExpectedTypeError('dict', f"{type(descriptors).__name__}") 

    @classmethod
    def register(cls, schema: list[str]) -> None:
        if isinstance(schema, list):
            if hasattr(cls, '_SCHEMA'):
                raise ValueError("Schema has been already configured")
            cls._SCHEMA = schema.copy()
        else:
            raise ValueError(f"schema has to be a list[str] of model descriptors")
    
    def __getitem__(self, key, default=None):
        return self.descriptors.get(key, default)
    
    @property
    def pipeline(self):
        if self._pipeline:
            return self._pipeline
        else:
            raise UnsetAttributeError()
    
    @pipeline.setter
    def pipeline(self, pipe):
        if isinstance(pipe, AbstractModelPipeline):
            self._pipeline = pipe
        else:
            raise TypeError(f"Pipeline must be an implementation of 'AbstractModelPipeline', but {type(pipe).__name__} was provided")

    @property
    def schema(self):
        if self._SCHEMA:
            return self._SCHEMA
        else:
            return
    
    @schema.setter
    def schema(self, value):
        raise ValueError("schema only can be assigned through classmethod 'register'")

    
