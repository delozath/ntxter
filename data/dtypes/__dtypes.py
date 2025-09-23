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


from typing import Any, ClassVar

class SingleAssignWithType:
    def __init__(self, type_) -> None:
        self.TYPE = type_
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = '_' + name
    #
    def __get__(self, obj, objtype=None):
        if obj is None: 
            return self
        if hasattr(obj, self.private_name):
            return getattr(obj, self.private_name)
        else:
            raise ValueError(f"Attribute {self.name} have not being set")
    
    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise ValueError(f"{self.name} can only be assigned once")
        if isinstance(value, self.TYPE):
            setattr(obj, self.private_name, value)
        else:
            raise TypeError(f"Expected type for the attribute '{self.private_name[1:]}' is '{self.TYPE.__name__}', but '{type(value).__name__}'--type was provided instead")

class ModelDescriptor:
    _SCHEMA: ClassVar[list[str]]
    descriptors = SingleAssignWithType(dict)
    tmp = SingleAssignWithType(int)
    
    def __init__(self, descriptors):
        if hasattr(self, '_SCHEMA'):
            if isinstance(descriptors, dict):
                if (set(descriptors.keys()) - set(self._SCHEMA)) or (set(self._SCHEMA) - set(descriptors.keys())):
                    raise ValueError("descriptors are not in the correct schema")
                self.descriptors = descriptors
                self.tmp = 5
            else:
                raise ValueError(f"Expected dict type for descriptors parameter, {type(descriptors)} was received instead")

    @classmethod
    def register(cls, schema: list[str]) -> None:
        if isinstance(schema, list):
            if hasattr(cls, '_SCHEMA'):
                raise ValueError("Schema has been already configured")
            cls._SCHEMA = schema.copy()
        else:
            raise ValueError(f"schema has to be a list[str] of model descriptors")
    
class otra_cosa:
    @property
    def schema(self):
        if self._SCHEMA:
            return self._SCHEMA
        else:
            return
    
    @schema.setter
    def schema(self, value):
        raise ValueError("schema only can be assigned through classmethod 'register'")
    
