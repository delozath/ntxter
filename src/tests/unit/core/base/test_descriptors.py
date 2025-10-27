from unittest.mock import MagicMock
import pytest

import numpy as np
import pandas as pd

from ntxter.core.base import descriptors

def test_set_private_name_and_getter_get_value():
    class Dummy:
        test_x = descriptors.SetPrivateNameAndGetter()
    
    instance = Dummy()
    instance.test_x = 99
    assert instance.test_x == 99

def test_set_private_name_and_getter_not_assigned():
    class Dummy:
        test_x = descriptors.SetPrivateNameAndGetter()

    instance = Dummy()
    with pytest.raises(ValueError):
        instance.test_x



"""
class _GeneralGetterAndSetPrivateName:
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


class SingleAssignNoType(_GeneralGetterAndSetPrivateName):   
    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise ValueError(f"{self.name} can only be assigned once")
        setattr(obj, self.private_name, value)


class SingleAssignWithType(_GeneralGetterAndSetPrivateName):
    def __init__(self, type_) -> None:
        self.TYPE = type_
    
    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise ValueError(f"{self.name} can only be assigned once")
        if isinstance(value, self.TYPE):
            setattr(obj, self.private_name, value)
        else:
            raise TypeError(f"Expected type for the attribute '{self.private_name[1:]}' is '{self.TYPE.__name__}', but '{type(value).__name__}'--type was provided instead")


class ArrayIndexSlice(_GeneralGetterAndSetPrivateName):
    VALUES = 0
    INDEX = 1
    def __set__(self, obj, value):
        if (
            isinstance(value      , tuple     ) and
            isinstance(value[ArrayIndexSlice.VALUES], np.ndarray) and
            isinstance(value[ArrayIndexSlice.INDEX], (list, np.ndarray))
         ):
            setattr(
                obj,
                self.private_name,
                value[ArrayIndexSlice.VALUES][value[ArrayIndexSlice.INDEX]]
             )
        #
        else:
            raise ValueError(f"Attribute {value} must be a tuple: (np.ndarray, list | np.ndarray)")
#
#
class UnpackDataAndCols(_GeneralGetterAndSetPrivateName):
    VALS = 0
    COLS = 1
    def __set__(self, obj, value):
        if isinstance(value, pd.DataFrame):
            col_names = value.columns.to_list()
            setattr(obj, self.private_name, (value.values, col_names))
        #
        elif isinstance(value, pd.Series):
            col_names = value.name
            setattr(obj, self.private_name, (value.to_numpy(), [col_names]))
        #
        elif isinstance(value, np.ndarray):
            ndim = value.ndim
            ncols = value.shape[-1] if ndim > 1 else 1
            # 
            col_names = list(range(ncols))
            #
            if (ndim==1 or (ncols==1)):
                value = value.flatten()
            #
            setattr(obj, self.private_name, (value, col_names))
        elif value is None:
                setattr(obj, self.private_name, None)
        #
        else:
            raise ValueError(f"Attribute {value} must be pd.DataFrame | pd.Series | np.ndarray | None")
"""