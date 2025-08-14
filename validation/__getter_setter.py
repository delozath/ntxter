import numpy as np
import pandas as pd

class _AbstractGetter:
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = '_' + name
    #
    def __get__(self, obj, objtype=None):
        if hasattr(obj, self.private_name):
            return getattr(obj, self.private_name)
        #
        else:
            raise ValueError(f"Attribute {self.name} have not being set")
#
#  
class ArrayIndexSlice(_AbstractGetter):
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
class UnpackDataAndCols(_AbstractGetter):
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
        #
        else:
            raise ValueError(f"Attribute {value} must be pd.DataFrame | pd.Series | np.ndarray")