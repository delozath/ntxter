from unittest.mock import MagicMock
import pytest

import numpy as np
import pandas as pd

from ntxter.core.base import descriptors

#[TEST]
#descriptors.SetPrivateNameAndGetter
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


#[TEST]
#descriptors.SingleAssignNoType
@pytest.fixture
def single_assign_notype():
    class Dummy():
        test_descr = descriptors.SingleAssignNoType()
    return Dummy()

def test_single_assign_no_type_case1(single_assign_notype):
    single_assign_notype.test_descr = 'hola'
    assert single_assign_notype.test_descr == 'hola'
    assert single_assign_notype._test_descr == single_assign_notype.test_descr
    with pytest.raises(ValueError, match='test_descr can only be assigned once'):
        single_assign_notype.test_descr = 'adios'


def test_single_assign_no_type_case2(single_assign_notype):
    df = pd.DataFrame()
    single_assign_notype.test_descr = df
    assert single_assign_notype.test_descr is df
    assert single_assign_notype._test_descr is single_assign_notype.test_descr

    with pytest.raises(ValueError, match='test_descr can only be assigned once'):
        single_assign_notype.test_descr = -56884
        single_assign_notype.test_descr = 'aaaalo'
        single_assign_notype.test_descr = 2j + 5


def test_single_assign_no_type_case3(single_assign_notype):
    single_assign_notype.test_descr = 3.1416
    assert single_assign_notype.test_descr == 3.1416
    assert single_assign_notype._test_descr == single_assign_notype.test_descr

    with pytest.raises(ValueError, match='test_descr can only be assigned once'):
        single_assign_notype.test_descr = 0


#[TEST]
#SingleAssignWithType
def test_single_assign_with_type_int():
    value = 10

    class Dummy():
        test_descr = descriptors.SingleAssignWithType(type(value))
    
    instance = Dummy()
    with pytest.raises(ValueError, match='Attribute test_descr have not being set'):
        instance.test_descr
    
    with pytest.raises(TypeError):
        instance.test_descr = str(value)
        instance.test_descr = float(value)
        instance.test_descr = pd.DataFrame([value])

    instance.test_descr = value
    assert instance.test_descr == value

    with pytest.raises(ValueError, match='test_descr can only be assigned once'):
        instance.test_descr = 2 * value
        instance.test_descr = 'a'


def test_single_assign_with_type_df():
    value = pd.DataFrame([1, 2, 3])

    class Dummy():
        test_descr = descriptors.SingleAssignWithType(type(value))
    
    instance = Dummy()
    with pytest.raises(ValueError, match='Attribute test_descr have not being set'):
        instance.test_descr
    
    with pytest.raises(TypeError):
        instance.test_descr = str(value)
        instance.test_descr = 3.5
        instance.test_descr = 2

    instance.test_descr = value
    assert isinstance(instance.test_descr, pd.DataFrame)
    assert instance.test_descr is value

    fake_value = pd.DataFrame([1, 2, 3])
    assert not instance.test_descr is fake_value
    
    with pytest.raises(ValueError, match='test_descr can only be assigned once'):
        instance.test_descr = 2 * value
        instance.test_descr = 'a'
        instance.test_descr = fake_value


#[TEST]
#ArrayIndexSlice
@pytest.fixture
def array_index_slice():
    class Dummy:
        test_descr = descriptors.ArrayIndexSlice()
    
    return Dummy()

def test_array_index_slice_assign_ordered(array_index_slice):
    data = np.arange(200).reshape(25, -1, order='F')
    index = np.arange(15)

    array_index_slice.test_descr = (data, index)
    assert all(array_index_slice.test_descr[:, 0] == index)

def test_array_index_slice_assign_unordered(array_index_slice):
    data = np.arange(200).reshape(25, -1, order='F')
    index = np.arange(25)
    choice = np.random.choice(index, 20, replace=False).copy()

    array_index_slice.test_descr = (data, choice)
    assert all(array_index_slice.test_descr[:, 0] == choice)

def test_array_index_slice_multiple_assigns(array_index_slice):
    data = np.arange(200).reshape(25, -1, order='F')
    index = np.arange(25)

    choice = np.random.choice(index, 20, replace=False).copy()
    choice = list(choice)
    array_index_slice.test_descr = (data, choice)
    assert all(array_index_slice.test_descr[:, 0] == choice)

    choice = np.random.choice(index, 10, replace=False).copy()
    choice = list(choice)
    array_index_slice.test_descr = (data, choice)
    assert all(array_index_slice.test_descr[:, 0] == choice)
    
def test_array_index_slice_validations(array_index_slice):
    data = np.arange(200).reshape(25, -1, order='F')
    index = np.arange(25)
    with pytest.raises(ValueError):
        array_index_slice.test_descr = data
        array_index_slice.test_descr = (pd.DataFrame(data), index)
        array_index_slice.test_descr = (data, )
        array_index_slice.test_descr = (data, 25)
        array_index_slice.test_descr = (32, 32)
    
    with pytest.raises(IndexError):
        array_index_slice.test_descr = (data, np.arange(26)[::-1][:10])
        array_index_slice.test_descr = (data, np.arange(50)[::-1][:15])
        array_index_slice.test_descr = (data, list(index) + [50])
    #breakpoint()

"""
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