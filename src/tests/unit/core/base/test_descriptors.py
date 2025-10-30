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


#[TEST]
#UnpackDataAndCols
@pytest.fixture
def unpack_data_and_cols():
    class Dummy:
        test_descr = descriptors.UnpackDataAndCols()

        def __init__(self) -> None:
            N = 25
            self.data_test = np.arange(200).reshape(N, -1, order='F')
            self.index_test = np.arange(N)

    return Dummy()

def test_unpack_data_and_cols_df(unpack_data_and_cols):
    import string
    
    letter = list(string.ascii_lowercase)
    np.random.shuffle(letter)

    data = pd.DataFrame(unpack_data_and_cols.data_test)
    data.columns = [f'cols_test_{l}_{c + 1}' for l, c in zip(letter, data.columns)]
    
    unpack_data_and_cols.test_descr = data
    X_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(X_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert X_test.shape == data.shape
    assert len(data.columns) == len(colnames_test)
    assert (X_test != data.values).sum() == 0
    assert all(data.columns == colnames_test)

def test_unpack_data_and_cols_unnamed_serie(unpack_data_and_cols):
    data = unpack_data_and_cols.data_test
    COL = np.random.randint(data.shape[-1])

    serie = pd.Series(data[:, COL])
    unpack_data_and_cols.test_descr = serie

    x_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(x_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert x_test.shape == data[:, COL].shape
    assert len(colnames_test) == 1
    assert all(x_test == serie.values)
    assert all(x_test == data[:, COL])
    assert colnames_test[0] == 'default'

def test_unpack_data_and_cols_named_serie(unpack_data_and_cols):
    import string

    letters = list(string.ascii_lowercase)

    data = unpack_data_and_cols.data_test
    COL = np.random.randint(data.shape[-1])
    test_name = ''.join(np.random.choice(letters, 10))

    serie = pd.Series(data[:, COL], name=test_name)
    unpack_data_and_cols.test_descr = serie

    x_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(x_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert x_test.shape == data[:, COL].shape
    assert len(colnames_test) == 1
    assert all(x_test == serie.values)
    assert all(x_test == data[:, COL])
    assert colnames_test[0] == test_name

def test_unpack_data_and_cols_np_long(unpack_data_and_cols):
    data = unpack_data_and_cols.data_test
    
    unpack_data_and_cols.test_descr = data
    X_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(X_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert X_test.shape == data.shape
    assert data.shape[-1] == len(colnames_test)
    assert (X_test != data).sum() == 0
    assert colnames_test == list(range(data.shape[-1]))

def test_unpack_data_and_cols_np_wide(unpack_data_and_cols):
    data = unpack_data_and_cols.data_test.T
    
    unpack_data_and_cols.test_descr = data
    X_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(X_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert X_test.shape == data.shape
    assert data.shape[-1] == len(colnames_test)
    assert (X_test != data).sum() == 0
    assert all(colnames_test == data[0])

def test_unpack_data_and_cols_np_1d(unpack_data_and_cols):
    #[TEST]
    #select randomly X or X.T to test independence 1d np.ndarray
    if np.random.randint(2)==0:
        data = unpack_data_and_cols.data_test.T
    else:
        data = unpack_data_and_cols.data_test
    
    COL = np.random.randint(data.shape[-1])
    
    unpack_data_and_cols.test_descr = data[:, COL]
    X_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(X_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert X_test.shape == data[:, COL].shape
    assert len(colnames_test) == 1
    assert (X_test != data[:, COL]).sum() == 0
    assert colnames_test[0] == 0

def test_unpack_data_and_cols_np_multidim(unpack_data_and_cols):
    tmp = unpack_data_and_cols.data_test
    data = np.concatenate(
        (
            tmp[:, :, None],
            -tmp[:, :, None],
            tmp[:, :, None]**2, 2*tmp[:, :, None],
            np.sqrt(tmp)[:, :, None]
         ), axis=-1
     )
    
    unpack_data_and_cols.test_descr = data
    X_test, colnames_test = unpack_data_and_cols.test_descr

    assert isinstance(X_test, np.ndarray)
    assert isinstance(colnames_test, list)

    assert X_test.shape == data.shape
    assert len(colnames_test) == data.shape[-1]
    assert (X_test != data).sum() == 0
    assert colnames_test == list(range(data.shape[-1]))

def test_unpack_data_and_cols_validations(unpack_data_and_cols):
    data = unpack_data_and_cols.data_test
    COL = np.random.randint(data.shape[-1])
    
    unpack_data_and_cols.test_descr = None
    assert unpack_data_and_cols.test_descr is None

    with pytest.raises(ValueError):
         unpack_data_and_cols.test_descr = list(data[:, COL])
         unpack_data_and_cols.test_descr = 5
         unpack_data_and_cols.test_descr = 3.1416


#[TEST]
#UniversalGetterSetter
#[PASSED]
def test_UniversalGetterSetter():
    value1 = 10
    value2 = [1, 5, 7]

    class Dummy():
        test_descr = descriptors.SetterAndGetter()
    
    instance = Dummy()
    with pytest.raises(ValueError):
        instance.test_descr
    
    instance.test_descr = value1
    assert instance.test_descr == value1

    instance.test_descr = value2
    assert instance.test_descr == value2

    instance.test_descr = pd.DataFrame(value2)
    assert (instance.test_descr != pd.DataFrame(value2)).sum().sum() == 0


#[NOTE]
# add new descriptor tests here to control
# the tested classes 
print("descriptors.py successfully tested")