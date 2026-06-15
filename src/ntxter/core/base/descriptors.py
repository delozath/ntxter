"""
Data descriptors for attribute management with validation, type enforcement,
and automatic private name handling.

This module provides a collection of Python descriptor classes that can be used
as class attributes to control how values are get, set, and validated. Descriptors
are useful to enforce types, restrict assignment, and transform input data
transparently at the attribute level.

Classes
-------
SetPrivateNameAndGetter
    Base descriptor providing automatic private name registration and getter logic.
SetterAndGetter
    Descriptor with unrestricted get/set behavior using private storage.
SetterAndGetterType
    Descriptor that enforces a specific type on assignment.
SingleAssignNoType
    Descriptor that allows assignment only once (no type check).
SingleAssignWithType
    Descriptor that allows assignment only once and enforces a specific type.
ArrayIndexSlice
    Descriptor that accepts a (values, index) tuple and stores the indexed slice.
UnpackDataAndCols
    Descriptor that unpacks a DataFrame, Series, or ndarray into (values, col_names).

Examples
--------
>>> from ntxter.core.base.descriptors import SetterAndGetterType, SingleAssignWithType

>>> class Model:
...     learning_rate = SetterAndGetterType(float)
...     name = SingleAssignWithType(str)

>>> m = Model()
>>> m.learning_rate = 0.01
>>> m.name = "my_model"
>>> m.learning_rate
0.01
>>> m.name
'my_model'
"""

import numpy as np
import pandas as pd


class SetPrivateNameAndGetter:
    """
    Base descriptor that registers a private attribute name and provides getter logic.

    This class implements ``__set_name__`` to automatically derive a private storage
    name (``'_' + name``) and ``__get__`` to retrieve the stored value. It is
    intended to be subclassed; subclasses provide ``__set__`` implementations.

    Attributes
    ----------
    name : str
        The public attribute name as declared in the owner class.
    private_name : str
        The private storage name, equal to ``'_' + name``.

    Notes
    -----
    When accessed from the class (``obj is None``), returns the descriptor itself,
    which is standard Python descriptor protocol behavior.

    Raises
    ------
    ValueError
        If the attribute has not been set and ``__get__`` is called on an instance.

    Examples
    --------
    >>> class MyDescriptor(SetPrivateNameAndGetter):
    ...     def __set__(self, obj, value):
    ...         setattr(obj, self.private_name, value)

    >>> class Foo:
    ...     x = MyDescriptor()

    >>> f = Foo()
    >>> f.x = 42
    >>> f.x
    42
    >>> Foo.x  # returns the descriptor itself
    <MyDescriptor object>
    """

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if hasattr(obj, self.private_name):
            return getattr(obj, self.private_name)
        else:
            raise ValueError(f"Attribute {self.name} have not being set")


class SetterAndGetter(SetPrivateNameAndGetter):
    """
    Descriptor with unrestricted get and set behavior using private attribute storage.

    Extends ``SetPrivateNameAndGetter`` with a plain ``__set__`` that stores any
    value without validation. The value is stored under the private name derived
    from ``__set_name__``.

    Examples
    --------
    >>> class Config:
    ...     value = SetterAndGetter()

    >>> c = Config()
    >>> c.value = [1, 2, 3]
    >>> c.value
    [1, 2, 3]
    """

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class SetterAndGetterType(SetPrivateNameAndGetter):
    """
    Descriptor that enforces a specific type on every assignment.

    Parameters
    ----------
    expected_type : type
        The type (or tuple of types) that the attribute value must be an instance of.

    Attributes
    ----------
    _TYPE : type
        The expected type passed at construction time.

    Raises
    ------
    TypeError
        If the assigned value is not an instance of ``expected_type``.

    Examples
    --------
    >>> class Params:
    ...     n_estimators = SetterAndGetterType(int)
    ...     learning_rate = SetterAndGetterType(float)

    >>> p = Params()
    >>> p.n_estimators = 100
    >>> p.learning_rate = 0.05
    >>> p.n_estimators = "bad"
    TypeError: Expected type for the attribute 'n_estimators' is 'int',
               but 'str'--type was provided instead
    """

    def __init__(self, expected_type) -> None:
        self._TYPE = expected_type

    def __set__(self, obj, value):
        if isinstance(value, self._TYPE):
            setattr(obj, self.private_name, value)
        else:
            raise TypeError(
                f"Expected type for the attribute '{self.private_name[1:]}' is "
                f"'{self._TYPE.__name__}', but '{type(value).__name__}'--type was "
                f"provided instead"
            )


class SingleAssignNoType(SetPrivateNameAndGetter):
    """
    Descriptor that allows assignment only once, without type checking.

    Once a value has been set on an instance, any subsequent assignment raises
    a ``ValueError``. No type enforcement is applied.

    Raises
    ------
    ValueError
        If an attempt is made to assign a value after the attribute has already
        been set on the instance.

    Examples
    --------
    >>> class Pipeline:
    ...     run_id = SingleAssignNoType()

    >>> p = Pipeline()
    >>> p.run_id = "abc-123"
    >>> p.run_id
    'abc-123'
    >>> p.run_id = "xyz-456"
    ValueError: run_id can only be assigned once
    """

    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise ValueError(f"{self.name} can only be assigned once")
        setattr(obj, self.private_name, value)


class SingleAssignWithType(SetPrivateNameAndGetter):
    """
    Descriptor that allows assignment only once and enforces a specific type.

    Combines the single-assignment semantics of ``SingleAssignNoType`` with the
    type validation of ``SetterAndGetterType``.

    Parameters
    ----------
    type_ : type
        The type (or tuple of types) that the attribute value must be an instance of.

    Attributes
    ----------
    TYPE : type
        The expected type passed at construction time.

    Raises
    ------
    ValueError
        If an attempt is made to re-assign the attribute after it has been set.
    TypeError
        If the assigned value is not an instance of ``type_``.

    Examples
    --------
    >>> class Experiment:
    ...     dataset_path = SingleAssignWithType(str)

    >>> e = Experiment()
    >>> e.dataset_path = "/data/train.csv"
    >>> e.dataset_path
    '/data/train.csv'
    >>> e.dataset_path = "/data/other.csv"
    ValueError: dataset_path can only be assigned once
    >>> e2 = Experiment()
    >>> e2.dataset_path = 42
    TypeError: Expected type for the attribute 'dataset_path' is 'str',
               but 'int'--type was provided instead
    """

    def __init__(self, type_) -> None:
        self.TYPE = type_

    def __set__(self, obj, value):
        if hasattr(obj, self.private_name):
            raise ValueError(f"{self.name} can only be assigned once")
        if isinstance(value, self.TYPE):
            setattr(obj, self.private_name, value)
        else:
            raise TypeError(
                f"Expected type for the attribute '{self.private_name[1:]}' is "
                f"'{self.TYPE.__name__}', but '{type(value).__name__}'--type was "
                f"provided instead"
            )


class ArrayIndexSlice(SetPrivateNameAndGetter):
    """
    Descriptor that accepts a (values, index) tuple and stores only the indexed rows.

    The descriptor expects a 2-element tuple where the first element is a
    ``numpy.ndarray`` of values and the second is a list or ``numpy.ndarray``
    of integer indices. It stores the result of ``values[index]`` rather than
    the original tuple.

    Class Attributes
    ----------------
    VALUES : int
        Positional constant for the values array in the input tuple (``0``).
    INDEX : int
        Positional constant for the index array in the input tuple (``1``).

    Raises
    ------
    ValueError
        If the input is not a tuple of the form ``(np.ndarray, list | np.ndarray)``.
    IndexError
        If any index value exceeds the number of rows in the values array.

    Examples
    --------
    >>> import numpy as np

    >>> class Dataset:
    ...     subset = ArrayIndexSlice()

    >>> d = Dataset()
    >>> arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> d.subset = (arr, [0, 2])
    >>> d.subset
    array([[1, 2],
           [5, 6]])
    """

    VALUES = 0
    INDEX = 1

    def __set__(self, obj, value):
        if (
            isinstance(value, tuple) and
            isinstance(value[ArrayIndexSlice.VALUES], np.ndarray) and
            isinstance(value[ArrayIndexSlice.INDEX], (list, np.ndarray))
        ):
            if max(value[ArrayIndexSlice.INDEX]) > value[ArrayIndexSlice.VALUES].shape[0]:
                raise IndexError('Index out of range')

            setattr(
                obj,
                self.private_name,
                value[ArrayIndexSlice.VALUES][value[ArrayIndexSlice.INDEX]]
            )

        else:
            raise ValueError(
                f"Attribute {value} must be a tuple: (np.ndarray, list | np.ndarray)"
            )


class UnpackDataAndCols(SetPrivateNameAndGetter):
    """
    Descriptor that unpacks tabular data into a ``(values, col_names)`` tuple.

    Accepts a ``pandas.DataFrame``, ``pandas.Series``, ``numpy.ndarray``, or
    ``None``. Regardless of the input type, the attribute stores a 2-element
    tuple: ``(numpy array of values, list of column names)``. When ``None`` is
    assigned, ``None`` is stored directly.

    Class Attributes
    ----------------
    VALS : int
        Positional constant for the values array in the stored tuple (``0``).
    COLS : int
        Positional constant for the column names list in the stored tuple (``1``).

    Notes
    -----
    - ``pd.DataFrame``: stores ``(df.values, df.columns.tolist())``.
    - ``pd.Series``: stores ``(series.to_numpy(), [series.name or 'default'])``.
    - ``np.ndarray`` (1-D or single-column): flattened; column names are ``[0]``.
    - ``np.ndarray`` (2-D multi-column): column names are ``[0, 1, ..., n-1]``.
    - ``None``: stores ``None``.

    Raises
    ------
    ValueError
        If the assigned value is not one of the accepted types.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd

    >>> class Transformer:
    ...     data = UnpackDataAndCols()

    >>> t = Transformer()

    >>> # From DataFrame
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> t.data = df
    >>> t.data
    (array([[1, 3], [2, 4]]), ['a', 'b'])

    >>> # From ndarray
    >>> t2 = Transformer()
    >>> t2.data = np.array([[10, 20], [30, 40]])
    >>> t2.data
    (array([[10, 20], [30, 40]]), [0, 1])

    >>> # From Series
    >>> t3 = Transformer()
    >>> t3.data = pd.Series([5, 6, 7], name="score")
    >>> t3.data
    (array([5, 6, 7]), ['score'])
    """

    VALS = 0
    COLS = 1

    def __set__(self, obj, value):
        if isinstance(value, pd.DataFrame):
            col_names = value.columns.to_list()
            setattr(obj, self.private_name, (value.values, col_names))
        #
        elif isinstance(value, pd.Series):
            col_names = value.name if value.name is not None else 'default'
            setattr(obj, self.private_name, (value.to_numpy(), [col_names]))
        #
        elif isinstance(value, np.ndarray):
            ndim = value.ndim
            ncols = value.shape[-1] if ndim > 1 else 1
            #
            col_names = list(range(ncols))
            #
            if (ndim == 1 or (ncols == 1)):
                value = value.flatten()
            #
            setattr(obj, self.private_name, (value, col_names))
        elif value is None:
            setattr(obj, self.private_name, None)
        #
        else:
            raise ValueError(
                f"Attempts to assign {type(value)} type, use "
                f"pd.DataFrame | pd.Series | np.ndarray | None, instead"
            )


class RegistryFunctionDescriptor:
    def __init__(self) -> None:
        self.private_name: None | str = None 
    
    def __set_name__(self, owner, name):
        self.private_name = "_" + name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, self.private_name):
            setattr(instance, self.private_name, {})
        
        hidden_dict = getattr(instance, self.private_name)

        return KeyLookup(hidden_dict)
    
    def __set__(self, instance, value) -> None:
        raise AttributeError("registry is read-only.")

    def __delete__(self, instance) -> None:
        raise AttributeError("registry is read-only.")
        

class KeyLookup:
    def __init__(self, hidden_dict: dict) -> None:
        self._hidden_dict = hidden_dict
    
    def __getitem__(self, key):
        if not key in self._hidden_dict:
            raise KeyError(f"Key `{key}` not found in registry")
        
        return self._hidden_dict[key]
    
    def __set__(self, value) -> None:
        raise AttributeError("Cannot set value in registry, use `register` method instead.")
    
    def __repr__(self) -> str:
        return f"<View of {list(self._hidden_dict.keys())} keys>"
    
    def items(self):
        return self._hidden_dict.items()
    
    def register(self, key: str):
        def wrapper(func):
            if key in self._hidden_dict:
                raise ValueError(f"Key `{key}` already registered")
            if not callable(func):
                raise TypeError("Dict.value to registry must be a callable function")
            self._hidden_dict[key] = func
            return func
        return wrapper