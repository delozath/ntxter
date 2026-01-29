from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd


def safe_kwargs(cls, **kwargs):
    """
    Filter kwargs to only those accepted by the dataclass.

    Parameters
    ----------
    cls : dataclass
        The dataclass to filter kwargs for.
    **kwargs : dict
        The keyword arguments to filter.

    Returns
    -------
    tuple of dict
        A tuple containing two dictionaries:
        - The first dictionary contains the filtered keyword arguments that match the dataclass fields.
        - The second dictionary contains the extra keyword arguments that do not match any dataclass fields.
    """
    cls_fields = {f.name for f in fields(cls)}
    params = set(kwargs.keys())

    cls_params = {k: kwargs[k] for k in params.intersection(cls_fields)}
    extra_params = {k: kwargs[k] for k in params - cls_fields}

    return cls_params, extra_params

def safe_init(cls, **kwargs):
    """
    Safely initialize a dataclass with filtered keyword arguments.

    Parameters
    ----------
    cls : dataclass
        The dataclass to initialize.
    **kwargs : dict
        The keyword arguments to filter and pass to the dataclass.
    
    Returns
    -------
    tuple
        A tuple containing:
        - instance of the dataclass initialized with filtered keyword arguments.
        - dict of extra keyword arguments that were not used in initialization.
    """
    cls_params, extra_params = safe_kwargs(cls, **kwargs)
    return cls(**cls_params), extra_params


def _check_list_str_type(
    cols: list[str] | str
  ) -> list[str]:
    """
    Check and convert input to list of strings.
    
    Parameters
    ----------
    cols : list of str or str
        Column names or column name to check.
    
    Returns
    -------
    list of str
        List of column names.
    
    Raises
    -------
    ValueError
        If input is not a string or list of strings.
    """
    col_type = type(cols).__name__
    match col_type:
        case 'str':
            cols = [cols]
        case 'list':
            pass
        case _:
            raise ValueError("cols must be a list of column names or lists of column names.")
    return cols

def _check_list_cols(df, cols: list[str] | str):
    """
    Check that all columns in cols exist in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    cols : list of str or str
        Column names or column name to check.
    
    Returns
    -------
    list of str
        List of column names.
    
    Raises
    -------
    ValueError
        If any column in cols does not exist in the DataFrame.
    """
    cols = _check_list_str_type(cols)
    diff = set(cols) - set(df.columns)
    if len(diff) != 0:
        raise ValueError("There are some columns in `cols` that are not found in DataFrame.")
    return cols

def dropna_cols(
    df: pd.DataFrame,
    cols: list[str] | str
  ) -> pd.DataFrame:
    """
    Remove rows with NaN values in specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    cols : list of str or str
        Column names or column name to check for NaN values. Rows with NaN in any of these columns will be removed.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with rows containing no NaN values in the specified columns removed.
    """
    cols = _check_list_cols(df, cols)

    return df[cols].dropna().copy()

def split_ft_cols(
    df: pd.DataFrame,
    fts: list[str] | str,

  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into feature columns and remaining columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    fts : list of str or str
        Column names or column name to select as feature columns.
    
    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing two DataFrames: the first with feature columns, the second with remaining columns.
    
    Raises
    -------
    ValueError
        If no remaining columns are left after selecting feature columns.
    """

    col_fts = _check_list_cols(df, fts)
    col_remain = list(set(df.columns) - set(col_fts))

    if len(col_remain) == 0:
        raise ValueError("No remaining columns left after selecting feature columns.")
    
    return df[col_fts].copy(), df[col_remain].copy()

def colname2index(search: list[str], cols: list[str]) -> list[int]:
    """
    Convert column names to their corresponding indices.

    Parameters
    ----------
    search : list of str
        List of column names to search for.
    cols : list of str
        List of all column names.
    
    Returns
    -------
    list of int
        List of indices corresponding to the searched column names.
    
    Raises
    -------
    ValueError
        If any column name in `search` is not found in `cols`.
    
    Notes
    -----
    Matrix multiplication is used to find indices efficiently.
    """
    if len(set(search) - set(cols)) != 0:
        raise ValueError("Some column names in `search` are not found in `cols`.")
    
    mask = np.array(search)[:, None] == cols
    return (mask @ np.arange(mask.shape[1])).tolist()


def paths_checker(pthfname: Path) -> Path:
    pthfname = Path(pthfname)
    breakpoint()