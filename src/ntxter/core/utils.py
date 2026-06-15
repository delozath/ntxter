from typing import Any
from dataclasses import fields
from pathlib import Path

from collections.abc import Iterable


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
    
    if isinstance(cols, Iterable):
        return cols if isinstance(cols, list) else list(cols)
    if isinstance(cols, (str, bool, int, float)):
        return [cols]
    else:
        raise ValueError("cols must be a list of column names or lists of column names.")

def check_list_cols(df, cols: list[str] | str):
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
        raise ValueError(f"There are some columns in `{list(diff)}` that are not found in DataFrame.")
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
    cols = check_list_cols(df, cols)

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

    col_fts = check_list_cols(df, fts)
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


def path_check(pth_fname: str, replace=False) -> Path:
    pth = Path(pth_fname)
    if not pth.parent.exists():
        pth.parent.mkdir(parents=True, exist_ok=True)
    
    if pth.exists() and not replace:
        raise FileExistsError(f"File {pth} already exists. To overwrite, set `replace=True`.")
    
    return pth

def yes_no_to_num_map(
        df: pd.DataFrame, 
        col: str | list[str],
        yes_val: int = 1,
        no_val: int = 0,
        yes_opts: str | list[str] = 'default', 
        no_opts: str | list[str] = 'default'
    ) -> pd.DataFrame:
    """Replace values in specified columns with 'Yes' and 'No' based on provided mappings.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str or list of str
        Column name or list of column names to replace values in.
    yes_opts : str or list of str, optional       
        Value(s) to replace with 'Yes'. If 'default', uses the default mapping.
    no_opts : str or list of str, optional
        Value(s) to replace with 'No'. If 'default', uses the default mapping.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with replaced values in the specified columns.
    
    Raises
    -------
    ValueError
        If any specified column does not exist in the DataFrame.
    """
    cols = [col] if isinstance(col, str) else list(col)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in DataFrame: {missing}")
    
    if yes_opts=='default':
        yes_opts = ['yes', 'y', '1', 'si', 'sí', 's', 'true', 'verdadero', 'verdadera', 'positivo', 'positiva']
    
    if no_opts=='default':
        no_opts = ['no', 'n', '0', 'false', 'f', 'falso', 'falsa', 'negativo', 'negativa']

    replace = {i: yes_val for i in yes_opts} | {i: no_val for i in no_opts}
    for s in df[cols]:
        print(f"processing {s}")
        s = (df[s]
             .str.lower()
             .str.strip()
             .str.replace(' ', '')
             .replace(replace)
            )

        if s.nunique(dropna=True) > 2:
            raise ValueError(
                f"Column '{s.name}' has more than 2 unique non-null values after replacement; "
                f"not expected for a binary column"
            )

        df = df.assign(
            **{s.name: s.astype(float) if s.isnull().any() else s.astype(int)}
         )
    
    return df

def binarize_by_zero_ref(
        df: pd.DataFrame,
        cols_zero_ref: dict[str, str],
        zero_val: int = 0,
        others: int = 1
    ):
    for col, zero_ref in cols_zero_ref.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        values = zero_ref[zero_val]
        df = df.assign(
            **{col: df[col].replace({v:zero_val for v in values})}
        )
        df.loc[df[col]!=zero_val, col] = others
        df[col] = df[col].astype(float) if df[col].isnull().any() else df[col].astype(int)
    
    return df

def check_only_n_args(n: int, /, *args, **kwargs):
        if kwargs:
            raise ValueError("Only accepts columns as a positional argument.")
        
        args_cpy = [*args]
        if args_cpy is None or len(args_cpy)!=n:
            raise ValueError("Number of args differ from expected number of arguments or None were provided")

        return args_cpy