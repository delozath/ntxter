from dataclasses import fields


import pandas as pd


def split_dataclass_kwargs(dclass, **kwargs):
    dclass_fields = {f.name for f in fields(dclass)}
    params = set(kwargs.keys())

    dclass_params = {k: kwargs[k] for k in params.intersection(dclass_fields)}
    another_params = {k: kwargs[k] for k in params - dclass_fields}

    return dclass(**dclass_params), another_params


def dropna_on_lists_cols(df: pd.DataFrame, cols: list | str | int):
    if isinstance(cols, list):
        cols_ = []
        for c in cols:
            cols_ += c if isinstance(c, list) else [c]
    else:
        cols_ = [cols]     
    
    return df[cols_].dropna()