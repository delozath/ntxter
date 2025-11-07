from dataclasses import fields


import pandas as pd


def safe_kwargs(cls, **kwargs):
    cls_fields = {f.name for f in fields(cls)}
    params = set(kwargs.keys())

    cls_params = {k: kwargs[k] for k in params.intersection(cls_fields)}
    extra_params = {k: kwargs[k] for k in params - cls_fields}

    return cls_params, extra_params

def safe_init(cls, **kwargs):
    cls_params, extra_params = safe_kwargs(cls, **kwargs)
    return cls(**cls_params), extra_params


def dropna_on_lists_cols(df: pd.DataFrame, cols: list | str | int):
    if isinstance(cols, list):
        cols_ = []
        for c in cols:
            cols_ += c if isinstance(c, list) else [c]
    else:
        cols_ = [cols]     
    
    return df[cols_].dropna()