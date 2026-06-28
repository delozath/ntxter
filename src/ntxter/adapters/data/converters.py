from typing import Self, TypedDict, override
from dataclasses import dataclass
import pandas as pd

from ntxter.core.data.types import TidyDataFrameRetriever
from ntxter.core.data.converter import TidyTable


@dataclass
class PandasTidyFrameRetriever(TidyDataFrameRetriever[pd.DataFrame]):
    @override
    def _data_type_check(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
    
    @override
    def _colunm_names_check(self) -> None:
        super()._colunm_names_check()
        cols =  self.cols + self.id_vars
        test_cols = [i in self.data for i in cols]
        if not all(test_cols):
            diff = [c for c, test in zip(cols, test_cols) if not test]
            raise KeyError(f"Column specified in `cols` and `id_vars` must be valid colunm names in `data`. Missing columns: {str(diff)}")


class PandasTidyTable(TidyTable[pd.DataFrame]):
    def __init__(self, data, cols, id_vars, query=""):
        self.df_rtv = PandasTidyFrameRetriever(data=data, cols=cols, id_vars=id_vars, query=query)

    @classmethod
    def compose(cls, *args, **kwargs) -> pd.DataFrame | Self:
        if args:
            inst = cls(*args)
        elif kwargs:
            inst = cls(**kwargs)
        else:
            raise ValueError("No arguments or keyword arguments provided. PandasTidyFrameRetriever cannot be instantiated")
        
        return inst
    
    def query_tidy(self, add_cols: list | None = None):
        cols = self.id_vars + self.cols
        id_vars = self.id_vars

        if add_cols is not None:
            if isinstance(add_cols, str):
                cols = cols + [add_cols]
                id_vars = self.id_vars + [add_cols]
            elif not isinstance(add_cols, list):
                raise TypeError("`add_cols` must be a list or str")
            else:
                cols = cols + add_cols
                id_vars = id_vars + add_cols
        
        if self.query!="":
            return (
                    self.data.query(self.query)[cols]
                             .melt(id_vars=id_vars)
                )
        else:
            return (
                self.data[cols]
                    .melt(id_vars=id_vars)
            )
