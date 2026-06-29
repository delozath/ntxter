from typing import Self, TypedDict, override
from dataclasses import dataclass
import pandas as pd

from ntxter.core.data.types import QueryContainer
from ntxter.core.data.converter import QueryTable


@dataclass
class QueryPandasContainer(QueryContainer[pd.DataFrame]):
    @override
    def _data_type_check(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
    
    @override
    def _colunm_names_check(self) -> None:
        test_cols = [i in self.data for i in self.cols]
        if not all(test_cols):
            diff = [c for c, test in zip(self.cols, test_cols) if not test]
            raise KeyError(f"Column specified in `cols` and `id_vars` must be valid colunm names in `data`. Missing columns: {str(diff)}")


class QueryPandas(QueryTable[pd.DataFrame]):
    def __init__(self, data, cols, query=None):
        self._container = QueryPandasContainer(data=data, cols=cols, query=query)

    @classmethod
    def exec(cls, *args, **kwargs) -> pd.DataFrame | Self:
        if len(args)!=0:
            return cls(*args)
        
        data = kwargs.pop('data')
        cols = kwargs.pop('cols')
        query = kwargs.pop('query')
        returns = kwargs.pop('returns', None)
        if (data is None) or (cols is None):
            raise ValueError("Data and columns are required for execution.")
        
        if not (returns in ('instance', 'query')):
            raise ValueError("Returns must be either 'instance' or 'query'")
        
        inst = cls(data=data, cols=cols, query=query)
        if returns == 'instance':
            return inst
        else:


        return inst
        breakpoint()

    
    def query_tidy(self, cols: list | str | None = None, query: str | None = None):


        if cols is not None:
            if isinstance(cols, str):
                cols = [cols] if cols!="" else ""
            elif not isinstance(cols, list):
                raise TypeError("`cols` must be a list, str or None or empty string")
        else:
            cols = ""
        
        if cols!="":
            return self.data.query(self.query)[cols]
        else:
            return (
                self.data[cols]
                    .melt(id_vars=id_vars)
            )
        return (
                self.data.query(self.query)[cols]
                            .melt(id_vars=id_vars)
            )
