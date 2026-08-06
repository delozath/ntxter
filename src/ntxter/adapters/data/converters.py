from typing import Self, TypedDict, override
from dataclasses import dataclass
import pandas as pd

from ntxter.core.data.types import QueryContainer
from ntxter.core.data.converter import QueryTable, TableLevelContainer


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
        returns = kwargs.pop('returns', 'query')
        if len(args)!=0:
            if returns=='query':
                return cls(*args)._query()
            else:
                return cls(*args)
        
        data = kwargs.pop('data')
        cols = kwargs.pop('cols')
        query = kwargs.pop('query', "")

        if (data is None) or (cols is None):
            raise ValueError("Data and columns are required for execution.")
        
        if not (returns in ('instance', 'query')):
            raise ValueError("Returns must be either 'instance' or 'query'")
        
        inst = cls(data=data, cols=cols, query=query)
        if returns == 'instance':
            return inst
        else:
            return inst._query()

    def _query(self, cols: list | str | None = None, query: str | None = None):
        f_cols = cols is None
        f_query = query is None

        if f_cols and f_query:
            return self.data.query(self.query)[self.cols]
        elif f_cols and not f_query:
            return self.data.query(query)[self.cols]
        elif not f_cols and f_query:
            return self.data.query(self.query)[cols]
        elif not f_cols and not f_query:
            return self.data.query(query)[cols]
        else:
            raise TypeError("Invalid arguments for `query` or `cols` method.")


DeprecationWarning("Quizá no sea necesario")
class PandasTidyLevelFrame(TableLevelContainer[pd.DataFrame]):
    def __init__(self, data: pd.DataFrame, n_columns: int, scheme: list):
        self.data = data
        self.n_columns = n_columns
        self.scheme = scheme
    
    @override
    def _check_number_of_columns(self):
        if (n_cols:=len(self.data.columns)) != self.n_columns:
            raise ValueError(f"Expected {self.n_columns} but {n_cols} columnes in dataframe were provided")
        
        breakpoint()