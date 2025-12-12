from typing import Dict, Self, override

import numpy as np
import pandas as pd


from scipy.stats import shapiro, normaltest, anderson, kstest, jarque_bera


from ntxter.core.base.errors import UnknownDataTypeError
from ntxter.ports.statistics.base import StatisticsBase, StatisticsContainer

class PandasDFAnalyzer:
    @staticmethod
    def disaggregate_cols(df, MAX_CAT_LEVELS: int = 5) -> Dict[str, list | Dict]:
        disaggregated = {}
        
        for nm, xf in df.items():
            if isinstance(xf.dtype, pd.CategoricalDtype):
                dtype = 'categorical'
            elif pd.api.types.is_bool_dtype(xf):
                dtype = 'bool'
            elif pd.api.types.is_numeric_dtype(xf):
                dtype = 'numerical'
            elif pd.api.types.is_datetime64_any_dtype(xf):
                dtype = 'datetime'
            elif pd.api.types.is_string_dtype(xf):
                dtype = 'string'
            elif pd.api.types.is_object_dtype(xf):
                dtype = 'object'
            else:
                raise UnknownDataTypeError(str(xf.dtype))
            
            ntype = 'discrete' if xf.nunique(dropna=True) < MAX_CAT_LEVELS + 1 else 'continuous'
            disaggregated[nm] = {'dtype': dtype, 'ntype': ntype}

        return disaggregated


class StatisticsNormalityPandasDF(StatisticsBase[pd.DataFrame, pd.DataFrame]):
    def __init__(self, data) -> None:
        self.container = StatisticsContainer[pd.DataFrame](
            name="StatisticsNormalityPandasDF",
            description="Computes normality statistics for a Pandas DataFrame",
         )
        self.data = data

    @override
    def compute(self, grouping: list[str] | str | None = None) -> Self | StatisticsContainer[pd.DataFrame]:
        if grouping is not None:
            grouping = [grouping] if isinstance(grouping, str) else grouping
        
        if isinstance(self.data, pd.DataFrame):
            for loop in self._query(self.data, grouping=grouping):
                self._shapiro(loop[2])
                breakpoint()
        else:
            raise TypeError("Data must be a Pandas DataFrame.")
        
        return self
    
    @override
    def disaggregate_cols(self, data) -> Dict[str, list | Dict]:
        disaggregated = PandasDFAnalyzer.disaggregate_cols(data)
        
        return disaggregated

    def disaggregate_to_df(self, data):
        disaggregated = self.disaggregate_cols(data)
        df = pd.DataFrame.from_dict(disaggregated, orient='index')
        
        return df

    def _query(self, x: pd.DataFrame, grouping: list[str] | None = None) -> pd.DataFrame:
        disagg = self.disaggregate_to_df(x)
        cols = disagg.query("(dtype == 'numerical') & (ntype=='continuous')").index.tolist()

        #if grouping is None:
        #   # for name, col inx[cols]
        #   yield from x[cols].items()
        #else:
        df = x[cols + grouping]
        for name, group in df.groupby(grouping):
            group = group.drop(columns=grouping)
            for col, xf in group.items():
                yield name, col, xf
    
    def _shapiro(self, x: pd.Series) -> Dict[str, float]:
        stat, p_value = shapiro(x.dropna())
        breakpoint()
        return {'shapiro_stat': stat, 'shapiro_p_value': p_value}