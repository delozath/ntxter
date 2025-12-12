from typing import Dict, Self, override

import numpy as np
import pandas as pd


from scipy.stats import shapiro, normaltest, anderson, kstest, jarque_bera


from ntxter.core.base.errors import UnknownDataTypeError
from ntxter.ports.statistics.base import StatisticsBase, StatisticsDataContainer

class PandasDFAnalyzer:
    @staticmethod
    def disaggregate_cols(df, MAX_CAT_LEVELS: int = 5) -> Dict[str, list | Dict]:
        disaggregated = {}
        
        for nm, xf in df.items():
            if pd.api.types.is_categorical_dtype(xf):
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


class StatisticsNormalityPandasDF(StatisticsBase[pd.DataFrame, pd.Series]):
    def __init__(self, data) -> None:
        self.container = StatisticsDataContainer[pd.DataFrame, pd.Series](
            name="StatisticsNormalityPandasDF",
            description="Computes normality statistics for a Pandas DataFrame",
            data=data
         )
    
    @override
    def _compute(self, x, grouping=None) -> dict[str, float]:
        disagg = self.disaggregate_to_df(x)
        cols = disagg.query("(dtype == 'numerical') & (ntype=='continuous')").index.tolist()

        df = x[cols + grouping]
        self.test_(df, grouping)
        breakpoint()

    def test_(self, df, grouping):
        for name, group in df.groupby(grouping):
            group = group.drop(columns=grouping)
            breakpoint()

    @override
    def compute(self, data: pd.DataFrame, grouping: None = None) -> Self:
        if grouping is not None:
            grouping = [grouping] if isinstance(grouping, str) else grouping
        
        if isinstance(data, pd.DataFrame):
            df = self._compute(data, grouping=grouping)
        else:
            raise TypeError("Data must be a Pandas DataFrame or a StatisticsDataContainer containing a Pandas DataFrame.")
        
        return self
    
    @override
    def disaggregate_cols(self, data) -> Dict[str, list | Dict]:
        disaggregated = PandasDFAnalyzer.disaggregate_cols(data)
        
        return disaggregated

    def disaggregate_to_df(self, data):
        disaggregated = self.disaggregate_cols(data)
        df = pd.DataFrame.from_dict(disaggregated, orient='index')
        
        return df
    