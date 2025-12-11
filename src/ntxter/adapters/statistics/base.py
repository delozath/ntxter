from typing import Dict, override

import numpy as np
import pandas as pd

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
    def _compute(self, x) -> dict[str, float]:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            mean_value = x.mean().to_dict()
        elif isinstance(x, np.ndarray):
            mean_value = float(np.mean(x))
        else:
            raise TypeError("Input data must be a NumPy array, Pandas DataFrame, or Pandas Series")
        return {"mean": mean_value}

    @override
    def compute(self, data: 'StatisticsDataContainer') -> 'StatisticsMean':
        if data.data is None:
            raise ValueError("Data container has no data to compute statistics on.")
        stats = self._compute(data.data)
        data.statistics = stats
        self.container = data
        return self
    
    @override
    def disaggregate_cols(self, data) -> Dict[str, list | Dict]:
        disaggregated = PandasDFAnalyzer.disaggregate_cols(data)
        
        return disaggregated

    def disaggregate_to_df(self, data):
        disaggregated = self.disaggregate_cols(data)
        df = pd.DataFrame.from_dict(disaggregated, orient='index')
        breakpoint()
        return df