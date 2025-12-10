from typing import override

import numpy as np
import pandas as pd

from ntxter.ports.statistics.base import StatisticsBase, StatisticsDataContainer

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
    
    def get_numerical_columns(self) -> list[str]:
        if self.container.data is None:
            raise ValueError("Data container has no data.")
        df = self.container.data
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numerical_cols

    @override
    def get_numerical_cols(self, data) -> list:
        return data.select_dtypes(include=np.number).columns.tolist()

    @override
    def get_number_(self, data) -> list:
        cols = data.select_dtypes(include=np.number).columns.tolist()

        breakpoint()

    @override
    def get_number_ncat_cols(self) -> list:
        ...