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
        funcs = {
            'shapiro': self._shapiro,
            'dagostino': self._dagostino,
            'anderson': self._anderson,
            'ks': self._kolmogorov_smirnov,
        }

        if grouping is not None:
            grouping = [grouping] if isinstance(grouping, str) else grouping
        
        if isinstance(self.data, pd.DataFrame):
            Test = []
            for loop in self._query(self.data, grouping=grouping):
                for test, func in funcs.items():
                    df = self._to_frame(
                        test,
                        loop[0],
                        loop[1],
                        func(loop[2])
                    )
                    
                    Test.append(df)
            result_df = pd.concat(Test, ignore_index=True)
            self.container.summary = {'results': result_df}
            return result_df
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

        if grouping is not None:
            df = x[cols + grouping]
        else:
            grouping = ['fake_group']
            df = x[cols]
            df['fake_group'] = 'all_data'
        
        for name, group in df.groupby(grouping):
            group = group.drop(columns=grouping)
            for col, xf in group.items():
                yield name, col, xf
    
    def _to_frame(self, test, group, column, values) -> pd.DataFrame:
        group = '' if group is None else '_'.join(map(str, group))

        df = (pd.DataFrame
            .from_dict(values, orient='index')
            .reset_index()
            .rename(columns={'index': 'params', 0: 'value'})
         )
        
        df['test'] = test
        df['column'] = column
        df['group'] = group
        
        return df

    def _shapiro(self, x: pd.Series) -> Dict[str, float]:
        stat, p_value = shapiro(x.dropna())
        params = {
            'stats': stat, 
            'p_value': p_value
        }
        
        return params
    
    def _dagostino(self, x: pd.Series) -> Dict[str, float]:
        stat, p_value = normaltest(x.dropna())
        params = {
            'stats': stat, 
            'p_value': p_value
        }
        
        return params
    
    def _anderson(self, x: pd.Series) -> Dict[str, float]:
        result = anderson(x.dropna(), dist='norm')
        params = {
            'stats': result.statistic, 
            'critical_values': result.critical_values.tolist(),
            'significance_level': result.significance_level.tolist()
        }
        
        return params
    
    def _kolmogorov_smirnov(self, x: pd.Series) -> Dict[str, float]:
        mean = np.mean(x.dropna())
        std = np.std(x.dropna())
        stat, p_value = kstest(x.dropna(), 'norm', args=(mean, std))
        params = {
            'stats': stat, 
            'p_value': p_value
        }
        
        return params