import pandas as pd
import pingouin as pg

from typing import Self

from ntxter.ports.statistics.base import StatisticsContainer

class ClassifierMultipleComparison:
    container: StatisticsContainer[str]
    COLS_SING_METRIC = ['model_col', 'iteration', 'value']
    COLS_MULT_METRIC = ['model', 'iteration', 'metric', 'variable']
    
    def __init__(self) -> None:
        self.container = StatisticsContainer[str](
            name="ClassifierMultipleComparison",
            description="Compares multiple classifiers using statistical tests",
            summary=None
         )

    def compute(
            self,
            data: pd.DataFrame,
            **kwargs
         ) -> Self | StatisticsContainer[str]:
        
        if isinstance(data, pd.DataFrame):
            n_cols = len(data.columns)
            if n_cols==4:
                func = self._frame_compute_mult_metrics
            elif n_cols==3:
                raise ValueError("Not implemented yet")
            else:
                raise ValueError("DataFrame has to contain columns associated with either [`model`, `iteration`, `value`] or [`model`, `iteration`, `metric`, `value`].")
        
        func(data, **kwargs)

        return self

    def _frame_compute_mult_metrics(
            self,
            df: pd.DataFrame,
            model='model',
            iteration='iteration',
            metric='metric',
            value='value' 
         ) -> Self | StatisticsContainer[str]:
        cols = df.columns.tolist()
        
        colnames_test =all(map(lambda x, ref=cols: x in ref, [model, iteration, metric, value]))
        if not colnames_test:
            df.columns = [model, iteration, metric, value]
            print(f"Log: Column names changed to [{model}, {iteration}, {metric}, {value}], assuming consistent order.")
        
        fried_tests = []
        posthoc_test = {}
        append = fried_tests.append
        for grpnm, grp in df.groupby('metric'):
            fried = pg.friedman(data=grp, dv="value", within="model", subject="iteration", method='f')
            fried['metric'] = grpnm
            append(fried)
            if fried['p-unc'].values[0]<0.05:
                #print("Significant differences found, proceeding with post-hoc")
                posthoc = pg.pairwise_tests(
                    data=grp,
                    dv="value",
                    within="model",
                    subject="iteration",
                    parametric=False,
                    padjust='holm',
                    effsize='r'
                )
                posthoc_test[grpnm] = posthoc.sort_values("p-corr")

        self.container.summary = {
            'friedman': pd.concat(fried_tests, ignore_index=True),
            'posthocs': posthoc_test
        }

        return self