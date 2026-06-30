import pandas as pd
import pingouin as pg

from typing import Self

from ntxter.ports.statistics.base import StatisticsContainer

class MultipleComparison:
    def run(self, df: pd.DataFrame, dv="value", within="model", subject="iteration", method='f', padjust='holm'
, effsize='r', parametric=False) -> Self:
        fried_tests = []
        posthoc_test = {}
        append = fried_tests.append
        for grpnm, grp in df.groupby('metric'):
            fried = pg.friedman(data=grp, dv=dv, within=within, subject=subject, method=method)
            fried['metric'] = grpnm
            append(fried)
            if fried['p-unc'].values[0]<0.05:
                #print("Significant differences found, proceeding with post-hoc")
                posthoc = pg.pairwise_tests(
                    data=grp,
                    dv=dv,
                    within=within,
                    subject=subject,
                    padjust=padjust,
                    effsize=effsize,
                    parametric=parametric
                )
                posthoc_test[grpnm] = posthoc.sort_values("p-corr")

        self.container.summary = {
            'friedman': pd.concat(fried_tests, ignore_index=True),
            'posthocs': posthoc_test
        }

        return self