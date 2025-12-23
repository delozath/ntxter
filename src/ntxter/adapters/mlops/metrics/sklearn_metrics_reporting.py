from typing import Dict, Any, override


import pandas as pd


from ntxter.core import utils
from ntxter.core.mlops.metrics import BaseReportMetrics, Reporting



class SklearnMetricsReporting(BaseReportMetrics[float]):
    def __init__(self) -> None:
        super().__init__()

    @override
    def to_reporting(
        self,
        identifier: int | str,
        /,
        iteration: int = 0,
        performances: Dict[str, float] | None = None,
        optional: Dict[str, Any] | None = None,

     ) -> Reporting[float]:
        cls_kwargs, _ = utils.safe_init(Reporting, 
            identifier=identifier,
            iteration=iteration,
            performances=performances if performances is not None else {},
            optional=optional
        )
        return cls_kwargs

    @override
    def add(self, report: Reporting[float]) -> None:
        super().add(report)
        self._registry[report.identifier] = report
    
    @override
    def build(self) -> pd.DataFrame:
        metrics = set(self._registry[0].performances.keys())
        records = []
        append = records.append
        for report in self._registry.values():
            record = {
                'identifier': report.identifier,
                'iteration': report.iteration,
                **report.performances,
                **(report.optional if report.optional is not None else {})
            }
            append(record)
            metrics = metrics.union(report.performances.keys())
        
        self.metrics = list(metrics)
        
        return pd.DataFrame(records)
    
    def build_table_medians(self, df_metrics: pd.DataFrame, groupby: list[str] | str) -> pd.DataFrame:
        cols = self.metrics + [groupby] if isinstance(groupby, str) else self.metrics + groupby
        table = df_metrics[cols].melt(id_vars=groupby)
        table.pivot_table(index=groupby, columns='variable', aggfunc=self._median_iqr)
        table = (
            df_metrics[cols]
                .melt(id_vars=groupby)
                .pivot_table(
                    index=groupby,
                    columns='variable',
                    aggfunc=self._median_iqr
                 )
        )
        return table
    
    def _median_iqr(self, series: pd.Series, dec: int = 4) -> str:
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        return f"{median:.{dec}f} ({q1:.{dec}f}--{q3:.{dec}f})"

