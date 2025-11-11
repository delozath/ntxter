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
        records = []
        for report in self._registry.values():
            record = {
                'identifier': report.identifier,
                'iteration': report.iteration,
                **report.performances,
                **(report.optional if report.optional is not None else {})
            }
            records.append(record)
        
        return pd.DataFrame(records)

