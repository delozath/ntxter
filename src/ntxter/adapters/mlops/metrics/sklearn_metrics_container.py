from typing import Dict, override


import pandas as pd


from ntxter.core import utils

from ntxter.core.mlops.metrics import BaseMetricsContainter, Metric


class SklearnMetricsContainer(BaseMetricsContainter):
    def __init__(self) -> None:
        super().__init__()
    
    @override
    def register(self, name: str, /, **kwargs) -> None:
        if name in self._registry:
            raise ValueError(f"Metric with name `{name}` is already registered.")
        if name not in kwargs:
            kwargs = kwargs | {'name': name}

        cls_kwargs, _ = utils.safe_init(Metric, **kwargs)
        self._registry[name] = cls_kwargs

    @override
    def compute(self, y_true, y_pred) -> pd.DataFrame | Dict:
        results = {}
        for name, metric in self._registry.items():
            func = metric.function
            options = metric.options
            results[name] = func(y_true, y_pred, **options)
        
        return results