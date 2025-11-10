from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Callable, override


import pandas as pd


from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core import utils

from ntxter.core.mlops.metrics import MetricsContainter, Metric, Reporting


class SklearnMetricsContainer(MetricsContainter):
    def __init__(self) -> None:
        super().__init__()
    
    @override
    def register(self, name: str, /, **kwargs) -> None:
        if name in self._registry:
            raise ValueError(f"Metric with name `{name}` is already registered.")
        if name not in kwargs:
            kwargs = kwargs | {'name': name}

        cls_kwargs, _ = utils.safe_init(Metric, **kwargs)
        breakpoint()
    
    @override
    def compute(self, y_true, y_pred) -> pd.DataFrame:
        pass

    @override
    def set_info_report(self, fold, /, **kwargs):
        pass
    


"""
@dataclass
class Metric:
    name: str
    options: Dict
    function: Callable

    def __post_init__(self):
        if not callable(self.function):
            raise ValueError("The `function` attribute must be callable.")


@dataclass
class Reporting:
    pipeline_id: str
    fold: int


class MetricsContainter:

    @abstractmethod
    def register(self, name: str, /, **kwargs) -> None:
        if name in self._registry:
            raise ValueError(f"Metric with name '{name}' is already registered.")
        
        #
    
    @abstractmethod
    def compute(self, y_true, y_pred) -> pd.DataFrame: ...

    @abstractmethod
    def set_info_report(self, fold, /, **kwargs): ...
"""