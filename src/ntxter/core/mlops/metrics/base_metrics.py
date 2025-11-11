from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Callable


import pandas as pd


from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core import utils


@dataclass
class Metric:
    name: str
    function: Callable
    options: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.function):
            raise ValueError("The `function` attribute must be callable.")


@dataclass
class Reporting[T]:
    pipeline_id: str
    fold: int
    performances: Dict[str, T]


class BaseMetricsContainter(ABC):
    registry = SetterAndGetterType(dict)

    def __init__(self) -> None:
        self._registry: dict[str, Metric] = {}

    @abstractmethod
    def register(self, name: str, /, **kwargs) -> None:
        if name in self._registry:
            raise ValueError(f"Metric with name '{name}' is already registered.")
        
        #cls_kwargs, _ = utils.safe_init(Metric, **kwargs)
    
    @abstractmethod
    def compute(self, y_true, y_pred) -> pd.DataFrame: ...


class BaseReportMetrics[T](ABC):
    @abstractmethod
    def add(self, report: Reporting[T]) -> None: ...

    @abstractmethod
    def build(self) -> pd.DataFrame: ...