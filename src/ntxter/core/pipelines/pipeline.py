from abc import ABC, abstractmethod
from typing import override, Iterator, Dict, Callable, List, Self
from dataclasses import dataclass, field


import numpy as np
import pandas as pd


from ntxter.core import utils
from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.data.types import BasePipelineStage


class BasePipeline(ABC):
    pipeline =  SetterAndGetterType(BasePipelineStage)

    def __init__(self, **kwargs) -> None:
        self.pipeline, extra_params = utils.safe_init(BasePipelineStage, **kwargs)
        self._post_init(**extra_params)
    
    @abstractmethod
    def _post_init(self, **kwargs) -> None:
        pass



class BaseRegistry[T](ABC):
    registry: Dict[str, T] = SetterAndGetterType(dict)
    
    def __init__(self) -> None:
        self.registry = dict()
    
    @abstractmethod
    def register(self, name: str, **kwargs) -> None:
        if name in self._registry:
            raise ValueError(f"An item with name '{name}' is already registered.")
    
    def iterate(self) -> Iterator[tuple[str, T]]:
        for name, item in self._registry.items():
            yield name, item


class BasePipelineContainer(BaseRegistry[BasePipeline], ABC):
    def __init__(self) -> None:
        super().__init__()

    @override
    def register(self, name: str, **pipeline_kwargs) -> None:
        super().register(name, **pipeline_kwargs)
        self._pipelines[name] = PipelineStageConfig(**pipeline_kwargs)

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, 
                  y: np.ndarray | pd.Series) -> Iterator[tuple[str, T]]:
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> Iterator[int | float | np.ndarray | pd.Series | List]:
        ...
    
    @abstractmethod
    def validate(self, X: np.ndarray | pd.DataFrame, 
                       y: np.ndarray | pd.Series,
                       X_test: np.ndarray | pd.DataFrame) -> Iterator[int | float | np.ndarray | pd.Series | List]:
        ...