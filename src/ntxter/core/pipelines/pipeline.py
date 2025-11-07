from abc import ABC, abstractmethod
from typing import override, Iterator, Dict, Callable, List, Self, Protocol, Type, Tuple
from dataclasses import dataclass, field


import numpy as np
import pandas as pd


from ntxter.core import utils
from ntxter.core.data.types import EstimatorProtocol as Estimator
from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.data.types import BasePipelineStage


class BaseRegistry(Protocol):
    _registry: Dict[str, BasePipelineStage]
    
    @abstractmethod
    def register(self, name: str,
                       stage: str,
                       estimator: Type[Estimator],
                       params: Dict
         ) -> None:
        if name in self._registry:
            raise ValueError(f"An item with name '{name}' is already registered.")
        ...
    
    @abstractmethod
    def iterate(self) -> Iterator[tuple[str, BasePipelineStage]]:
        for name, item in self._registry.items():
            yield name, item

class BasePipelineContainer(BaseRegistry, ABC):
    registry = SetterAndGetterType(dict)

    @override
    def register(self, name: str,     
                       stage: str,
                       estimator: Type[Estimator],
                       params: Dict
         ) -> None:
        super().register(name, stage, estimator, params)
        self._registry[name] = BasePipelineStage(stage, estimator, params)

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, 
                  y: np.ndarray | pd.Series) -> Iterator[Tuple[str, Type[Estimator]]]:
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> \
            Iterator[int | float | np.ndarray | pd.Series | List]:
        ...
    
    @abstractmethod
    def validate(self, X: np.ndarray | pd.DataFrame, 
                       y: np.ndarray | pd.Series,
                       X_test: np.ndarray | pd.DataFrame) -> \
            Iterator[Tuple[Type[Estimator], int | float | np.ndarray | pd.Series | List]]:
        ...