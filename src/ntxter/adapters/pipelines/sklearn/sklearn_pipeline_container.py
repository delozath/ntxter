from typing import override, Iterator, Dict, List, Protocol, Type, Tuple


import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


from ntxter.core import utils
from ntxter.core.data.types import EstimatorProtocol as Estimator
from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.pipelines.pipeline import BasePipelineContainer
from ntxter.core.pipelines.pipeline import BasePipelineStage
from ntxter.core.data.types import PipelineProtocol


class SklearnPipelineContainer(BasePipelineContainer):
    def __init__(self) -> None:
        self.registry: dict[str, Pipeline] = {}
    
    @override
    def register(self, name: str,
                       estimator: Estimator,
         ) -> None:
        if name in self._registry: # type: ignore[override]
            raise ValueError(f"An item with name '{name}' is already registered.")

        self._registry[name] = BasePipelineStage(stage=name, estimator=estimator)
    
    @override    
    def iterate(self) -> Iterator[tuple[str, BasePipelineStage]]:
        for name, item in self._registry.items():
            yield name, item

    @override
    def fit(self, X: np.ndarray | pd.DataFrame, 
                  y: np.ndarray | pd.Series) -> Iterator[Tuple[str, Estimator | Type[Estimator]]]:
        for name, stage in self.iterate():
            stage.estimator.fit(X, y)
            yield name, stage.estimator
    
    def fit_pipeline(self, name: str,
                         X: np.ndarray | pd.DataFrame, 
                         y: np.ndarray | pd.Series) -> Estimator | Type[Estimator]:
        if name not in self._registry:
            raise ValueError(f"No pipeline with name '{name}' is registered.")
        
        pipeline = self._registry[name]
        pipeline.estimator.fit(X, y)
        return pipeline.estimator
    
    def predict_pipeline(self, name: str,
                             X: np.ndarray | pd.DataFrame) -> \
            int | float | np.ndarray | pd.Series | List:
        if name not in self._registry:
            raise ValueError(f"No pipeline with name '{name}' is registered.")
        
        pipeline = self._registry[name]
        return pipeline.estimator.predict(X)
    
    @override
    def predict(self, X: np.ndarray | pd.DataFrame) -> \
            Iterator[int | float | np.ndarray | pd.Series | List]:
        pass
    
    @override
    def validate(self, X: np.ndarray | pd.DataFrame, 
                       y: np.ndarray | pd.Series,
                       X_test: np.ndarray | pd.DataFrame) -> \
            Iterator[Tuple[Type[Estimator], int | float | np.ndarray | pd.Series | List]]:
        pass