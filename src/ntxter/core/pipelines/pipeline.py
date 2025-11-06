from abc import ABC, abstractmethod
from turtle import st
from typing import Self, Iterable, Any
from dataclasses import dataclass, field


from sklearn.pipeline import Pipeline


from ntxter.core.base.descriptors import SetterAndGetter, SetterAndGetterType


class BasePipeline():
    pipeline_ =  SetterAndGetter()

    def insert_stage(self, place: int, stage: tuple):
        self._pipeline_.steps.insert(place, stage)


@dataclass
class BaseModelConfig:
    family: str
    subfamily: str
    type: str
    params: dict = dict()

class BasePipelineContiner(ABC):
    registry = SetterAndGetterType(dict)

    def register(self, name: str, pipeline: Pipeline) -> Self:
        if name in self._registry.keys():
            raise ValueError(f"Pipeline with name {name} is already registered.")
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be an instance of sklearn.pipeline.Pipeline")
        self._registry(name, pipeline)
        return self

    @abstractmethod
    def _register(self, name: str, pipeline: Pipeline) -> Self:
        pass

    def remove(self, name: str) -> None:
        if name in self._registry().keys():
            self._registry().pop(name)

    def __get_item__(self, name: str) -> Pipeline:
        if name not in self._registry.keys():
            raise KeyError(f"Pipeline with name {name} is not registered.")
        return self._registry()[name]
    
    def fit(self, X, y=None) -> Iterable:
        try:
            yield from self._fit(X, y)
        except StopIteration:
            print(f"Error occurred while fitting: {e}")

    def _fit(self, X, y):
        for name, stage in self._registry().items():
            stage.fit(X, y)
            yield name, stage
        
        return self


    @abstractmethod
    def __str__(self) -> str:
        ...

class BasePipelineContiner(ABC):

    
    @abstractmethod
    def predict(self, X) -> Iterable:
        ...
    