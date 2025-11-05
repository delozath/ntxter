from abc import ABC, ABCMeta, abstractmethod
from typing import Self, Iterable

from sklearn.pipeline import Pipeline


from ntxter.core.base.descriptors import SetterAndGetter, SetterAndGetterType


class BasePipeline():
    pipeline_ =  SetterAndGetter()

    def insert_stage(self, place: int, stage: tuple):
        self._pipeline_.steps.insert(place, stage)


class BasePipelineContiner(ABC):
    pipelines = SetterAndGetterType(dict)

    def register(self, name: str, pipeline: Pipeline) -> Self:
        if name in self._pipelines.keys():
            raise ValueError(f"Pipeline with name {name} is already registered.")
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be an instance of sklearn.pipeline.Pipeline")
        self._register(name, pipeline)
        return self
    
    @abstractmethod
    def _register(self, name: str, pipeline: Pipeline) -> Self:
        pass

    @abstractmethod
    def fit(self, X, y=None) -> Iterable:
        ...
    
    @abstractmethod
    def predict(self, X) -> Iterable:
        ...
    
    def remove(self, name: str) -> None:
        if name in self.pipelines.keys():
            self.pipelines.pop(name)

    def list(self):
        for name in self.pipelines.keys():
            print(name)