from abc import ABC, abstractmethod
from typing import override, Iterator, Dict, Callable
from dataclasses import dataclass, field


from ntxter.core.base.descriptors import SetterAndGetter, SetterAndGetterType


class BasePipeline():
    pipeline_ =  SetterAndGetter()

    def insert_stage(self, place: int, stage: tuple):
        self._pipeline_.steps.insert(place, stage)


@dataclass
class PipelineStageConfig:
    stage: str
    family: str
    subfamily: str
    estimator: Callable
    params: Dict

    def __post_init__(self):
        self.estimator(**self.params)


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


class BasePipelineContainer[T](BaseRegistry[T], ABC):
    def __init__(self) -> None:
        super().__init__()

    @override
    def register(self, name: str, **kwargs) -> None:
        super().register(name, **kwargs)
        self._pipelines[name] = self.build_pipeline_config(**kwargs)

    @abstractmethod
    def build_pipeline_config(self, **kwargs) -> PipelineStageConfig:
        ...