from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Self, Type


import numpy as np


from ntxter.core.data.types import PipelineProtocol as Pipeline


class BaseRemoveOutliers(ABC):   
    pipeline: Type[Pipeline]
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Self:
        ...
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        ...

    @abstractmethod
    def is_outlier(self, *args, **kwargs) -> np.ndarray:
        ...