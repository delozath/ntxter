from abc import ABC, abstractmethod
from typing import Self, Type


import numpy as np


from ntxter.core.data.types import PipelineProtocol as Pipeline


class BaseRemoveOutliers(ABC):   
    pipeline: Type[Pipeline]
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Self:
        ...
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        ...

    @abstractmethod
    def is_outlier(self, *args, **kwargs) -> np.ndarray:
        ...