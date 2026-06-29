from typing import Self
from abc import ABC, abstractmethod

from ntxter.core.data.types import QueryContainer


class QueryTable[T](ABC):
    _container: QueryContainer

    @abstractmethod
    def exec(self, *args, **kwargs) -> T | Self:
        raise NotImplementedError("Method `compose` must be implemented")
    
    def __getattr__(self, name):
        if hasattr(self._container, name):
            return getattr(self._container, name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute `{name}` associated to the composition into DataFrameRetriever") 