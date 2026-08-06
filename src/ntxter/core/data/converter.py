from typing import Self, Protocol
from abc import ABC, abstractmethod

from dataclasses import dataclass

from ntxter.core.data.types import QueryContainer

class Table(Protocol):
    shape: tuple
    column: list


@dataclass
class TableLevelContainer[T](ABC):
    data: T
    n_columns: int = 2
 
    def __post_init__(self) -> None:
        self._check_number_of_columns()
    
    @abstractmethod
    def _check_number_of_columns(self):
        raise NotImplementedError("Method `_check_number_of_columns` must be implemented")


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