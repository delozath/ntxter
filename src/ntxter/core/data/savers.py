from abc import ABC, abstractmethod
from typing import Self

from ntxter.core.utils import file_exists


class DataSaver[T](ABC):
    data: T | dict[str, T]
    def __init__(self, pthfname: str, replace: bool=False) -> None:
        rep = 'create' if replace else 'raise'
        self.pth_fname = file_exists(pthfname, mode=rep)
      
    @abstractmethod
    def prepare(self, data: T | dict[str, T], *args, **kwargs) -> Self | None:
        ...
    
    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        ...