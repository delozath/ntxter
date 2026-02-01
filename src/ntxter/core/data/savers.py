from abc import ABC, abstractmethod
from typing import Self

from ntxter.core.utils import path_check


class DataSaver[T](ABC):
    data: T
    def __init__(self, pthfname: str, replace: bool=False) -> None:
        self.pth_fname = path_check(pthfname, replace)
      
    @abstractmethod
    def prepare(self, data: T, *args, **kwargs) -> Self | None:
        ...
    
    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        ...