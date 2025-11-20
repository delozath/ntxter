from abc import ABC, abstractmethod


from pathlib import Path

class BaseDataFrameReport[T, U](ABC):
    @abstractmethod
    def build(self, data: T, /,**kwargs) -> U: ...

    @abstractmethod
    def save(self, pfname: Path, /, **kwargs) -> None | U: ...