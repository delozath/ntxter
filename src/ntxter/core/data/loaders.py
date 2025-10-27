from abc import ABC, abstractmethod

from pathlib import Path


from pandas import DataFrame


class DataLoader(ABC):
    def __init__(self, pfname: Path) -> None:
        pfname = self._file_check(pfname)
        self.pfname = pfname

        self.data: DataFrame
        self.ext = self._get_extension(pfname)
    
    def _file_check(self, pfname_):
        pfname = Path(pfname_)
        if not pfname.exists():
            raise FileNotFoundError(f"file {pfname} not found")
        else:
            return pfname
    
    def _get_extension(self, pfname):
        return pfname.suffix[1:].lower()
    
    @abstractmethod
    def load(self, *args, **kwargs) -> DataFrame:
        ...