from abc import ABC, abstractmethod

from pathlib import Path


from pandas import DataFrame


class DataLoader:
    def __init__(self, pfname: Path) -> None:
        pfname = self._file_check(pfname)
        self.pfname = pfname

        self.data: DataFrame
        self.ext = pfname.suffix[1:].lower()
    
    def _file_check(self, pfname_):
        pfname = Path(pfname_)
        if not pfname.exists():
            raise FileNotFoundError(f"file {pfname} not found")
        else:
            return pfname
    
    @abstractmethod
    def load(self, *args, **kwargs) -> DataFrame:
        ...