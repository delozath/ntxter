from typing import Self
import pandas as pd


from ntxter.core.data.savers import DataSaver


class PandasFrameSafeFactory(DataSaver[pd.DataFrame]):
    def __init__(self, data: pd.DataFrame, pth_fname: str, replace: bool):
        super().__init__(pth_fname, replace)
        self.data = data
        self.prepare()
    
    def prepare(self, *args, **kwargs) -> None:
        self._loaders = {
            'sav' : self._write_spss,
            'xls' : self._write_excel,
            'xlsx': self._write_excel,
            'csv' : self._write_csv,
            'md': self._write_md,
            'parquet': self._write_parquet
        }
    
    @classmethod
    def save(cls, data: pd.DataFrame, pth_fname: str, replace: bool, *args, **kwargs) -> None:
        instance = cls(data, pth_fname, replace)
        ext = instance.pth_fname.suffix[1:].lower()
        
        func = instance._loaders.get(ext, None)
        if func is not None:
            return func(**kwargs)
        else:
            raise NotImplementedError(f"Save manager for `{ext}` is not yet implemented")

    def _write_md(self):
        self.data.to_markdown(self.pth_fname)

    def _write_spss(self):
        raise NotImplementedError("SPSS loader is not implemented yet.")

    
    def _write_excel(self, sheets=None):
        raise NotImplementedError("Excel loader is not implemented yet.")

    
    def _write_csv(self, **kwargs):
        index = kwargs.get('index', True)
        self.data.to_csv(self.pth_fname, index=index)
    
    def _write_parquet(self):
        raise NotImplementedError("Parquet loader is not implemented yet.")


class MarkdownSafe(DataSaver[str]):
    content: str

    def __init__(self, pth_name: str, replace: bool=False) -> None:
        super().__init__(pth_name, replace)
    
    def prepare(self, data: str, *args, **kwargs) -> Self:
        if not isinstance(data, str):
            raise ValueError("Data must be a string containing markdown content.")
        
        self.content = data
        return self
    
    @classmethod
    def save(cls, data: str, pth_name: str, replace: bool, *args, **kwargs) -> None:
        instance = cls(pth_name, replace)
        instance.prepare(data)

        with open(instance.pth_fname, 'w') as f:
            f.write(instance.content)