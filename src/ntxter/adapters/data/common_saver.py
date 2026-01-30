import pandas as pd


from ntxter.core.data.savers import DataSaver


class PandasFrameSafeFactory(DataSaver[pd.DataFrame]):
    def __init__(self, data, pfname, replace):
        super().__init__(pfname, replace)
        self.data = data
        self.prepare()
    
    def prepare(self, *args, **kwargs) -> None:
        self._loaders = {
            'sav' : self._write_spss,
            'xls' : self._write_excel,
            'xlsx': self._write_excel,
            'csv' : self._write_csv,
            'parquet': self._write_parquet
        }
    
    def save(self, *args, **kwargs) -> None:
        ext = self.pth_fname.suffix[1:].lower()
        
        func = self._loaders.get(ext, None)
        if func is not None:
            return func(**kwargs)
        else:
            raise NotImplementedError(f"Save manager for `{ext}` is not yet implemented")

    def _write_spss(self):
        raise NotImplementedError("SPSS loader is not implemented yet.")

    
    def _write_excel(self, sheets=None):
        raise NotImplementedError("Excel loader is not implemented yet.")

    
    def _write_csv(self, **kwargs):
        index = kwargs.get('index', True)
        self.data.to_csv(self.pth_fname, index=index)
    
    def _write_parquet(self):
        raise NotImplementedError("Parquet loader is not implemented yet.")