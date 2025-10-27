import pandas as pd


from ntxter.core.data.loaders import DataLoader


class CommonTypesLoader(DataLoader):
    def __init__(self, pfname):
        super().__init__(pfname)

        self._loaders = {
            'sav' : self._read_spss,
            'xls' : self._read_excel,
            'xlsx': self._read_excel,
            'csv' : self._read_csv
        }
    
    def _read_spss(self):
        data = pd.read_spss(self.pfname)
        return data
    
    def _read_excel(self, sheets=None):
        data = pd.read_excel(
            self.pfname,
            sheet_name=sheets
        )
        return data
    
    def _read_csv(self):
        data = pd.read_csv(self.pfname)
        return data
    
    def load(self, *args, **kwargs):
        func = self._loaders.get(self.ext, None)
        if func is not None:
            return func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Loader for {self.ext} is not implemented")
