import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, pfname):
        self._loaders = {
            'sav' : self.read_spss,
            'xls' : self.read_excel,
            'xlsx': self.read_excel,
            'csv' : self.read_csv
        }
        #
        self.pfname = pfname
        pfname_ = self._file_check(pfname)
        self.ext = pfname_.suffix[1:].lower()
    #
    def get(self, **kwargs):
        func = self._loaders[self.ext]
        func(**kwargs)
        print(f"""file "{self.pfname}" loaded""")
    #
    def _file_check(self, pfname_):
        pfname = Path(pfname_)
        if not pfname.exists():
            raise FileNotFoundError(f"file {pfname} not found")
        else:
            return pfname
    #
    def _path_assign(self, pfname_):
        return self._file_check(pfname_) if pfname_ is not None else self.pfname
    #
    def read_spss(self, pfname_=None):
        pfname = self._path_assign(pfname_)
        self.data = pd.read_spss(pfname)
    #
    def read_excel(self, pfname_=None, sheets=None):
        pfname = self._path_assign(pfname_)
        data = pd.read_excel(
            pfname,
            sheet_name=sheets
        )
        #
        self.data = data
    #
    def read_csv(self, pfname_=None):
        pfname = self._path_assign(pfname_)
        self.data = pd.read_csv(pfname)