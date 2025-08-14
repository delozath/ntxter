import pandas as pd

class DataFramePerformanceIterative:
    def __init__(self, cols):
        self.columns = cols
        self._table = []
        self._row = None
    #
    @property
    def row(self):
        if self._row:
            return self._row
        else:
            return None
    #
    @row.setter
    def row(self, value):
        if self._row is None:
            self._row = value
        else:
            raise RuntimeError(f"current row has not been append to the report table")
    #
    @property
    def table(self):
        if len(self._table)>0:
            return pd.concat(self._table)
        else:
            return
    #
    def append_to_row(self, values):
        if self._row is not None:
            self._row |= values
        else:
            raise RuntimeError(f"current row has not been inicialized")
    #
    def append(self):
        if self._row is not None:
            tmp = pd.DataFrame([self._row])
            self._table.append(tmp)
            self._row = None
        else:
            print("No row to append. Continue")