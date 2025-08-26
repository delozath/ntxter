import pandas as pd

def median_iqr_report(x, decimals):
    return f"{x.median():.{decimals}f} ({x.quantile(0.25):.{decimals}f} -- {x.quantile(0.75):.{decimals}f})"

class MetricsTableDriver:
    def __init__(self, cols):
        self.cols = cols
        self._row = None
        self._table = []
    #
    @property
    def row(self):
        return self._row
    #
    @row.setter
    def row(self, value):
        self._row = self._row | value if self._row else value
    #
    def append(self): 
        self._table.append(self._row)
        self._row = None
    #
    def generate(self):
        return pd.DataFrame(self._table)