import pandas as pd


from ntxter.data.dtypes import EvalLoopStatus


def median_iqr_report(x, decimals):
    return f"{x.median():.{decimals}f} ({x.quantile(0.25):.{decimals}f} -- {x.quantile(0.75):.{decimals}f})"


class ModelPerformances:
    def __init__(self, state_kfold: EvalLoopStatus) -> None:
        self._performances: list = []
        self._metrics: dict = {}
        self.state_kfold: EvalLoopStatus = state_kfold

    def register(self, name):
        def deco(func):
            if name in self._metrics.keys():
                raise KeyError(f"{name}: Metric ID already added")
            if not callable(func):
                raise TypeError("func parameter has to be callable")
            self._metrics[name] = func
            return func
        return deco
    
    @property
    def performances(self):
        table = pd.DataFrame(self._performances)
        return table
    
    @performances.setter
    def performances(self, value):
        raise ValueError(f"performance property is not allowed to be assigned")
    
    def evaluate(self, y, y_est):
        metrics = {id: fn(y, y_est) for id, fn in self._metrics.items()}
        metrics |= self.state_kfold.k_iter
        self._performances.append(metrics)
        return metrics