from ntxter.ports.estimators.BaseEstimator import BaseEstimator

from typing import override

import numpy as np
import pandas as pd
import polars as pl

from ntxter.core.data.types import EstimatorProtocol
from ntxter.core.base.descriptors import RegistryFunctionDescriptor

class SklearnSingleEstimator(BaseEstimator):
    metrics_registry = RegistryFunctionDescriptor()

    def __init__(self) -> None:
        self.metrics = {}

    @override
    def perform(self,
            model: EstimatorProtocol,
            X: np.ndarray | pd.DataFrame | pl.DataFrame,
            y: np.ndarray | pd.Series,
            tn_idx: np.ndarray | list,
            tt_idx: np.ndarray | list
        ):

        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise ValueError("X and y must be numpy arrays")
        
        X_train, y_train = X[tn_idx], y[tn_idx]
        X_test, y_test = X[tt_idx], y[tt_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        self.metrics = {}
        for key, func in self.metrics_registry.items():
            self.metrics[key] = func(y_test, predictions)
