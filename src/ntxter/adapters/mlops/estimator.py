from curses import meta

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
        self.available_pred = {
            'empty': 'predict',
            'predict': 'predict',
            'proba': 'predict_proba',
            'score': 'decision_function',
            'log_proba': 'predict_log_proba'

        }

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
        
        self._check_predict_types(model)

        self.metrics = {}
        pred_cache = {
            'empty': None,
            'predict': None,
            'proba': None,
            'score': None,
            'log_proba': None,
        }
        for key, metric in self.metrics_registry.items():
            metadata = metric['metadata']
            func = metric['func']
            predictions = pred_cache[metadata]
            if predictions is None:
                predictor_name = self.available_pred[metadata]
                predictions = getattr(model, predictor_name)(X_test)
                pred_cache[metadata] = predictions

            self.metrics[key] = func(y_test, predictions)
    
    def _check_predict_types(self, model):
        metadata = {
            v['metadata'] for _, v in self.metrics_registry.items()
        }
        if len(metadata)<1:
            raise AttributeError("Metrics registry must contain at least one valid output format")

        for m in metadata:
            pred = self.available_pred.get(m, None)
            if pred is None:
                raise KeyError(f"Predict type named `{m}` is not admitted, just admit one of the following: {self.available_pred.keys()}")
            
            if not hasattr(model, pred):
                raise AttributeError(f"Model does not have a predictor fuction named `{pred}`")
