from abc import ABC, abstractmethod
from typing import Callable, Dict, Mapping

from ntxter.core.mlops.base import BaseEstimator
from ntxter.core.data.types import EstimatorProtocol


import numpy as np
import pandas as pd
import polars as pl


class MetricDescriptor:
    def __get__(
        self,
        instance: BaseEstimator,
        owner,
    ) -> Mapping[str, Callable]:
        if instance is None:
            return {}
        registry = instance.__dict__.get("_metrics_registry", {})
        if not registry:
            raise ValueError(
                "__metrics_registry requires at least one registered metric."
            )
        return registry

    def __set__(self, instance: BaseEstimator, value) -> None:
        raise AttributeError("registry is read-only.")

    def __delete__(self, instance: BaseEstimator) -> None:
        raise AttributeError("registry is read-only.")


class BaseEstimator(ABC):
    __slots__ = ("__metrics_registry", )
    __metrics_registry = MetricDescriptor()
    
    def __init__(self):
        self.__metrics_registry = {}

    @abstractmethod   
    def perform(self,
            model: EstimatorProtocol,
            X: np.ndarray | pd.DataFrame | pl.DataFrame,
            y: np.ndarray | pd.Series,
            tn_idx: np.ndarray | list,
            tt_idx: np.ndarray | list
        ):
        raise NotImplementedError("Subclasses must implement 'perform'.")
    
    def add_metrics(self, name: str, metric: Callable):
        breakpoint()
        self.__metrics_registry
        """
        y_test = bdle.y_test.ravel()
        _metrics = {
            'auroc': roc_auc_score(y_test, y_est),
            'sensitivity': recall_score(y_test, y_est, pos_label=0),
            'specificity': recall_score(y_test, y_est, pos_label=self.POS_LABEL),
            #'fbeta_score': fbeta_score(y_test, y_est, beta=0.5, pos_label=self.POS_LABEL, average='binary'),
            'auprc_score': self.pr_auc_score(y_test, y_est, pos_label=self.POS_LABEL),
            'auprc_lift': self.pr_auc_lift(y_test, y_est, bdle.y_train.ravel(), pos_label=self.POS_LABEL),
            'prevalence': (bdle.y_train.ravel()==self.POS_LABEL).mean()
        }"""
