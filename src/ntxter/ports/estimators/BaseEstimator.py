from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import polars as pl

from ntxter.core.data.types import EstimatorProtocol
from ntxter.core.base.descriptors import RegistryFunctionDescriptor

class BaseEstimator(ABC):
    metrics_registry = RegistryFunctionDescriptor()

    @abstractmethod   
    def perform(self,
            model: EstimatorProtocol,
            X: np.ndarray | pd.DataFrame | pl.DataFrame,
            y: np.ndarray | pd.Series,
            tn_idx: np.ndarray | list,
            tt_idx: np.ndarray | list
        ):
        raise NotImplementedError("This method should be overridden by subclasses.")