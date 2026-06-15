import numpy as np
import pandas as pd
import polars as pl

from ntxter.core.mlops.base import BaseEstimator
from ntxter.core.data.types import EstimatorProtocol


class SklearnEstimator(BaseEstimator):
    def perform(
            model: EstimatorProtocol,
            X: np.ndarray | pd.DataFrame | pl.DataFrame,
            y: np.ndarray | pd.Series,
            tn_idx: np.ndarray | list,
            tt_idx: np.ndarray | list
        ):
        breakpoint()