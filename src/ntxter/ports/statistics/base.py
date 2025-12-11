from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Self

import numpy as np
import pandas as pd

from ntxter.core.base.descriptors import SetterAndGetterType


@dataclass
class StatisticsDataContainer[U, T]:
    """Base class for statistics data representation.
    Attributes
    ----------
    name : str
        Name of the statistics data.
    description : str
        Description of the statistics data.
    data : np.ndarray | pd.DataFrame | pd.Series | None
        The actual data, which can be a NumPy array, Pandas DataFrame, or Series.
    
    TODO
    ----
    Extend data type such as polars.
    """
    name: str
    description: str = ""
    data: U | None = None
    statistics: Dict[str, T] | None = None

    def __post_init__(self):
        if self.data is not None:
            if not isinstance(self.data, (np.ndarray, pd.DataFrame, pd.Series)):
                raise TypeError("data must be a NumPy array, Pandas DataFrame, or Pandas Series")


class StatisticsBase[U, T](ABC):
    container = SetterAndGetterType(StatisticsDataContainer)
    """
    Base class for statistics performance metrics

    Methods
    -------
    _compute(x) -> Dict[str, T]
        Abstract method to compute performance metrics from input data.
    
    compute(data: StatisticsDataContainer) -> self
        Computes the performance metrics based on the provided statistics data and stores in the data instance
    
    """
    
    @abstractmethod
    def _compute(self, x) -> Dict[str, T]:
        ...
        raise NotImplementedError("Subclasses must implement the compute method.")

    @abstractmethod
    def compute(self) -> Self:
        """Computes the performance metrics based on the provided statistics data.
        Parameters
        ----------
        data : StatisticsDataContainer
            The input statistics data.
        
        Returns
        -------
        StatisticsDataContainer
            The computed performance metrics encapsulated in a StatisticsDataContainer instance.
        """
        res = self._compute(self.container.data)
        self.container.statistics = res
        return self

    @abstractmethod
    def disaggregate_cols(self, data) -> Dict[str, list | Dict]:
        ...
    
    