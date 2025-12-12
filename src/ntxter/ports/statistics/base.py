from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Self

import numpy as np
import pandas as pd

from ntxter.core.base.descriptors import SetterAndGetterType

@dataclass
class StatisticsContainer[T]:
    name: str
    description: str = ""
    summary: Dict[str, T] | None = None


class StatisticsBase[U, T](ABC):
    data: U
    stats = SetterAndGetterType(StatisticsContainer[T])
    """
    Base class for statistics performance metrics

    Methods
    -------
    compute(self, data: StatisticsDataContainer) -> Self
        Computes the performance metrics based on the provided statistics data.
    """

    @abstractmethod
    def compute(self, grouping: list[str] | str | None = None) -> Self | StatisticsContainer[T]:
        """
        Computes the performance metrics based on the provided statistics data.
        Parameters
        ----------
        grouping : list[str] | str | None, optional
            Columns to group by before computing statistics, by default None
        
        Returns
        -------
        Self | StatisticsDataContainer[T]
            The instance itself with computed statistics or the statistics container.
        """
        return self

    @abstractmethod
    def disaggregate_cols(self, data) -> Dict[str, list | Dict]:
        """
        Disaggregates the columns of the data based on their types.
        Parameters
        ----------
        grouping : list[str] | str | None, optional
            Columns to group by before computing statistics, by default None
        
        Returns
        -------
        Dict[str, list | Dict]
            A dictionary with column names as keys and their types as values.
        """
        ...
    
    