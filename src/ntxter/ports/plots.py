"""Ports implemented by concrete plotting backends."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ntxter.core.plots import PlotResult, PlotSpec


DataT = TypeVar("DataT")
FigureT = TypeVar("FigureT")


class BasePlot(ABC, Generic[DataT, FigureT]):
    """Contract for a named, reusable plot."""

    @property
    @abstractmethod
    def spec(self) -> PlotSpec:
        """Return the stable specification of this plot."""
        ...

    @abstractmethod
    def build(self, data: DataT, **kwargs) -> PlotResult[FigureT]:
        """Build the figure without imposing a plotting backend."""
        ...
