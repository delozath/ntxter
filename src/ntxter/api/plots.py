"""Public API for reusable plots."""

from ntxter.adapters.plots import MatplotlibAssociationHeatmap
from ntxter.core.plots import PlotResult, PlotSpec
from ntxter.ports.plots import BasePlot

__all__ = [
    "PlotSpec",
    "PlotResult",
    "BasePlot",
    "MatplotlibAssociationHeatmap",
]
