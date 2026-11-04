"""Pandas feature-exploration pipeline."""

from dataclasses import dataclass

from matplotlib.figure import Figure
import pandas as pd

from ntxter.adapters.data.feature_eda import (
    PandasRedundancyResult,
    PandasRedundancySelector,
)
from ntxter.adapters.plots.exploration import MatplotlibAssociationHeatmap
from ntxter.core.base.classes import BaseRunnableClass
from ntxter.core.plots import PlotResult, PlotSpec


@dataclass(slots=True)
class PandasFeatureExplorationResult:
    """Redundancy selection and its graphical representation."""

    selection: PandasRedundancyResult
    plot: PlotResult[Figure]


class PandasFeatureExplorationPipeline(BaseRunnableClass):
    """Select redundant features and render their association matrix."""

    def __init__(self, spec: PlotSpec | None = None) -> None:
        self._selector = PandasRedundancySelector()
        self._plot = MatplotlibAssociationHeatmap(spec=spec)

    def run(
        self,
        df: pd.DataFrame,
        columns=None,
        threshold: float = 0.8,
        numeric_method: str = "spearman",
        categorical_method: str = "cramer_v",
        mixed_method: str = "correlation_ratio",
        n_jobs: int = 1,
        **plot_kwargs,
    ) -> PandasFeatureExplorationResult:
        selection = self._selector.run(
            df=df,
            columns=columns,
            threshold=threshold,
            numeric_method=numeric_method,
            categorical_method=categorical_method,
            mixed_method=mixed_method,
            n_jobs=n_jobs,
        )
        plot = self._plot.build(
            selection.association_matrix,
            **plot_kwargs,
        )
        return PandasFeatureExplorationResult(
            selection=selection,
            plot=plot,
        )


__all__ = [
    "PandasFeatureExplorationPipeline",
    "PandasFeatureExplorationResult",
]
