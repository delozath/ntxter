"""Pandas feature-exploration pipeline."""

from dataclasses import dataclass

from matplotlib.figure import Figure
import pandas as pd

from ntxter.adapters.data.feature_eda import (
    PandasRedundancyResults,
    PandasRedundancySelector,
)
from ntxter.adapters.plots.exploration import MatplotlibAssociationHeatmap
from ntxter.core.base.classes import BaseRunnableClass
from ntxter.core.plots import PlotResult, PlotSpec


@dataclass(slots=True)
class PandasFeatureExplorationResult:
    """The three redundancy selections and their available heatmaps."""

    selection: PandasRedundancyResults
    plots: dict[str, PlotResult[Figure]]

    @property
    def plot(self) -> PlotResult[Figure] | None:
        """Compatibility view returning the mixed plot when it exists."""

        return self.plots.get("mixed") or next(iter(self.plots.values()), None)


class PandasFeatureExplorationPipeline(BaseRunnableClass):
    """Analyze feature distributions by numeric, categorical, and mixed views.

    ``outcomes`` are excluded before any association matrix is computed. The
    pipeline is intentionally unsupervised at this stage: it only inspects
    feature-to-feature redundancy.
    """

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
        outcomes=None,
        categorical_columns=None,
        **plot_kwargs,
    ) -> PandasFeatureExplorationResult:
        selection = self._selector.run(
            df=df,
            columns=columns,
            outcomes=outcomes,
            categorical_columns=categorical_columns,
            threshold=threshold,
            numeric_method=numeric_method,
            categorical_method=categorical_method,
            mixed_method=mixed_method,
            n_jobs=n_jobs,
        )
        plots = {
            name: self._plot.build(
                analysis.association_matrix,
                **plot_kwargs,
            )
            for name, analysis in selection.analyses.items()
            if not analysis.association_matrix.empty
        }
        return PandasFeatureExplorationResult(
            selection=selection,
            plots=plots,
        )


__all__ = [
    "PandasFeatureExplorationPipeline",
    "PandasFeatureExplorationResult",
]
