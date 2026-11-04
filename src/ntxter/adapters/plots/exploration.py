"""Reusable plots for exploratory analysis."""

from dataclasses import dataclass
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from ntxter.core import utils
from ntxter.core.plots import PlotResult, PlotSpec
from ntxter.ports.plots import BasePlot


@dataclass(frozen=True, slots=True)
class MatplotlibFigureOptions:
    figsize: tuple[Real, Real] = (8, 6)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.figsize, tuple)
            or len(self.figsize) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or value <= 0
                for value in self.figsize
            )
        ):
            raise ValueError("figsize must be a tuple of two positive numbers.")


class MatplotlibAssociationHeatmap(BasePlot[pd.DataFrame, Figure]):
    """Render an association matrix as an annotated diverging heatmap."""

    DEFAULT_OPTIONS = {
        "cmap": "RdBu_r",
        "vmin": -1,
        "vmax": 1,
        "center": 0,
        "annot": True,
        "fmt": ".3g",
        "linewidths": 1,
        "linecolor": "white",
        "square": False,
        "cbar_kws": {"label": "Association"},
    }

    def __init__(self, spec: PlotSpec | None = None) -> None:
        if spec is not None and not isinstance(spec, PlotSpec):
            raise TypeError("spec must be a PlotSpec or None.")
        self._spec = spec or PlotSpec(
            name="matplotlib_association_heatmap",
            title="Feature association matrix",
            options=self.DEFAULT_OPTIONS,
        )

    @property
    def spec(self) -> PlotSpec:
        return self._spec

    def build(self, data: pd.DataFrame, **kwargs) -> PlotResult[Figure]:
        if not np.isfinite(data.to_numpy()).all():
            raise ValueError("association matrix must contain only finite values.")

        figure_options, extra = utils.safe_init(MatplotlibFigureOptions, **kwargs)
        if extra:
            raise TypeError(f"Unknown build options: {sorted(extra)}")

        figure, axes = plt.subplots(
            figsize=figure_options.figsize,
            layout="constrained",
        )
        sns.heatmap(
            data,
            ax=axes,
            **self.spec.options,
        )
        if self.spec.title is not None:
            axes.set_title(self.spec.title)
        return PlotResult(
            spec=self.spec,
            figure=figure,
            metadata={"axes": axes},
        )


__all__ = ["MatplotlibFigureOptions", "MatplotlibAssociationHeatmap"]
