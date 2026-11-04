"""Pandas feature exploration and redundancy selection."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from arfs.association import (
    correlation_ratio,
    cramer_v,
    symmetric_function,
    weighted_corr,
    weighted_theils_u,
)
from arfs.feature_selection import CollinearityThreshold, make_fs_summary
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from sklearn.pipeline import Pipeline

from ntxter.core import utils
from ntxter.core.base.classes import BaseRunnableClass


class BasePandasFeatureAnalyzer(BaseRunnableClass):
    """Provide shared DataFrame and column validation."""

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        columns=None,
        *,
        numeric_only: bool = False,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not df.columns.is_unique:
            raise ValueError("df column names must be unique.")

        if columns is None:
            columns = (
                df.select_dtypes(include="number").columns.tolist()
                if numeric_only
                else df.columns.tolist()
            )
        elif not isinstance(columns, list) or len(columns) < 2:
            raise ValueError("columns must be a list with at least two column names.")
        else:
            columns = utils.check_list_cols(df, columns)

        if len(columns) < 2:
            raise ValueError("At least two columns are required for analysis.")
        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates.")

        data = df.loc[:, columns]
        numeric = data.select_dtypes(include="number")
        numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric_values).any():
            raise ValueError("Numeric columns must not contain infinite values.")

        constant_columns = data.columns[data.nunique(dropna=True) < 2].tolist()
        if constant_columns:
            raise ValueError(
                "Analyzed columns must have at least two observed values: "
                f"{constant_columns}."
            )

        return data


class PandasFeatureAnalyzer(BasePandasFeatureAnalyzer):
    CORRELATION_METHODS = {"pearson", "spearman", "kendall"}

    def run(
        self,
        df: pd.DataFrame,
        columns=None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        if (
            not isinstance(method, str)
            or method not in self.CORRELATION_METHODS
        ):
            raise ValueError(
                "method must be 'pearson', 'spearman', or 'kendall'."
            )

        df = self._validate_dataframe(df, columns, numeric_only=True)
        return df.corr(method=method, numeric_only=True)


@dataclass(slots=True)
class PandasSelectionReport:
    """Tabular and styled ARFS feature-selection summary."""

    summary: pd.DataFrame
    styled_summary: Styler

    @classmethod
    def from_arfs(cls, selector_pipeline: Pipeline) -> "PandasSelectionReport":
        styled_summary = make_fs_summary(selector_pipeline)
        summary = styled_summary.data.rename(
            columns={
                "predictor": "feature",
                "redundancy": "selected",
            }
        ).copy()
        summary["selected"] = summary["selected"].eq(1)
        return cls(
            summary=summary,
            styled_summary=styled_summary,
        )


@dataclass(slots=True)
class PandasRedundancyResult:
    """Output produced by ARFS collinearity selection."""

    data: pd.DataFrame
    association_matrix: pd.DataFrame
    selected_columns: tuple[str, ...]
    redundant_columns: tuple[str, ...]
    report: PandasSelectionReport


@dataclass(frozen=True, slots=True)
class ArfsAssociationMethods:
    """ARFS callables used for each pair of feature types."""

    numeric: Callable
    categorical: Callable
    mixed: Callable


class ArfsAssociationFactory:
    """Build type-specific association methods supported by ARFS."""

    NUMERIC_METHODS = {"pearson", "spearman"}
    CATEGORICAL_METHODS = {
        "cramer_v": cramer_v,
        "theils_u": weighted_theils_u,
    }
    MIXED_METHODS = {"correlation_ratio": correlation_ratio}

    @classmethod
    def create(
        cls,
        numeric_method: str = "spearman",
        categorical_method: str = "cramer_v",
        mixed_method: str = "correlation_ratio",
    ) -> ArfsAssociationMethods:
        if numeric_method not in cls.NUMERIC_METHODS:
            raise ValueError("numeric_method must be 'pearson' or 'spearman'.")
        if categorical_method not in cls.CATEGORICAL_METHODS:
            raise ValueError(
                "categorical_method must be 'cramer_v' or 'theils_u'."
            )
        if mixed_method not in cls.MIXED_METHODS:
            raise ValueError("mixed_method must be 'correlation_ratio'.")

        numeric = symmetric_function(
            partial(weighted_corr, method=numeric_method)
        )
        return ArfsAssociationMethods(
            numeric=numeric,
            categorical=cls.CATEGORICAL_METHODS[categorical_method],
            mixed=cls.MIXED_METHODS[mixed_method],
        )


class PandasRedundancySelector(BasePandasFeatureAnalyzer):
    """Remove redundant pandas columns with ARFS associations."""

    def run(
        self,
        df: pd.DataFrame,
        columns=None,
        threshold: float = 0.8,
        numeric_method: str = "spearman",
        categorical_method: str = "cramer_v",
        mixed_method: str = "correlation_ratio",
        n_jobs: int = 1,
    ) -> PandasRedundancyResult:
        data = self._validate_dataframe(df, columns)
        analysis_data = self._prepare_arfs_data(data)
        methods = ArfsAssociationFactory.create(
            numeric_method=numeric_method,
            categorical_method=categorical_method,
            mixed_method=mixed_method,
        )
        selector_pipeline = Pipeline(
            steps=[
                (
                    "redundancy",
                    CollinearityThreshold(
                        threshold=threshold,
                        method="association",
                        n_jobs=n_jobs,
                        num_num_assoc=methods.numeric,
                        nom_nom_assoc=methods.categorical,
                        nom_num_assoc=methods.mixed,
                    ),
                )
            ]
        )
        selector_pipeline.fit(analysis_data)
        selector = selector_pipeline.named_steps["redundancy"]
        selected_data = data.loc[:, selector.selected_features_]

        return PandasRedundancyResult(
            data=selected_data,
            association_matrix=selector.assoc_matrix_,
            selected_columns=tuple(selector.selected_features_),
            redundant_columns=tuple(selector.not_selected_features_),
            report=PandasSelectionReport.from_arfs(selector_pipeline),
        )

    @staticmethod
    def _prepare_arfs_data(data: pd.DataFrame) -> pd.DataFrame:
        supported_columns = data.select_dtypes(
            include=["number", "object", "category", "bool", "string"]
        ).columns
        unsupported_columns = data.columns.difference(supported_columns).tolist()
        if unsupported_columns:
            raise ValueError(
                "ARFS redundancy selection does not support columns: "
                f"{unsupported_columns}."
            )

        analysis_data = data.copy()
        string_columns = analysis_data.select_dtypes(include="string").columns
        analysis_data[string_columns] = analysis_data[string_columns].astype(
            "category"
        )
        return analysis_data


__all__ = [
    "BasePandasFeatureAnalyzer",
    "ArfsAssociationFactory",
    "ArfsAssociationMethods",
    "PandasFeatureAnalyzer",
    "PandasRedundancyResult",
    "PandasRedundancySelector",
    "PandasSelectionReport",
]
