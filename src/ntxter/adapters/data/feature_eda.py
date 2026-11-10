"""Pandas feature exploration and redundancy selection."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from numbers import Real

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
        min_columns: int = 2,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not df.columns.is_unique:
            raise ValueError("df column names must be unique.")
        if not isinstance(min_columns, int) or min_columns < 0:
            raise ValueError("min_columns must be a non-negative integer.")

        if columns is None:
            columns = (
                df.select_dtypes(include="number").columns.tolist()
                if numeric_only
                else df.columns.tolist()
            )
        else:
            columns = self._normalize_columns(df, columns)

        if len(columns) < min_columns:
            raise ValueError(
                f"At least {min_columns} column(s) are required for analysis."
            )
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


@dataclass(slots=True)
class PandasRedundancyResults:
    """The three independent redundancy analyses over the feature set.

    ``numeric`` contains non-categorical numeric columns, ``categorical``
    contains nominal columns (including numeric values stored as categories),
    and ``mixed`` contains both groups when both are present.  Each analysis
    selects and discards columns independently; this object deliberately does
    not collapse those decisions into one global selection.
    """

    numeric: PandasRedundancyResult
    categorical: PandasRedundancyResult
    mixed: PandasRedundancyResult
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]

    @property
    def numeric_categorical(self) -> PandasRedundancyResult:
        """Alias for the pure categorical analysis.

        The categorical group includes categorical columns whose values are
        numeric as well as text/object categories.  ARFS treats both as
        nominal variables for this analysis.
        """

        return self.categorical

    @property
    def outcome_columns(self) -> tuple[str, ...]:
        """Alias that makes the exclusion contract explicit to callers."""

        return self.excluded_columns

    @property
    def primary(self) -> PandasRedundancyResult:
        """Compatibility result equivalent to the former single analysis."""

        for analysis in (self.mixed, self.numeric, self.categorical):
            if analysis.selected_columns or not analysis.association_matrix.empty:
                return analysis
        return self.mixed

    @property
    def data(self) -> pd.DataFrame:
        return self.primary.data

    @property
    def association_matrix(self) -> pd.DataFrame:
        return self.primary.association_matrix

    @property
    def selected_columns(self) -> tuple[str, ...]:
        return self.primary.selected_columns

    @property
    def redundant_columns(self) -> tuple[str, ...]:
        return self.primary.redundant_columns

    @property
    def report(self) -> PandasSelectionReport:
        return self.primary.report

    @property
    def numeric_analysis(self) -> PandasRedundancyResult:
        return self.numeric

    @property
    def categorical_analysis(self) -> PandasRedundancyResult:
        return self.categorical

    @property
    def mixed_analysis(self) -> PandasRedundancyResult:
        return self.mixed

    @property
    def analyses(self) -> Mapping[str, PandasRedundancyResult]:
        """Return the analyses using stable names for iteration/reporting."""

        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "mixed": self.mixed,
        }

    @property
    def reports(self) -> Mapping[str, PandasSelectionReport]:
        """Return one selection report for each independent analysis."""

        return {
            name: analysis.report
            for name, analysis in self.analyses.items()
        }


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
    """Run unsupervised redundancy selection in three feature-type views.

    Outcomes are accepted only to exclude them from the feature universe. No
    outcome is passed to ARFS and no relevance/predictive selection is done.
    """

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
    ) -> PandasRedundancyResults:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not df.columns.is_unique:
            raise ValueError("df column names must be unique.")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a real number between 0 and 1.")

        candidate_columns = (
            df.columns.tolist()
            if columns is None
            else self._normalize_columns(df, columns)
        )
        outcome_columns = self._resolve_outcomes(df, outcomes)
        feature_columns = [
            column
            for column in candidate_columns
            if column not in outcome_columns
        ]
        if not feature_columns:
            raise ValueError(
                "At least one feature must remain after excluding outcomes."
            )

        declared_categorical = (
            []
            if categorical_columns is None
            else self._normalize_columns(df, categorical_columns)
        )
        declared_categorical = [
            column
            for column in declared_categorical
            if column in feature_columns
        ]

        data = self._validate_dataframe(
            df,
            feature_columns,
            min_columns=1,
        )
        # Validate supported dtypes once, before partitioning. This prevents a
        # datetime or other unsupported feature from being silently omitted.
        analysis_data = self._prepare_arfs_data(
            data,
            categorical_columns=declared_categorical,
        )
        numeric_columns, categorical_columns, mixed_columns = (
            self._partition_columns(analysis_data)
        )
        methods = ArfsAssociationFactory.create(
            numeric_method=numeric_method,
            categorical_method=categorical_method,
            mixed_method=mixed_method,
        )
        analyses = {
            "numeric": self._run_analysis(
                data=data,
                analysis_data=analysis_data,
                columns=numeric_columns,
                threshold=threshold,
                methods=methods,
                n_jobs=n_jobs,
            ),
            "categorical": self._run_analysis(
                data=data,
                analysis_data=analysis_data,
                columns=categorical_columns,
                threshold=threshold,
                methods=methods,
                n_jobs=n_jobs,
            ),
            "mixed": self._run_analysis(
                data=data,
                analysis_data=analysis_data,
                columns=mixed_columns,
                threshold=threshold,
                methods=methods,
                n_jobs=n_jobs,
            ),
        }

        return PandasRedundancyResults(
            numeric=analyses["numeric"],
            categorical=analyses["categorical"],
            mixed=analyses["mixed"],
            feature_columns=tuple(feature_columns),
            excluded_columns=tuple(outcome_columns),
        )

    @staticmethod
    def _resolve_outcomes(df: pd.DataFrame, outcomes) -> list[str]:
        if outcomes is None:
            return []
        resolved = PandasRedundancySelector._normalize_columns(df, outcomes)
        if len(resolved) != len(set(resolved)):
            raise ValueError("outcomes must not contain duplicates.")
        return resolved

    @staticmethod
    def _normalize_columns(df: pd.DataFrame, columns) -> list[str]:
        if isinstance(columns, str):
            columns = [columns]
        return utils.check_list_cols(df, columns)

    @staticmethod
    def _partition_columns(
        data: pd.DataFrame,
    ) -> tuple[list[str], list[str], list[str]]:
        numeric = data.select_dtypes(include="number").columns.tolist()
        categorical = data.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()
        mixed = (
            [
                column
                for column in data.columns
                if column in set(numeric).union(categorical)
            ]
            if numeric and categorical
            else []
        )
        return numeric, categorical, mixed

    def _run_analysis(
        self,
        *,
        data: pd.DataFrame,
        analysis_data: pd.DataFrame,
        columns: list[str],
        threshold: float,
        methods: ArfsAssociationMethods,
        n_jobs: int,
    ) -> PandasRedundancyResult:
        """Run one ARFS selector, handling groups too small for ARFS itself."""

        group = data.loc[:, columns]
        analysis_group = analysis_data.loc[:, columns]
        if not columns:
            return self._manual_result(group, columns)

        group = self._validate_dataframe(
            data,
            columns,
            min_columns=1,
        )
        if len(columns) == 1:
            return self._manual_result(group, columns)

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
        selector_pipeline.fit(analysis_group)
        selector = selector_pipeline.named_steps["redundancy"]
        selected_columns = tuple(selector.selected_features_)

        return PandasRedundancyResult(
            data=group.loc[:, list(selected_columns)],
            association_matrix=selector.assoc_matrix_,
            selected_columns=selected_columns,
            redundant_columns=tuple(selector.not_selected_features_),
            report=PandasSelectionReport.from_arfs(selector_pipeline),
        )

    @staticmethod
    def _manual_result(
        data: pd.DataFrame,
        columns: list[str],
    ) -> PandasRedundancyResult:
        """Represent zero/one-column groups without invoking ARFS."""

        association_matrix = pd.DataFrame(
            np.zeros((len(columns), len(columns))),
            index=columns,
            columns=columns,
        )
        summary = pd.DataFrame(
            {"feature": columns, "selected": [True] * len(columns)}
        )
        return PandasRedundancyResult(
            data=data.loc[:, columns],
            association_matrix=association_matrix,
            selected_columns=tuple(columns),
            redundant_columns=(),
            report=PandasSelectionReport(
                summary=summary,
                styled_summary=summary.style,
            ),
        )

    @staticmethod
    def _prepare_arfs_data(
        data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
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
        to_category = list(string_columns)
        if categorical_columns is not None:
            to_category.extend(categorical_columns)
        to_category = list(dict.fromkeys(to_category))
        if to_category:
            analysis_data[to_category] = analysis_data[to_category].astype(
                "category"
            )
        return analysis_data


__all__ = [
    "BasePandasFeatureAnalyzer",
    "ArfsAssociationFactory",
    "ArfsAssociationMethods",
    "PandasFeatureAnalyzer",
    "PandasRedundancyResult",
    "PandasRedundancyResults",
    "PandasRedundancySelector",
    "PandasSelectionReport",
]
