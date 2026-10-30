"""Profile, validate, and describe DataFrames in separate stages.

Dependencies: pip install 'pandera[pandas]==0.35.0' pyyaml

    contract = DataProfiler().run(reference_df)
    contract.columns['age'].checks['in_range'] = {
        'min_value': 18, 'max_value': 100
    }
    schema_yaml = contract.to_yaml()
    report = DataValidator(schema_yaml).run(new_df)
    report['valid']
    report['failures']
    report['conflicting_columns']
    report['conflicting_rows']

    # A later stage describes the approved DataFrame with the same contract.
    descriptive = DataDescriber().run(approved_df, schema_yaml)
    descriptive['numeric']
    descriptive['categories']
    descriptive['information']

All inputs and outputs stay in memory. Storage and schema review are external.
"""
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.config import ValidationDepth, config_context


def _yaml_value(value):
    """Convert inferred NumPy/pandas parameters to YAML-compatible values."""
    if isinstance(value, dict):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ColumnContract:
    """Column settings and built-in Pandera checks; no custom field validation.

    checks maps check names to a single argument or keyword arguments:
        {'greater_than_or_equal_to': 0}
        {'in_range': {'min_value': 18, 'max_value': 100}}
        {'isin': ['A', 'B', 'C'], 'str_matches': '^[A-Z]$'}
    """

    dtype: str
    nullable: bool = False
    required: bool = True
    unique: bool = False
    checks: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> pa.Column:
        checks = [
            getattr(pa.Check, name)(**arguments)
            if isinstance(arguments, dict)
            else getattr(pa.Check, name)(arguments)
            for name, arguments in self.checks.items()
        ]
        return pa.Column(
            self.dtype, checks=checks, nullable=self.nullable,
            required=self.required, unique=self.unique,
        )


@dataclass
class DataContract:
    """Editable contract with in-memory dict, YAML, and Pandera conversions.

    YAML follows these dataclass fields, with checks nested under each column.
    Use built-in Pandera check names and YAML-compatible check arguments.
    No dates, approval state, file operations, or custom validators are added.
    """

    columns: dict[str, ColumnContract] = field(default_factory=dict)
    strict: bool = True
    ordered: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict) -> 'DataContract':
        return cls(**{
            **data,
            'columns': {
                name: ColumnContract(**settings)
                for name, settings in data['columns'].items()
            },
        })

    @classmethod
    def from_yaml(cls, text: str) -> 'DataContract':
        return cls.from_dict(yaml.safe_load(text))

    def to_schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {name: column.to_schema() for name, column in self.columns.items()},
            strict=self.strict, ordered=self.ordered, unique_column_names=True,
        )


class DataProfiler:
    """Infer a draft with Pandera and return an editable DataContract."""

    def run(
        self,
        df: pd.DataFrame,
        min_categories: int = 10,
        *,
        infer_ranges: bool = False,
        infer_categories: bool = False,
        strict: bool = True,
    ) -> DataContract:
        """Infer dtypes, required columns, and nullability from reference data.

        infer_ranges: include Pandera's observed numeric/datetime bounds.
        infer_categories: infer category dtype and allowed values for any
            column with 0 < non-null cardinality < min_categories.
            min_categories is the exclusive upper bound (legacy parameter name).
        strict: reject additional columns when validating.

        Observed bounds, categories, and nullability require human review.
        All-null columns retain their inferred dtype. Category order is not inferred.
        The index is a record identifier, not a validated schema component.
        """
        if df.empty:
            raise ValueError('Schema inference requires rows and columns.')
        if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
            raise ValueError('Reference columns must have unique string names.')

        schema = pa.infer_schema(df)

        for name, column in schema.columns.items():
            column.checks = [
                check for check in column.checks
                if infer_ranges and check.name != 'isin'
            ]
            if infer_categories:
                categories = df[name].dropna().unique().tolist()
                if 0 < len(categories) < min_categories:
                    column.dtype = 'category'
                    # Membership replaces numeric bounds for categorical data.
                    column.checks = [pa.Check.isin(categories)]

        columns = {
            name: ColumnContract(
                dtype=str(column.dtype),
                nullable=bool(column.nullable),
                checks={
                    check.name: _yaml_value(check.statistics)
                    for check in column.checks
                },
            )
            for name, column in schema.columns.items()
        }
        return DataContract(columns=columns, strict=strict)


class DataValidator:
    """Validate original values against a reviewed schema without re-inference."""

    def __init__(self, contract: DataContract | str | dict):
        """Accept a contract dataclass, its YAML text, or its dictionary."""
        if isinstance(contract, str):
            contract = DataContract.from_yaml(contract)
        elif isinstance(contract, dict):
            contract = DataContract.from_dict(contract)
        self.schema = contract.to_schema()
        self._report: str | None = None

    def run(self, df: pd.DataFrame) -> dict:
        """Return compliance, failure details, and original conflicting records.

        valid: whether the complete DataFrame satisfies the schema.
        failures: Pandera failure cases plus zero-based row_position and the
            original row_index. Missing positions denote global failures.
        conflicting_columns: names implicated in failures, including missing
            columns. DataFrame-wide failures may not identify any column.
        conflicting_row_positions: sorted positions of explicitly failed rows.
        conflicting_rows: those rows with their original values and index.
        global_failures: failures that cannot be attributed to specific rows,
            such as dtype mismatches or missing columns. Rows not listed in
            conflicting_rows are not certified as valid when these exist.

        Always uses lazy validation to collect all failures reported by Pandera.
        Schema/configuration exceptions propagate to the calling framework.
        """
        # Positional IDs remain unambiguous with duplicate labels or MultiIndex.
        work = df.reset_index(drop=True)
        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            try:
                self.schema.validate(work, lazy=True, inplace=False)
            except pa.errors.SchemaErrors as error:
                failures = error.failure_cases.copy()
            else:
                failures = pd.DataFrame(columns=[
                    'schema_context', 'column', 'check', 'check_number',
                    'failure_case', 'index',
                ])

        # Structural failures store column names in failure_case, not column.
        structural = failures['check'].isin([
            'column_in_dataframe', 'column_in_schema', 'column_ordered',
        ])
        failures.loc[structural, 'column'] = failures.loc[structural, 'failure_case']
        duplicated = failures['check'].eq('dataframe_column_labels_unique')
        if duplicated.any():
            failures = pd.concat([
                failures.loc[~duplicated],
                failures.loc[duplicated].explode('failure_case').assign(
                    column=lambda frame: frame['failure_case']
                ),
            ], ignore_index=True)

        failures = failures.rename(columns={'index': 'row_position'}).reset_index(drop=True)
        failures['row_position'] = pd.array(failures['row_position'], dtype='Int64')
        failures['row_index'] = pd.Series([
            df.index[int(position)] if pd.notna(position) else None
            for position in failures['row_position']
        ], dtype=object)
        positions = sorted(failures['row_position'].dropna().unique().tolist())
        report = {
            'valid': failures.empty,
            'failures': failures,
            'conflicting_columns': failures['column'].dropna().unique().tolist(),
            'conflicting_row_positions': positions,
            'conflicting_rows': df.iloc[positions].copy(),
            'global_failures': failures.loc[failures['row_position'].isna()].copy(),
        }
        self._report = self._summarize(report)
        return report

    @staticmethod
    def _summarize(report: dict, max_examples: int = 5) -> str:
        """Build a readable summary of a run() report.

        Failures are grouped by column and check with their count and up to
        max_examples distinct failure cases.
        """
        failures = report['failures']
        lines = [
            f"Valid: {report['valid']}",
            f"Failures: {len(failures)} "
            f"({len(report['global_failures'])} global, "
            f"{len(report['conflicting_row_positions'])} rows)",
            f"Conflicting columns: {report['conflicting_columns'] or '-'}",
        ]
        if failures.empty:
            return '\n'.join(lines)

        grouped = (
            failures.assign(column=failures['column'].fillna('<dataframe>'))
            .groupby(['column', 'check'], sort=False)
            .agg(
                count=('failure_case', 'size'),
                rows=('row_position', 'count'),
                examples=('failure_case', lambda cases: list(
                    pd.unique(cases.astype(str))[:max_examples]
                )),
            )
            .reset_index()
        )
        lines += ['', grouped.to_string(index=False)]
        return '\n'.join(lines)

    def summarize(self) -> str | None:
        """Print and return the summary of the last run(); None before any run."""
        print(self._report)
        return self._report

    def apply_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for name, column in self.schema.columns.items():
            if name not in result.columns:
                continue

            if str(column.dtype) != "category":
                continue

            isin = next(
                (check for check in column.checks or []
                if check.name == "isin"),
                None,
            )

            if isin is None:
                continue

            categories = isin.statistics["allowed_values"]

            invalid = (
                result[name].notna()
                & ~result[name].isin(categories)
            )

            if invalid.any():
                values = result.loc[invalid, name].unique().tolist()
                raise ValueError(
                    f"Column '{name}' contains invalid categories: {values}"
                )

            result[name] = result[name].astype(
                pd.CategoricalDtype(
                    categories=categories,
                    ordered=False,
                )
            )

        return result


class DataDescriber:
    """Describe approved observations without inferring or validating a schema.

    ``run`` returns four DataFrames: ``variables`` (availability and types),
    ``numeric`` (central tendency and dispersion), ``categories`` (counts and
    row-level proportions), and ``information`` (categorical heterogeneity and
    descriptive signals). Tables other than ``categories`` are indexed by
    column name. No rows or variables are removed from the input.

    Signals describe this sample; entropy and dispersion alone cannot establish
    relevance to an outcome. Thresholds are optional review criteria, not
    automatic exclusion rules. The contract supplies columns, kinds and
    category levels; data is described, not validated against it.
    """

    _VARIABLE_COLUMNS = (
        'column', 'dtype', 'kind', 'n_total', 'n_observed', 'n_missing',
        'missing_ratio', 'n_distinct',
    )
    _NUMERIC_COLUMNS = (
        'column', 'n_finite', 'n_nonfinite', 'mean', 'median', 'std', 'var',
        'min', 'q1', 'q3', 'max', 'iqr',
    )
    _CATEGORY_COLUMNS = (
        'column', 'category', 'count', 'proportion_observed', 'proportion_total',
    )
    _INFORMATION_COLUMNS = (
        'column', 'n_categories', 'entropy', 'normalized_entropy', 'gini',
        'kl_uniform', 'dominant_ratio', 'signals',
    )

    def run(
        self,
        df: pd.DataFrame,
        profile: DataContract | str | dict,
        *,
        min_observations: int = 2,
        max_missing_ratio: float | None = None,
        min_normalized_entropy: float | None = None,
        min_category_count: int | None = None,
        max_distinct_ratio: float | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return descriptive tables without changing values, dtypes, or index.

        ``profile`` is a contract dataclass, its YAML text, or its dictionary.
        Its columns are described in contract order; absent optional columns
        are skipped and other columns of ``df`` are ignored. The kind follows
        the contract dtype: category, boolean, string and object dtypes are
        categorical, real numeric dtypes numeric, and others receive
        availability counts only.

        Category levels come from the contract ``isin`` check, including unused
        levels; booleans use False/True, other columns their observed levels in
        first appearance order. Values outside contract levels raise ValueError.
        Object levels use Python equality, so 1, 1.0 and True form one level.
        Unhashable values raise TypeError naming the column.
        ``proportion_observed`` divides by non-null observations, whereas
        ``proportion_total`` divides by all rows. These are unweighted row-level
        proportions; interpreting them as prevalence requires an appropriate
        observation unit and denominator.

        Entropy (natural logarithms), Gini impurity and dominant ratio use only
        non-null observations. Zero probabilities contribute zero. For K levels,
        normalized entropy is H/log(K), defined as zero when K=1 and observed.
        KL is relative to uniform over that same domain: log(K)-H; it supplies
        no independent evidence beyond H and K. With no observations, these
        metrics and observed proportions are NaN.

        Numeric summaries use finite non-null values, sample std/variance
        (ddof=1), and linearly interpolated quartiles. Infinities are counted
        separately as ``n_nonfinite`` and flagged, never changed in the input.
        Fewer than two finite values give NaN std/variance. A numeric column is
        constant when its finite values are. Empty inputs retain the output
        table structure.

        ``min_observations`` flags nonempty samples below that count (finite
        observations for numeric columns, so all-infinite columns are flagged).
        Optional thresholds flag missing ratios above their maximum, normalized
        entropy below its minimum, positive category counts below their
        minimum, or categorical distinct/observed ratios above their maximum
        (identifiers or free text). All inequalities are strict; absent
        category levels are reported separately.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame.')
        if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
            raise ValueError('Columns must have unique string names.')
        for name, value in (
            ('min_observations', min_observations),
            ('min_category_count', min_category_count),
        ):
            if value is None and name == 'min_category_count':
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1
            ):
                raise ValueError(f'{name} must be a positive integer.')
        for name, value in (
            ('max_missing_ratio', max_missing_ratio),
            ('min_normalized_entropy', min_normalized_entropy),
            ('max_distinct_ratio', max_distinct_ratio),
        ):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not 0 <= value <= 1
            ):
                raise ValueError(f'{name} must be a number between 0 and 1.')
        if isinstance(profile, str):
            profile = DataContract.from_yaml(profile)
        elif isinstance(profile, dict):
            profile = DataContract.from_dict(profile)

        missing = [
            name for name, column in profile.columns.items()
            if column.required and name not in df.columns
        ]
        if missing:
            raise KeyError(f'Required profile columns not in data: {missing}')

        variables, numeric, categories, information = [], [], [], []
        n_total = len(df)
        for name, column in profile.columns.items():
            if name not in df.columns:
                continue
            series = df[name]
            observed = series.dropna()
            n_observed = len(observed)
            n_missing = n_total - n_observed
            try:
                n_distinct = int(observed.nunique())
            except TypeError as error:
                raise TypeError(f'Column {name!r} contains unhashable values.') from error
            missing_ratio = n_missing / n_total if n_total else np.nan
            kind = self._kind(pd.api.types.pandas_dtype(column.dtype))
            variables.append({
                'column': name, 'dtype': str(series.dtype), 'kind': kind,
                'n_total': n_total, 'n_observed': n_observed,
                'n_missing': n_missing, 'missing_ratio': missing_ratio,
                'n_distinct': n_distinct,
            })
            signals = []
            if not n_observed:
                signals.append('no_observations')
            elif n_distinct == 1 and kind != 'numeric':
                signals.append('constant')
            if max_missing_ratio is not None and missing_ratio > max_missing_ratio:
                signals.append('high_missing_ratio')

            metrics = dict.fromkeys(self._INFORMATION_COLUMNS[1:-1], np.nan)
            n_usable = n_observed
            if kind == 'numeric':
                summary = self._numeric_summary(observed)
                numeric.append({'column': name, **summary})
                n_usable = summary['n_finite']
                if n_usable and summary['min'] == summary['max']:
                    signals.append('constant')
                if summary['n_nonfinite']:
                    signals.append('non_finite_values')
            elif kind == 'categorical':
                counts = self._category_counts(name, series, column)
                metrics['n_categories'] = len(counts)
                if (
                    max_distinct_ratio is not None and n_observed
                    and n_distinct / n_observed > max_distinct_ratio
                ):
                    signals.append('high_cardinality')
                categories.extend({
                    'column': name, 'category': category, 'count': int(count),
                    'proportion_observed': count / n_observed if n_observed else np.nan,
                    'proportion_total': count / n_total if n_total else np.nan,
                } for category, count in counts.items())
                if n_observed:
                    probabilities = counts.to_numpy(dtype=float) / n_observed
                    positive = probabilities[probabilities > 0]
                    log_k = float(np.log(len(counts)))
                    # Equal integer counts are exactly uniform. Preserve this
                    # boundary so floating-point error cannot flag H/log(K)<1.
                    # Adding 0.0 turns the -0.0 of a single level into 0.0.
                    entropy = log_k if (counts == counts.iloc[0]).all() else float(
                        np.clip(-np.sum(positive * np.log(positive)), 0.0, log_k)
                    ) + 0.0
                    metrics.update(
                        entropy=entropy,
                        normalized_entropy=entropy / log_k if log_k else 0.0,
                        gini=float(1 - np.sum(probabilities ** 2)),
                        kl_uniform=max(0.0, log_k - entropy),
                        dominant_ratio=float(probabilities.max()),
                    )
                    if (
                        min_normalized_entropy is not None
                        and metrics['normalized_entropy'] < min_normalized_entropy
                    ):
                        signals.append('low_normalized_entropy')
                    if min_category_count is not None and (
                        (counts > 0) & (counts < min_category_count)
                    ).any():
                        signals.append('rare_categories')
                    if (counts == 0).any():
                        signals.append('unobserved_categories')
            if n_observed and n_usable < min_observations:
                signals.append('insufficient_observations')
            information.append({'column': name, **metrics, 'signals': tuple(signals)})

        return {
            'variables': pd.DataFrame(variables, columns=self._VARIABLE_COLUMNS).set_index('column'),
            'numeric': pd.DataFrame(numeric, columns=self._NUMERIC_COLUMNS).set_index('column'),
            'categories': pd.DataFrame(categories, columns=self._CATEGORY_COLUMNS),
            'information': pd.DataFrame(information, columns=self._INFORMATION_COLUMNS).set_index('column'),
        }

    @staticmethod
    def _kind(dtype) -> str:
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            return 'categorical'
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
            return 'numeric'
        return 'other'

    @staticmethod
    def _category_counts(
        name: str, series: pd.Series, column: ColumnContract
    ) -> pd.Series:
        counts = series.value_counts(dropna=True, sort=False)
        # Profiler writes {'allowed_values': [...]}; hand-written YAML a list.
        levels = column.checks.get('isin')
        if isinstance(levels, dict):
            levels = levels['allowed_values']
        if levels is None and pd.api.types.is_bool_dtype(
            pd.api.types.pandas_dtype(column.dtype)
        ):
            levels = [False, True]
        if levels is None:
            return counts
        counts = counts[counts > 0]
        outside = [value for value in counts.index if value not in levels]
        if outside:
            raise ValueError(f'Column {name!r} has values outside profile levels: {outside}')
        return counts.reindex(levels, fill_value=0)

    @staticmethod
    def _numeric_summary(observed: pd.Series) -> dict:
        finite = observed.loc[np.isfinite(observed)]
        summary = {
            'n_finite': len(finite), 'n_nonfinite': len(observed) - len(finite),
            **dict.fromkeys(DataDescriber._NUMERIC_COLUMNS[3:], np.nan),
        }
        if not finite.empty:
            quartiles = finite.quantile([0.25, 0.75], interpolation='linear')
            summary.update(
                mean=float(finite.mean()), median=float(finite.median()),
                std=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                var=float(finite.var(ddof=1)) if len(finite) > 1 else np.nan,
                min=float(finite.min()), q1=float(quartiles.iloc[0]),
                q3=float(quartiles.iloc[1]), max=float(finite.max()),
                iqr=float(quartiles.iloc[1] - quartiles.iloc[0]),
            )
        return summary
