"""Create editable dataclass contracts and validate DataFrames with Pandera.

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
        return {
            'valid': failures.empty,
            'failures': failures,
            'conflicting_columns': failures['column'].dropna().unique().tolist(),
            'conflicting_row_positions': positions,
            'conflicting_rows': df.iloc[positions].copy(),
            'global_failures': failures.loc[failures['row_position'].isna()].copy(),
        }