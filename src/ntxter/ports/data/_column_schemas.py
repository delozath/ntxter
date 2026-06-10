from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Mapping, Sequence


class ColumnDataType(StrEnum):
    """Supported logical data types for column schemas."""

    STRING = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"

_DATE_DATA_TYPES = {
    ColumnDataType.DATE,
    ColumnDataType.DATETIME,
}


class ColumnKind(StrEnum):
    """Generic column roles used by adapters and domain services."""

    IGNORE = "ignore"
    CATEGORY = "category"
    LONG_CATEGORY = "long_category"
    NO_CATEGORY = "no_category"
    POSITIVE = "positive"
    REAL = "real"
    TEXT = "text"
    EMAIL = "email"
    TELEPHONE = "telephone"
    UNIQUE = "unique"
    

def _validate_bool_flag(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"`{name}` must be a bool.")


def _validate_numeric_bound(name: str, value: int | float | None) -> None:
    if value is not None and not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be numeric or None.")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_date_like(value: Any) -> bool:
    return isinstance(value, (date, datetime))


def _category_type_error(data_type: ColumnDataType) -> str:
    labels = {
        ColumnDataType.STRING: "a string",
        ColumnDataType.INTEGER: "an integer",
        ColumnDataType.FLOAT: "a float",
        ColumnDataType.BOOLEAN: "a boolean",
        ColumnDataType.DATE: "a date or datetime",
        ColumnDataType.DATETIME: "a date or datetime",
        ColumnDataType.MIXED_DATE: "a date or datetime",
    }
    return labels[data_type]


def _matches_declared_type(data_type: ColumnDataType, value: Any) -> bool:
    validators = {
        ColumnDataType.STRING: lambda candidate: isinstance(candidate, str),
        ColumnDataType.INTEGER: _is_integer,
        ColumnDataType.FLOAT: _is_float_like,
        ColumnDataType.BOOLEAN: lambda candidate: isinstance(candidate, bool),
        ColumnDataType.DATE: _is_date_like,
        ColumnDataType.DATETIME: _is_date_like,
        ColumnDataType.MIXED_DATE: _is_date_like,
    }
    return validators[data_type](value)


@dataclass(frozen=True)
class ColumnSchema:
    """Base schema for a tabular column.

    Parameters
    ----------
    name : str
        Canonical column name used by the application.
    old_name : str | None, optional
        Source-system column name, by default None.
    data_type : ColumnDataType | str, optional
        Logical data type expected for the column, by default
        ColumnDataType.STRING.
    kind : str, optional
        Generic role of the column in the dataset, by default ColumnKind.TEXT.
    allowed : AllowedValues, optional
        Sequence of accepted values or "all" to disable membership validation,
        by default "all".
    nullable : bool, optional
        Whether None is accepted as a value, by default True.
    default : Any, optional
        Default value associated with the column, by default None.
    strip : bool, optional
        Whether string adapters should trim surrounding whitespace, by default
        True.

    Raises
    ------
    ValueError
        If the schema configuration is inconsistent.
    TypeError
        If a field receives an unsupported type.
    """

    name: str
    old_name: str | None = None
    data_type: ColumnDataType | str = ColumnDataType.STRING
    kind: str = ColumnKind.TEXT
    allowed: AllowedValues = "all"
    nullable: bool = True
    default: Any = None
    strip: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("`name` must be a non-empty string.")
        if self.old_name is not None and not isinstance(self.old_name, str):
            raise TypeError("`old_name` must be a string or None.")
        _validate_bool_flag("nullable", self.nullable)
        _validate_bool_flag("strip", self.strip)

        object.__setattr__(self, "data_type", ColumnDataType(self.data_type))
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("`kind` must be a non-empty string.")
        self._validate_allowed()
        self._validate_default()

    def validate_value(self, value: Any) -> None:
        """Validate a value against the base column rules.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        ValueError
            If the value is null in a required column or is not part of the
            allowed domain.
        """

        if value is None:
            if not self.nullable:
                raise ValueError(f"`{self.name}` does not allow null values.")
            return
        if self.allowed != "all" and value not in self.allowed:
            raise ValueError(f"`{self.name}` value is not allowed: {value!r}.")

    def _validate_allowed(self) -> None:
        if self.allowed == "all":
            return
        if isinstance(self.allowed, (str, bytes)) or not isinstance(
            self.allowed,
            Sequence,
        ):
            raise TypeError("`allowed` must be 'all' or a sequence of values.")
        if len(self.allowed) == 0:
            raise ValueError("`allowed` must contain at least one value.")

    def _validate_default(self) -> None:
        if self.default is None:
            if not self.nullable:
                raise ValueError("`default` cannot be None when nullable=False.")
            return
        self.validate_value(self.default)


@dataclass(frozen=True)
class NumericColumnSchema(ColumnSchema):
    """Schema for numeric columns with optional bounds.

    Parameters
    ----------
    min_value : int | float | None, optional
        Lower bound accepted for the column, by default None.
    max_value : int | float | None, optional
        Upper bound accepted for the column, by default None.
    min_inclusive : bool, optional
        Whether the lower bound is inclusive, by default True.
    max_inclusive : bool, optional
        Whether the upper bound is inclusive, by default True.

    Raises
    ------
    ValueError
        If the numeric bounds are inconsistent.
    TypeError
        If a bound or flag receives an unsupported type.
    """

    data_type: ColumnDataType | str = ColumnDataType.FLOAT
    kind: str = ColumnKind.CONTINUOUS
    min_value: int | float | None = None
    max_value: int | float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_numeric_bound("min_value", self.min_value)
        _validate_numeric_bound("max_value", self.max_value)
        _validate_bool_flag("min_inclusive", self.min_inclusive)
        _validate_bool_flag("max_inclusive", self.max_inclusive)
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("`min_value` cannot be greater than `max_value`.")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value == self.max_value
            and not (self.min_inclusive and self.max_inclusive)
        ):
            raise ValueError("Equal bounds must both be inclusive.")

    def validate_value(self, value: Any) -> None:
        """Validate numeric type, bounds, nullability, and allowed values.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If the value is not numeric.
        ValueError
            If the value falls outside the configured domain.
        """

        super().validate_value(value)
        if value is None:
            return
        if not isinstance(value, (int, float)):
            raise TypeError(f"`{self.name}` expects a numeric value.")
        self._validate_bounds(value)

    def _validate_bounds(self, value: int | float) -> None:
        if self.min_value is not None:
            valid = (
                value >= self.min_value
                if self.min_inclusive
                else value > self.min_value
            )
            if not valid:
                raise ValueError(f"`{self.name}` is below the minimum value.")
        if self.max_value is not None:
            valid = (
                value <= self.max_value
                if self.max_inclusive
                else value < self.max_value
            )
            if not valid:
                raise ValueError(f"`{self.name}` is above the maximum value.")


@dataclass(frozen=True)
class IntegerColumnSchema(NumericColumnSchema):
    """Schema for integer columns.

    Parameters
    ----------
    positive_only : bool, optional
        Whether values must be greater than zero, by default False.

    Raises
    ------
    TypeError
        If a non-integer value is validated.
    ValueError
        If positive_only is enabled and the value is not positive.
    """

    data_type: ColumnDataType | str = ColumnDataType.INTEGER
    positive_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_bool_flag("positive_only", self.positive_only)

    def validate_value(self, value: Any) -> None:
        """Validate integer type and inherited numeric rules.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If the value is not an integer.
        ValueError
            If the value is outside the configured domain.
        """

        super().validate_value(value)
        if value is None:
            return
        if not _is_integer(value):
            raise TypeError(f"`{self.name}` expects an integer value.")
        if self.positive_only and value <= 0:
            raise ValueError(f"`{self.name}` expects a positive integer.")


@dataclass(frozen=True)
class StringColumnSchema(ColumnSchema):
    """Schema for string columns with optional length limits.

    Parameters
    ----------
    min_length : int | None, optional
        Minimum accepted string length, by default None.
    max_length : int | None, optional
        Maximum accepted string length, by default None.

    Raises
    ------
    ValueError
        If the length limits are inconsistent.
    TypeError
        If a length limit or validated value receives an unsupported type.
    """

    data_type: ColumnDataType | str = ColumnDataType.STRING
    min_length: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for limit_name in ("min_length", "max_length"):
            limit = getattr(self, limit_name)
            if limit is not None and (not isinstance(limit, int) or limit < 0):
                raise ValueError(
                    f"`{limit_name}` must be a positive integer or None."
                )
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("`min_length` cannot be greater than `max_length`.")

    def validate_value(self, value: Any) -> None:
        """Validate string type, length, nullability, and allowed values.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If the value is not a string.
        ValueError
            If the value falls outside the configured domain.
        """

        super().validate_value(value)
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"`{self.name}` expects a string value.")
        if self.min_length is not None and len(value) < self.min_length:
            raise ValueError(f"`{self.name}` is shorter than `min_length`.")
        if self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"`{self.name}` is longer than `max_length`.")


@dataclass(frozen=True)
class CategoricalColumnSchema(ColumnSchema):
    """Schema for columns constrained to a closed value set.

    Notes
    -----
    This schema supports categorical values of any underlying type,
    including numeric categories represented as int or float.

    Raises
    ------
    ValueError
        If allowed values are not provided.
    """

    kind: str = ColumnKind.CATEGORY
    allowed: AllowedValues = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.allowed == "all":
            raise ValueError("Categorical columns require explicit allowed values.")

    def validate_value(self, value: Any) -> None:
        """Validate categorical values using base rules and declared type."""

        super().validate_value(value)
        if value is None:
            return

        if not _matches_declared_type(self.data_type, value):
            expected = _category_type_error(self.data_type)
            raise TypeError(f"`{self.name}` expects {expected} category.")


@dataclass(frozen=True)
class DateColumnSchema(ColumnSchema):
    """Schema for date or datetime columns.

    Parameters
    ----------
    accepted_formats : tuple[str, ...], optional
        Accepted text formats for adapters that parse string dates, by default
        an empty tuple.

    Raises
    ------
    TypeError
        If accepted_formats is not a tuple of strings.
    """

    data_type: ColumnDataType | str = ColumnDataType.DATE
    kind: str = ColumnKind.CONTINUOUS
    accepted_formats: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.data_type not in _DATE_DATA_TYPES:
            raise ValueError("DateColumnSchema requires a date-compatible data_type.")
        if not isinstance(self.accepted_formats, tuple) or not all(
            isinstance(item, str) for item in self.accepted_formats
        ):
            raise TypeError("`accepted_formats` must be a tuple of strings.")

    def validate_value(self, value: Any) -> None:
        """Validate date-like type, nullability, and allowed values.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If the value is not date-like.
        ValueError
            If the value is not part of the configured allowed values.
        """

        super().validate_value(value)
        if value is None:
            return
        if isinstance(value, str) and self.accepted_formats:
            for fmt in self.accepted_formats:
                try:
                    datetime.strptime(value, fmt)
                    return
                except ValueError:
                    continue
            raise ValueError(f"`{self.name}` does not match accepted date formats.")
        if not isinstance(value, (date, datetime)):
            raise TypeError(
                f"`{self.name}` expects a date, datetime, or formatted string."
            )


@dataclass(frozen=True)
class BooleanColumnSchema(ColumnSchema):
    """Schema for boolean columns.

    Raises
    ------
    TypeError
        If the value validated is not a bool.
    """

    data_type: ColumnDataType | str = ColumnDataType.BOOLEAN

    def validate_value(self, value: Any) -> None:
        """Validate boolean type, nullability, and allowed values.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If the value is not a bool.
        ValueError
            If the value is not part of the configured allowed values.
        """

        super().validate_value(value)
        if value is None:
            return
        if not isinstance(value, bool):
            raise TypeError(f"`{self.name}` expects a boolean value.")


@dataclass(frozen=True)
class IgnoredColumnSchema(ColumnSchema):
    """Schema for source columns intentionally excluded from processing."""

    kind: str = ColumnKind.IGNORE
    nullable: bool = True
    strip: bool = False

    def validate_value(self, value: Any) -> None:
        """Skip value validation for ignored columns.

        Parameters
        ----------
        value : Any
            Value intentionally ignored by this schema.
        """

        return None


@dataclass(frozen=True)
class TableSchema:
    """Collection of column schemas and optional replacement rules.

    Parameters
    ----------
    columns : tuple[ColumnSchema, ...]
        Column schemas that describe the table contract.
    replaces : Mapping[Any, Any] | None, optional
        Generic source-to-target value replacements applied by adapters, by
        default None.

    Raises
    ------
    ValueError
        If column names are duplicated.
    TypeError
        If columns or replacements receive unsupported types.
    """

    columns: tuple[ColumnSchema, ...]
    replaces: Mapping[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple) or not all(
            isinstance(column, ColumnSchema) for column in self.columns
        ):
            raise TypeError("`columns` must be a tuple of ColumnSchema instances.")
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Column names must be unique.")
        if self.replaces is not None and not isinstance(self.replaces, Mapping):
            raise TypeError("`replaces` must be a mapping or None.")

    def by_name(self) -> dict[str, ColumnSchema]:
        """Return schemas indexed by canonical column name.

        Returns
        -------
        dict[str, ColumnSchema]
            Mapping from canonical names to column schemas.
        """

        return {column.name: column for column in self.columns}
