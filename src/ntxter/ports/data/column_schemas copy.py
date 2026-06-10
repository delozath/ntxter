from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Mapping, Sequence

class ColumnDataType(StrEnum):
    """Supported logical data types for column schemas."""
    STR = "str"
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
    TEXT = "text"

class AllowedValues(StrEnum):
    ALL = "all"
    NULL = "NULL"
    POSITIVE = "positive"
    REAL = "real"
    EMAIL = "email"
    TELEPHONE = "telephone"
    UNIQUE = "unique"


@dataclass
class ColumnSchema:
    name: str
    old_name: str | None = None
    data_type: ColumnDataType | Sequence = ColumnDataType.STR
    kind: str = ColumnKind.TEXT
    allowed: Literal["all", "NULL"] | Sequence = AllowedValues.NULL
    nullable: bool = True
    default: Any = None

@dataclass
class NumericNoCatSchema(ColumnSchema):
    data_type: Literal[ColumnDataType.INT, ColumnDataType.FLOAT] = field(
        default=ColumnDataType.FLOAT, kw_only=True
    )
    allowed: AllowedValues
