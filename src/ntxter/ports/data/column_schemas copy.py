from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Mapping, List

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

def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, dict))


def _validate_allowed_value(allowed: str | List, options: tuple[AllowedValues, ...], schema: str) -> None:
    if _is_sequence(allowed):
        return
    if allowed not in options:
        valid_options = ", ".join(option.value for option in options)
        raise ValueError(f"{schema} can only be used with `{valid_options}` allowed values")


@dataclass
class ColumnSchema:
    name: str
    old_name: str | None = None
    data_type: str | List = ColumnDataType.STR
    kind: str = ColumnKind.TEXT
    allowed: str | List = AllowedValues.NULL
    nullable: bool = True
    default: Any = None


@dataclass
class NumericNoCatSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.FLOAT
    kind: str = ColumnKind.NO_CATEGORY
    allowed: str | List = AllowedValues.ALL
    min: int | float | None = None
    max: int | float | None = None
    
    def __post_init__(self):
        if not self.data_type in (ColumnDataType.INT, ColumnDataType.FLOAT):
            raise ValueError("NumericNoCatSchema can only be used with `int` or `float` data types")
        if not self.kind==ColumnKind.NO_CATEGORY:
            raise ValueError("NumericNoCatSchema can only be used with `no_category` kind")
        if self.allowed not in (
            AllowedValues.ALL,
            AllowedValues.NULL,
            AllowedValues.POSITIVE,
            AllowedValues.REAL,
            AllowedValues.UNIQUE
         ):
            raise ValueError("NumericNoCatSchema can only be used with `all`, `null`, `positive`, `real` and `unique` allowed values")


@dataclass
class NumericCatSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.FLOAT
    kind: str = ColumnKind.CATEGORY
    allowed: str | List = AllowedValues.NULL
    min: int | float | None = None
    max: int | float | None = None
    
    def __post_init__(self):
        if not self.data_type in (ColumnDataType.INT, ColumnDataType.FLOAT):
            raise ValueError("NumericNoCatSchema can only be used with `int` or `float` data types")
        if not self.kind==ColumnKind.CATEGORY:
            raise ValueError("NumericNoCatSchema can only be used with `no_category` kind")
        if not isinstance(self.allowed, (list, tuple, dict)):
            raise ValueError("NumericCatSchema mus be `list`, `tuple` or `dict`.")


@dataclass
class TextSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.STR
    kind: str = ColumnKind.TEXT
    allowed: str | List = AllowedValues.NULL
    min_length: int | None = 0
    max_length: int | None = 100
    strip: bool = True

    def __post_init__(self):
        if not self.data_type==ColumnDataType.STR:
            raise ValueError("TextSchema can only be used with `str` data type")
        if not self.kind==ColumnKind.TEXT:
            raise ValueError("TextSchema can only be used with `text` kind")
        _validate_allowed_value(
            self.allowed,
            (
                AllowedValues.ALL,
                AllowedValues.NULL,
                AllowedValues.EMAIL,
                AllowedValues.TELEPHONE,
                AllowedValues.UNIQUE,
            ),
            "TextSchema",
        )
        if self.min_length is not None and self.min_length < 0:
            raise ValueError("TextSchema `min_length` must be greater than or equal to zero")
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("TextSchema `max_length` must be greater than or equal to zero")
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("TextSchema `min_length` cannot be greater than `max_length`")
        if not isinstance(self.strip, bool):
            raise ValueError("TextSchema `strip` must be `bool`")


@dataclass
class TextCatSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.STR
    kind: str = ColumnKind.CATEGORY
    allowed: str | List = field(default_factory=list)
    min_length: int | None = None
    max_length: int | None = None

    def __post_init__(self):
        if not self.data_type==ColumnDataType.STR:
            raise ValueError("TextCatSchema can only be used with `str` data type")
        if not self.kind==ColumnKind.CATEGORY:
            raise ValueError("TextCatSchema can only be used with `category` kind")
        if not isinstance(self.allowed, (list, tuple, dict)):
            raise ValueError("TextCatSchema must be `list`, `tuple` or `dict`.")
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("TextCatSchema `min_length` cannot be greater than `max_length`")


@dataclass
class LongTextCatSchema(TextCatSchema):
    kind: str = ColumnKind.LONG_CATEGORY

    def __post_init__(self):
        if not self.data_type==ColumnDataType.STR:
            raise ValueError("LongTextCatSchema can only be used with `str` data type")
        if not self.kind==ColumnKind.LONG_CATEGORY:
            raise ValueError("LongTextCatSchema can only be used with `long_category` kind")
        if not isinstance(self.allowed, (list, tuple, dict)):
            raise ValueError("LongTextCatSchema must be `list`, `tuple` or `dict`.")


@dataclass
class DateSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.DATE
    kind: str = ColumnKind.NO_CATEGORY
    allowed: str | List = AllowedValues.NULL
    min: date | datetime | None = None
    max: date | datetime | None = None
    formats: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.data_type in _DATE_DATA_TYPES:
            raise ValueError("DateSchema can only be used with `date` or `datetime` data types")
        if not self.kind==ColumnKind.NO_CATEGORY:
            raise ValueError("DateSchema can only be used with `no_category` kind")
        _validate_allowed_value(
            self.allowed,
            (AllowedValues.ALL, AllowedValues.NULL, AllowedValues.UNIQUE),
            "DateSchema",
        )
        if self.min is not None and not isinstance(self.min, (date, datetime)):
            raise ValueError("DateSchema `min` must be `date`, `datetime` or `None`")
        if self.max is not None and not isinstance(self.max, (date, datetime)):
            raise ValueError("DateSchema `max` must be `date`, `datetime` or `None`")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("DateSchema `min` cannot be greater than `max`")
        if not isinstance(self.formats, list) or not all(isinstance(fmt, str) for fmt in self.formats):
            raise ValueError("DateSchema `formats` must be a list of strings")


@dataclass
class BoolSchema(ColumnSchema):
    data_type: str | List = ColumnDataType.BOOL
    kind: str = ColumnKind.CATEGORY
    allowed: str | List = AllowedValues.NULL

    def __post_init__(self):
        if not self.data_type==ColumnDataType.BOOL:
            raise ValueError("BoolSchema can only be used with `bool` data type")
        if not self.kind in (ColumnKind.CATEGORY, ColumnKind.NO_CATEGORY):
            raise ValueError("BoolSchema can only be used with `category` or `no_category` kind")
        _validate_allowed_value(
            self.allowed,
            (AllowedValues.ALL, AllowedValues.NULL, AllowedValues.UNIQUE),
            "BoolSchema",
        )

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_MEXICAN_STATES = (
    "Aguascalientes",
    "Baja California",
    "Baja California Sur",
    "Campeche",
    "Chiapas",
    "Chihuahua",
    "Ciudad de México",
    "Coahuila",
    "Colima",
    "Durango",
    "Guanajuato",
    "Guerrero",
    "Hidalgo",
    "Jalisco",
    "México",
    "Michoacán",
    "Morelos",
    "Nayarit",
    "Nuevo León",
    "Oaxaca",
    "Puebla",
    "Querétaro",
    "Quintana Roo",
    "San Luis Potosí",
    "Sinaloa",
    "Sonora",
    "Tabasco",
    "Tamaulipas",
    "Tlaxcala",
    "Veracruz",
    "Yucatán",
    "Zacatecas",
)


@dataclass
class EmailSchema(TextSchema):
    allowed: str | List = AllowedValues.EMAIL

    def __post_init__(self):
        super().__post_init__()
        if self.allowed != AllowedValues.EMAIL:
            raise ValueError("EmailSchema can only be used with `email` allowed value")

    def validate_value(self, value: Any) -> None:
        if value is None:
            if not self.nullable:
                raise ValueError("EmailSchema does not allow null values")
            return
        if not isinstance(value, str):
            raise TypeError("EmailSchema value must be `str`")
        candidate = value.strip() if self.strip else value
        if not _EMAIL_PATTERN.fullmatch(candidate):
            raise ValueError("EmailSchema value must be a valid email address")


@dataclass
class MexicanStateSchema(TextCatSchema):
    allowed: str | List = field(default_factory=lambda: list(_MEXICAN_STATES))

    def __post_init__(self):
        super().__post_init__()
        values = self.allowed.values() if isinstance(self.allowed, dict) else self.allowed
        invalid_states = sorted(set(values).difference(_MEXICAN_STATES))
        if invalid_states:
            raise ValueError(
                "MexicanStateSchema allowed values must be valid Mexican states: "
                + ", ".join(invalid_states)
            )

    def validate_value(self, value: Any) -> None:
        if value is None:
            if not self.nullable:
                raise ValueError("MexicanStateSchema does not allow null values")
            return
        if not isinstance(value, str):
            raise TypeError("MexicanStateSchema value must be `str`")
        if value not in set(self.allowed):
            raise ValueError("MexicanStateSchema value must be a valid Mexican state")


@dataclass
class IgnoreSchema(ColumnSchema):
    data_type: str | List = (
        ColumnDataType.STR,
        ColumnDataType.INT,
        ColumnDataType.FLOAT,
        ColumnDataType.BOOL,
        ColumnDataType.DATE,
        ColumnDataType.DATETIME,
    )
    kind: str = ColumnKind.IGNORE
    allowed: str | List = AllowedValues.ALL

    def __post_init__(self):
        if not self.kind==ColumnKind.IGNORE:
            raise ValueError("IgnoreSchema can only be used with `ignore` kind")
