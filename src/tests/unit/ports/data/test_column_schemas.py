from __future__ import annotations

import pytest

from ntxter.ports.data.column_schemas import (
    CategoricalColumnSchema,
    ColumnDataType,
    IntegerColumnSchema,
)


def test_categorical_column_accepts_integer_categories() -> None:
    schema = CategoricalColumnSchema(
        name="risk_level",
        data_type=ColumnDataType.INTEGER,
        allowed=(1, 2, 3),
    )

    schema.validate_value(2)


def test_categorical_column_rejects_bool_for_integer_category() -> None:
    schema = CategoricalColumnSchema(
        name="risk_level",
        data_type=ColumnDataType.INTEGER,
        allowed=(1, 2, 3),
    )

    with pytest.raises(TypeError, match="expects an integer category"):
        schema.validate_value(True)


def test_categorical_column_accepts_float_categories() -> None:
    schema = CategoricalColumnSchema(
        name="bucket",
        data_type=ColumnDataType.FLOAT,
        allowed=(0.1, 0.2, 1),
    )

    schema.validate_value(0.2)
    schema.validate_value(1)


def test_integer_schema_keeps_numeric_constraints_separate_from_category() -> None:
    schema = IntegerColumnSchema(name="count", min_value=1, positive_only=True)

    with pytest.raises(ValueError, match="below the minimum value"):
        schema.validate_value(0)
