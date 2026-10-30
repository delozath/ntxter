"""Descriptive statistics checked against analytical and independent results."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import entropy

from ntxter.adapters.data.eda import (
    ColumnContract,
    DataContract,
    DataDescriber,
    DataValidator,
)


def test_balanced_categories_use_observed_denominator_and_scipy_entropy():
    df = pd.DataFrame({"group": ["A", "B", None, None]})

    result = DataDescriber().run(df)

    variable = result["variables"].loc["group"]
    assert variable["kind"] == "categorical"
    assert variable["n_total"] == 4
    assert variable["n_observed"] == 2
    assert variable["n_missing"] == 2
    assert variable["missing_ratio"] == pytest.approx(0.5)
    assert variable["n_distinct"] == 2

    counts = result["categories"].set_index("category")
    assert counts["count"].to_dict() == {"A": 1, "B": 1}
    assert counts["proportion_observed"].to_dict() == {"A": 0.5, "B": 0.5}
    assert counts["proportion_total"].to_dict() == {"A": 0.25, "B": 0.25}
    info = result["information"].loc["group"]
    assert info["entropy"] == pytest.approx(entropy([1, 1]))
    assert info["entropy"] == pytest.approx(math.log(2))
    assert info["normalized_entropy"] == pytest.approx(1)
    assert info["gini"] == pytest.approx(0.5)
    assert info["kl_uniform"] == pytest.approx(0)
    assert info["dominant_ratio"] == pytest.approx(0.5)


def test_declared_categories_keep_zero_counts_and_define_uniform_domain():
    df = pd.DataFrame({
        "group": pd.Categorical(["A", "A", None], categories=["A", "B"]),
    })

    result = DataDescriber().run(df)

    counts = result["categories"].set_index("category")
    assert counts["count"].to_dict() == {"A": 2, "B": 0}
    assert counts.loc["B", "proportion_observed"] == 0
    assert counts.loc["B", "proportion_total"] == 0
    info = result["information"].loc["group"]
    assert info["n_categories"] == 2
    assert info["entropy"] == pytest.approx(0)
    assert info["normalized_entropy"] == pytest.approx(0)
    assert info["gini"] == pytest.approx(0)
    assert info["kl_uniform"] == pytest.approx(entropy([1, 0], [0.5, 0.5]))
    assert "constant" in info["signals"]


def test_numeric_descriptives_match_known_values():
    df = pd.DataFrame({"value": pd.Series([1, 2, 3, 4, pd.NA], dtype="Float64")})

    result = DataDescriber().run(df)

    stats = result["numeric"].loc["value"]
    assert stats["n_finite"] == 4
    assert stats["n_nonfinite"] == 0
    for metric, expected in {
        "mean": 2.5, "median": 2.5, "std": math.sqrt(5 / 3), "var": 5 / 3,
        "min": 1, "q1": 1.75, "q3": 3.25, "max": 4, "iqr": 1.5,
    }.items():
        assert stats[metric] == pytest.approx(expected), metric
    info = result["information"].loc["value"]
    for metric in (
        "n_categories", "entropy", "normalized_entropy", "gini", "kl_uniform",
        "dominant_ratio",
    ):
        assert pd.isna(info[metric]), metric


def test_nonfinite_values_are_counted_separately_from_missing_and_finite():
    df = pd.DataFrame({"value": [1.0, 2.0, np.inf, -np.inf, np.nan]})

    result = DataDescriber().run(df)

    variable = result["variables"].loc["value"]
    assert variable["n_observed"] == 4
    assert variable["n_missing"] == 1
    stats = result["numeric"].loc["value"]
    assert stats["n_finite"] == 2
    assert stats["n_nonfinite"] == 2
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["std"] == pytest.approx(math.sqrt(0.5))
    assert stats["min"] == 1
    assert stats["max"] == 2
    assert "non_finite_values" in result["information"].loc["value", "signals"]


def test_dtypes_define_roles_without_a_cardinality_heuristic():
    df = pd.DataFrame({
        "integer": pd.Series([1, 2, pd.NA], dtype="Int64"),
        "float": pd.Series([1, 2, pd.NA], dtype="Float64"),
        "boolean": pd.Series([True, True, pd.NA], dtype="boolean"),
        "category": pd.Categorical([1, 2, None], categories=[1, 2, 3]),
        "text": pd.Series(["A", "B", pd.NA], dtype="string"),
        "object": pd.Series(["A", "B", None], dtype=object),
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", None]),
        "complex": [1 + 2j, 2 + 3j, complex(np.nan, 0)],
    })

    result = DataDescriber().run(df)

    assert result["variables"]["kind"].to_dict() == {
        "integer": "numeric", "float": "numeric", "boolean": "categorical",
        "category": "categorical", "text": "categorical", "object": "categorical",
        "date": "other", "complex": "other",
    }
    boolean = result["categories"].query("column == 'boolean'").set_index("category")
    assert boolean["count"].to_dict() == {False: 0, True: 2}
    assert set(result["numeric"].index) == {"integer", "float"}


def test_explicit_columns_exclude_identifiers_and_override_numeric_codes():
    df = pd.DataFrame({"id": [101, 102, 103], "code": [0, 1, 1], "age": [20, 30, 40]})
    original = df.copy(deep=True)

    result = DataDescriber().run(df, columns=["code", "age"], categorical=["code"])

    assert result["variables"].index.tolist() == ["code", "age"]
    assert result["variables"].loc["code", "kind"] == "categorical"
    assert result["numeric"].index.tolist() == ["age"]
    assert set(result["categories"]["column"]) == {"code"}
    assert result["categories"].set_index("category")["count"].to_dict() == {0: 1, 1: 2}
    assert_frame_equal(df, original)


def test_single_observation_has_undefined_sample_variance():
    result = DataDescriber().run(pd.DataFrame({"value": [7.0]}))

    stats = result["numeric"].loc["value"]
    assert stats["mean"] == 7
    assert stats["median"] == 7
    assert stats["iqr"] == 0
    assert pd.isna(stats["std"])
    assert pd.isna(stats["var"])
    signals = result["information"].loc["value", "signals"]
    assert "constant" in signals
    assert "insufficient_observations" in signals


@pytest.mark.parametrize("n_rows", [0, 3])
def test_empty_and_all_missing_columns_keep_undefined_metrics(n_rows):
    df = pd.DataFrame({
        "group": pd.Categorical([None] * n_rows, categories=["A", "B"]),
        "value": pd.Series([pd.NA] * n_rows, dtype="Float64"),
    })

    result = DataDescriber().run(df)

    assert set(result) == {"variables", "numeric", "categories", "information"}
    assert result["variables"]["n_observed"].eq(0).all()
    if n_rows:
        assert result["variables"]["missing_ratio"].eq(1).all()
    else:
        assert result["variables"]["missing_ratio"].isna().all()
    counts = result["categories"]
    assert counts["count"].eq(0).all()
    assert counts["proportion_observed"].isna().all()
    if n_rows:
        assert counts["proportion_total"].eq(0).all()
    else:
        assert counts["proportion_total"].isna().all()
    info = result["information"].loc["group"]
    assert info["n_categories"] == 2
    for metric in ("entropy", "normalized_entropy", "gini", "kl_uniform", "dominant_ratio"):
        assert pd.isna(info[metric]), metric
    for signals in result["information"]["signals"]:
        assert "no_observations" in signals
        assert "constant" not in signals
        assert "insufficient_observations" not in signals
    for metric in ("mean", "median", "std", "var", "min", "q1", "q3", "max", "iqr"):
        assert pd.isna(result["numeric"].loc["value", metric]), metric


def test_zero_column_input_returns_stable_table_shapes():
    result = DataDescriber().run(pd.DataFrame())

    assert set(result) == {"variables", "numeric", "categories", "information"}
    assert all(table.empty for table in result.values())
    assert set(result["categories"].columns) == {
        "column", "category", "count", "proportion_observed", "proportion_total",
    }
    assert {"kind", "n_total", "n_observed", "n_missing", "missing_ratio"}.issubset(
        result["variables"].columns
    )
    assert {"entropy", "normalized_entropy", "signals"}.issubset(
        result["information"].columns
    )


def test_one_category_uses_zero_normalized_entropy():
    df = pd.DataFrame({"group": pd.Categorical(["A", "A"], categories=["A"])})

    info = DataDescriber().run(df)["information"].loc["group"]

    assert info["n_categories"] == 1
    assert info["entropy"] == 0
    assert info["normalized_entropy"] == 0
    assert info["kl_uniform"] == 0
    assert info["dominant_ratio"] == 1


def test_optional_thresholds_report_reasons_without_dropping_variables():
    df = pd.DataFrame({"group": ["A"] * 9 + ["B", None, None]})

    result = DataDescriber().run(
        df, min_observations=11, max_missing_ratio=0.1,
        min_normalized_entropy=0.5, min_category_count=2,
    )

    assert result["variables"].index.tolist() == ["group"]
    assert result["categories"]["count"].sum() == 10
    signals = result["information"].loc["group", "signals"]
    assert isinstance(signals, tuple)
    assert {
        "insufficient_observations", "high_missing_ratio",
        "low_normalized_entropy", "rare_categories",
    }.issubset(signals)
    default_signals = DataDescriber().run(df)["information"].loc["group", "signals"]
    assert "high_missing_ratio" not in default_signals
    assert "low_normalized_entropy" not in default_signals
    assert "rare_categories" not in default_signals


def test_threshold_boundaries_do_not_flag_equal_values_or_unobserved_categories():
    df = pd.DataFrame({
        "group": pd.Categorical(["A", "A", "B", "B", None], categories=["A", "B"]),
        "unused": pd.Categorical(["A"] * 5, categories=["A", "B"]),
    })

    result = DataDescriber().run(
        df, min_observations=4, max_missing_ratio=0.2,
        min_normalized_entropy=1.0, min_category_count=2,
    )

    signals = result["information"].loc["group", "signals"]
    for name in (
        "insufficient_observations", "high_missing_ratio",
        "low_normalized_entropy", "rare_categories",
    ):
        assert name not in signals
    assert "rare_categories" not in result["information"].loc["unused", "signals"]


@pytest.mark.parametrize("n_categories", [3, 5])
def test_uniform_entropy_stays_in_range_without_rounding_threshold_flags(n_categories):
    df = pd.DataFrame({"group": pd.Categorical(range(n_categories))})

    info = DataDescriber().run(
        df, min_normalized_entropy=1,
    )["information"].loc["group"]

    assert 0 <= info["normalized_entropy"] <= 1
    assert info["normalized_entropy"] == pytest.approx(1)
    assert "low_normalized_entropy" not in info["signals"]


def test_adding_missing_values_preserves_conditional_information_and_descriptives():
    df = pd.DataFrame({
        "group": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    extended = pd.DataFrame({
        "group": pd.Categorical(
            ["A", "A", "B", "B", None, None], categories=["A", "B", "C"],
        ),
        "value": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
    })

    initial = DataDescriber().run(df)
    with_missing = DataDescriber().run(extended)

    assert_frame_equal(initial["numeric"], with_missing["numeric"])
    assert_frame_equal(initial["information"], with_missing["information"])
    assert_frame_equal(
        initial["categories"].drop(columns="proportion_total"),
        with_missing["categories"].drop(columns="proportion_total"),
    )
    assert with_missing["categories"]["proportion_total"].sum() == pytest.approx(4 / 6)


def test_row_permutation_and_duplicate_indices_do_not_change_statistics():
    df = pd.DataFrame({
        "group": ["B", "A", "B", "C", None],
        "value": [2.0, 1.0, 3.0, 4.0, np.nan],
    }, index=[8, 8, 2, 2, 8])
    shuffled = df.iloc[[3, 0, 4, 2, 1]]

    initial = DataDescriber().run(df)
    permuted = DataDescriber().run(shuffled)

    for name in ("variables", "numeric", "information"):
        assert_frame_equal(initial[name], permuted[name])
    assert_frame_equal(
        initial["categories"].sort_values(["column", "category"]).reset_index(drop=True),
        permuted["categories"].sort_values(["column", "category"]).reset_index(drop=True),
    )


def test_input_values_index_dtypes_and_ordered_categories_remain_unchanged():
    df = pd.DataFrame({
        "group": pd.Categorical(["B", None, "A"], categories=["B", "A", "C"], ordered=True),
        "value": pd.array([1, pd.NA, 3], dtype="Int64"),
    }, index=pd.Index([7, 7, 2], name="record"))
    original = df.copy(deep=True)

    DataDescriber().run(df)

    assert_frame_equal(df, original)


def test_existing_contract_and_validator_still_work_independently():
    contract = DataContract(columns={
        "value": ColumnContract(dtype="int64", checks={"greater_than_or_equal_to": 0}),
    })
    reviewed = DataContract.from_yaml(contract.to_yaml())
    valid = pd.DataFrame({"value": [1, 2]})
    invalid = pd.DataFrame({"value": [1, -1]}, index=[100, 200])

    assert DataValidator(reviewed).run(valid)["valid"]
    failure = DataValidator(reviewed).run(invalid)

    assert not failure["valid"]
    assert failure["conflicting_columns"] == ["value"]
    assert failure["conflicting_row_positions"] == [1]
    assert_frame_equal(failure["conflicting_rows"], invalid.iloc[[1]])
