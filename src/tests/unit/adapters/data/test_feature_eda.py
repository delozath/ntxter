import pandas as pd
import pytest

from ntxter.adapters.data.feature_eda import PandasRedundancySelector
from ntxter.adapters.pipelines.feature_exploration import (
    PandasFeatureExplorationPipeline,
)


def _features_with_outcome() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric_a": range(30),
            "numeric_b": [value * 2 for value in range(30)],
            "categorical_a": pd.Series([0, 1, 2] * 10, dtype="category"),
            "categorical_b": pd.Series(["a", "b", "c"] * 10, dtype="category"),
            "outcome": [0, 1] * 15,
        }
    )


def test_redundancy_selector_runs_three_feature_views_without_outcome():
    result = PandasRedundancySelector().run(
        _features_with_outcome(),
        outcomes="outcome",
    )

    assert result.feature_columns == (
        "numeric_a",
        "numeric_b",
        "categorical_a",
        "categorical_b",
    )
    assert result.excluded_columns == ("outcome",)
    assert result.numeric.association_matrix.columns.tolist() == [
        "numeric_a",
        "numeric_b",
    ]
    assert result.categorical.association_matrix.columns.tolist() == [
        "categorical_a",
        "categorical_b",
    ]
    assert set(result.mixed.association_matrix.columns) == set(result.feature_columns)
    for analysis in result.analyses.values():
        assert "outcome" not in analysis.association_matrix.columns
        assert "outcome" not in analysis.selected_columns
        assert "outcome" not in analysis.redundant_columns

    # Preserve the public contract consumed by existing experiment workflows.
    assert result.report is result.mixed.report
    assert result.association_matrix is result.mixed.association_matrix
    assert result.selected_columns == result.mixed.selected_columns
    assert result.redundant_columns == result.mixed.redundant_columns


def test_redundancy_selector_handles_missing_views_and_single_columns():
    result = PandasRedundancySelector().run(
        pd.DataFrame({"numeric": [1, 2, 3], "outcome": [0, 1, 0]}),
        outcomes=["outcome"],
    )

    assert result.numeric.selected_columns == ("numeric",)
    assert result.numeric.redundant_columns == ()
    assert result.numeric.association_matrix.shape == (1, 1)
    assert result.categorical.selected_columns == ()
    assert result.categorical.association_matrix.empty
    assert result.mixed.selected_columns == ()
    assert result.mixed.association_matrix.empty


def test_declared_numeric_categories_are_not_analyzed_as_continuous():
    df = pd.DataFrame(
        {
            "continuous_a": [1.0, 2.0, 3.0, 4.0],
            "continuous_b": [2.0, 4.0, 6.0, 8.0],
            "category_a": [0, 1, 0, 1],
            "category_b": [1, 2, 1, 2],
        }
    )

    result = PandasRedundancySelector().run(
        df,
        categorical_columns=["category_a", "category_b"],
    )

    assert set(result.numeric.association_matrix.columns) == {
        "continuous_a",
        "continuous_b",
    }
    assert set(result.categorical.association_matrix.columns) == {
        "category_a",
        "category_b",
    }
    assert set(result.mixed.association_matrix.columns) == set(df.columns)


def test_redundancy_selector_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        PandasRedundancySelector().run(
            pd.DataFrame({"numeric": [1, 2, 3]}),
            threshold=1.1,
        )


def test_feature_exploration_pipeline_builds_one_plot_per_nonempty_analysis():
    result = PandasFeatureExplorationPipeline().run(
        _features_with_outcome(),
        outcomes="outcome",
    )

    assert set(result.plots) == {"numeric", "categorical", "mixed"}
    assert all(plot.figure is not None for plot in result.plots.values())

    for plot in result.plots.values():
        plot.figure.clf()
