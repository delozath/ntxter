import numpy as np
import pytest

from sklearn.model_selection import StratifiedKFold

from ntxter.adapters.mlops.partitioning import SklearnFoldsFactory


class SplitterSpy:
    def __init__(self):
        self.calls = []

    def split(self, X, y=None, groups=None):
        self.calls.append((X, y, groups))
        yield np.array([0, 2]), np.array([1, 3])


def test_SklearnFoldsFactory_create_is_lazy_and_forwards_groups():
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])
    splitter = SplitterSpy()

    folds = SklearnFoldsFactory(splitter).create(X, y, groups)

    assert splitter.calls == []
    train_idx, test_idx = next(folds)
    assert len(splitter.calls) == 1
    call_X, call_y, call_groups = splitter.calls[0]
    assert call_X is X
    assert call_y is y
    assert call_groups is groups
    np.testing.assert_array_equal(train_idx, [0, 2])
    np.testing.assert_array_equal(test_idx, [1, 3])


def test_SklearnFoldsFactory_matches_sklearn_splitter_output():
    X = np.arange(24).reshape(12, 2)
    y = np.array([0, 1] * 6)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    actual = list(SklearnFoldsFactory(splitter).create(X, y))
    expected = list(splitter.split(X, y))

    for (actual_train, actual_test), (expected_train, expected_test) in zip(actual, expected):
        np.testing.assert_array_equal(actual_train, expected_train)
        np.testing.assert_array_equal(actual_test, expected_test)


def test_SklearnFoldsFactory_requires_a_splitter():
    with pytest.raises(TypeError, match="must implement split"):
        SklearnFoldsFactory(object())
