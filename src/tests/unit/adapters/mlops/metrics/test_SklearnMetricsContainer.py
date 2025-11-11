import pytest
from typing import Tuple


import numpy as np


from ntxter.adapters.mlops.metrics import SklearnMetricsContainer

def gen_data(n_samples: int = 100, prop_error: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    N = n_samples if n_samples % 2 == 0 else n_samples - 1
    y_true = (np.ones((N // 2, 1)) @ [[-1, 1]]).ravel('F')
    
    N_errors = int(prop_error * N)
    error_indices = np.arange(N)
    np.random.shuffle(error_indices)
    error_indices = error_indices[:N_errors]

    y_pred = y_true.copy()
    y_pred[error_indices] *= -1

    return y_true, y_pred


@pytest.fixture
def metrics_container() -> SklearnMetricsContainer:
    return SklearnMetricsContainer()

def test_register_metric_metrics_added(metrics_container: SklearnMetricsContainer):
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    metrics_container.register(
        "accuracy",
        options={},
        function=accuracy_score
    )

    metrics_container.register(
        "recall",
        options={},
        function=recall_score
    )
    metrics_container.register(
        "precision",
        options={},
        function=precision_score
    )

    assert "accuracy" in metrics_container._registry
    assert "recall" in metrics_container._registry
    assert "precision" in metrics_container._registry
    
    assert len(metrics_container._registry) == 3


def test_register_metric_performances(metrics_container: SklearnMetricsContainer):
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    prop_error = 0.2
    y_true, y_pred = gen_data(100, prop_error)

    metrics_container.register(
        "accuracy",
        options={},
        function=accuracy_score
    )

    metrics_container.register(
        "recall",
        options={},
        function=recall_score
    )
    metrics_container.register(
        "precision",
        options={},
        function=precision_score
    )

    res = metrics_container.compute(y_true, y_pred)
    assert res.at[0, "accuracy"] == 1 - prop_error
    #res.at[0, "recall"] = 0.8
    #res.at[0, "precision"] = 0.7