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

def test_register_metric(metrics_container: SklearnMetricsContainer):
    from sklearn.metrics import accuracy_score

    y_true, y_pred = gen_data(100)

    metrics_container.register(
        "accuracy",
        options={},
        function=accuracy_score
    )
    assert "accuracy" in metrics_container._registry