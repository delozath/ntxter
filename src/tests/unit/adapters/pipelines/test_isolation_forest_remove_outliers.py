import pytest

import numpy as np

from sklearn.preprocessing import StandardScaler


from ntxter.adapters.pipelines import IsolationForestRemoveOutliers


def test_IsolationForestRemoveOutliers_init():
    X = np.arange(400).reshape(100, -1, order='F')
    instance = IsolationForestRemoveOutliers(n_estimators=500, random_state=0, param_test='must be ignored')
    instance.run(X, scaler = ('std_scaler', StandardScaler()))
    raise NotImplementedError("A known dataset is needed for testing")
    breakpoint()