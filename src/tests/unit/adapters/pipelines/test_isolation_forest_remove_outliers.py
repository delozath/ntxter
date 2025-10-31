import pytest

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score


from ntxter.adapters.pipelines import IsolationForestRemoveOutliers


def compare_arrays(a0, a1):
    return (a0 != a1).sum()

def gen_data():
    mp = np.array([10, 10, 10])
    m0 = np.array([0, 0, 0])
    mn = np.array([-10, -10, -10])
    s = 0.1 * np.identity(m0.shape[0])
    X = np.concatenate((
        np.random.multivariate_normal(mp, s, 4),
        np.random.multivariate_normal(m0, s, 30),
        np.random.multivariate_normal(mn, s, 5), 
    ))

    outliers = np.concatenate((
        np.ones(4),
        np.zeros(30),
        np.ones(5)
     ))
    
    return X, outliers


def test_IsolationForestRemoveOutliers_remove_outliers():
    X, outliers = gen_data()
    
    instance = IsolationForestRemoveOutliers(
        n_estimators=200, 
        contamination=0.25, 
        random_state=None, 
        param_test='must be ignored'
     )
    instance.run(X, scaler = ('std_scaler', StandardScaler()))
    
    assert recall_score(outliers, instance.outliers_.astype(int), pos_label=0) > 0.9
    assert recall_score(outliers, instance.outliers_.astype(int), pos_label=1) > 0.9

def test_IsolationForestRemoveOutliers_np():
    X, outliers = gen_data()
    
    instance = IsolationForestRemoveOutliers(
        n_estimators=200, 
        contamination=0.25, 
        random_state=None, 
     )
    
    instance.run(X, scaler = ('std_scaler', StandardScaler()))
    assert compare_arrays(
        pd.DataFrame(X)[instance.outliers_].values,
        X[instance.outliers_]
     ) == 0

def test_IsolationForestRemoveOutliers_df():
    X, outliers = gen_data()
    df = pd.DataFrame(X)

    instance = IsolationForestRemoveOutliers(
        n_estimators=200, 
        contamination=0.25, 
        random_state=None, 
     )
    
    instance.run(X, scaler = ('std_scaler', StandardScaler()))
    assert compare_arrays(
        df[instance.outliers_].values,
        X[instance.outliers_]
     ) == 0