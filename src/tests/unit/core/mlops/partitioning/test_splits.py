import pytest


import numpy as np
import pandas as pd

from sklearn.metrics import recall_score


from ntxter.core.mlops.partitioning import BaseQuantileStratifiedKFold

N_SPLITS = 5
N_ROWS = 200
N_COLS = 4

def gen_data():
    """
    Returns
    -------
    y: y[mask] enhanced outliers for testing purposes
    """
    y = np.random.normal(0, 1, N_ROWS)
    mask = (np.quantile(y, 0.025) < y) & (y <  np.quantile(y, 0.975))
    y[~mask] = 3 * y[~mask]

    M = np.random.uniform(-5, 5, N_COLS)
    S = np.identity(N_COLS)
    X = np.random.multivariate_normal(M, S, N_ROWS)

    return X, y, mask


@pytest.fixture
def base_qn_skf():
    class Dummy(BaseQuantileStratifiedKFold):
        def _iter_test_masks(self, *args, **kwargs):
            print('Mocked')
    
    instance = Dummy(N_SPLITS)
    
    return instance

#[TEST]
#[PASSED]
# return nsplits
def test_BaseQuantileStratifiedKFold_get_n_splits(base_qn_skf):
    assert base_qn_skf.get_n_splits() == N_SPLITS

#[TEST]
#[PASSED]
#_validations
# - validation expected X, y
# - error validation:
#   - none float data
#   - y is None
#   - X and y different number of rows
# - validation pd.DataFrame(X), pd.DataFrame(y) | pd.Series(y)
def test_BaseQuantileStratifiedKFold__validations(base_qn_skf):
    X, y, mask = gen_data()

    X_test, y_test = base_qn_skf._validations(X, y)
    assert (X_test != X).sum() == 0
    assert (y_test != y).sum() == 0

    with pytest.raises(ValueError):
        base_qn_skf._validations(['hola', 'como', 'te', 'va'], y)
        base_qn_skf._validations(X, ['hola', 'como', 'te', 'va'])
        base_qn_skf._validations(X, None)
        base_qn_skf._validations(X, y[1:])

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    y_s = pd.Series(y)

    X_test, y_test = base_qn_skf._validations(X_df, y_df)
    assert (X_test != X).sum() == 0
    assert (y_test.ravel() != y).sum() == 0

    X_test, y_test = base_qn_skf._validations(X_df, y_s)
    assert (X_test != X).sum() == 0
    assert (y_test != y).sum() == 0

#[TEST]
#[DEV]
# quantiles
# outlier detection rate using recall
def test_BaseQuantileStratifiedKFold_quantile(base_qn_skf):
    X, y, mask = gen_data()
    group, mk_outliers = (
        base_qn_skf.quantiles(y, n_bins=5, clip_outliers='tukey', k_outlier=1.5)
     )
    
    assert recall_score(mask.astype(int),  mk_outliers.astype(int), pos_label=0) > 0.9
    assert recall_score(mask.astype(int),  mk_outliers.astype(int), pos_label=1) > 0.9
    breakpoint()