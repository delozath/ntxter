import pytest

import numpy as np
import pandas as pd

from ntxter.adapters.statistics.base import StatisticsNormalityPandasDF


#NOTE: modify constants for matching categories to the lenght of N_ROWS
#in oder to avoid complicated situations with unbalanced categories that is not the
#focus of this test
OFFSET = 4
N_CONT_COLS = 5
N_ROWS = OFFSET * 100


def gen_data():
    """
    Returns
    -------
    """
    M = np.random.uniform(-5, 5, N_CONT_COLS)
    S = np.identity(N_CONT_COLS)
    X = np.random.multivariate_normal(M, S, N_ROWS)


    cat = np.vstack((
        (np.ones(N_ROWS // 2)[:, None] @ np.array([1, 2])[np.newaxis]).flatten('F'),
        (np.ones(N_ROWS // 4)[:, None] @ np.array([1, 2, 3, 4])[np.newaxis]).flatten('F'),
        (np.ones(N_ROWS // 5)[:, None] @ np.array([-2, -1, 0, 1, 2])[np.newaxis]).flatten('F')
    )).T
    
    str_X = X[:, 0:2].astype('str')

    X = pd.DataFrame(X)
    X.columns = [f'numeric_col_{c+1}' for c in range(N_CONT_COLS)]

    X_cat = pd.DataFrame(cat)
    X_cat.columns = [f'cat_col_{c+1}' for c in range(cat.shape[1])]

    X_str = pd.DataFrame(str_X)
    X_str.columns = [f'str_col_{c+1}' for c in range(str_X.shape[1])]

    df = pd.concat([X, X_cat, X_str], axis=1)
    return df

@pytest.fixture
def normality_pd_df():
    df = gen_data()
    instance = StatisticsNormalityPandasDF(df)
    return instance


def test_statistics_mean_pandas_df(normality_pd_df):
    df = normality_pd_df.container.data

    number_cols = normality_pd_df.get_numerical_cols(df)
    assert any(['str' in i for i in number_cols]) == False

    normality_pd_df.compute()

    breakpoint()

