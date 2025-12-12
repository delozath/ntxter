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

    X_str = pd.DataFrame(str_X + ' datos')
    X_str.columns = [f'str_col_{c+1}' for c in range(str_X.shape[1])]

    X_comp = pd.concat((
        pd.DataFrame(cat[:, 0:1]).replace({1: True, 2: False}).infer_objects(copy=False),
        pd.DataFrame(cat[:, 0:1]).replace({1: True, 2: False}).infer_objects(copy=False),
        pd.DataFrame(cat[:, 0:1]),
        pd.DataFrame(cat[:, 0:1]).astype(str) + ' str testing',
        X[['numeric_col_1']]
     ), axis=1
    )
    X_comp.columns = ['bool_complete', 'bool_missings', 'cat_missings', 'cat_str_missings', 'numeric_missings']
    cols = X_comp.filter(like='missing').columns
    X_comp.loc[X_comp.index[:10], cols] = 'Nulificacion'

    df = pd.concat([X, X_cat, X_str, X_comp], axis=1)
    return df

def df_structure_for_test():
    df = df = pd.DataFrame({
    "dtype": [
        "numerical", "numerical", "numerical", "numerical", "numerical",
        "numerical", "numerical", "numerical",
        "string", "string",
        "bool", "object", "object", "string", "object"
    ],
    "ntype": [
        "continuous", "continuous", "continuous", "continuous", "continuous",
        "discrete", "discrete", "discrete",
        "continuous", "continuous",
        "discrete", "discrete", "discrete", "discrete", "continuous"
    ]
    }, index=[
        "numeric_col_1",
        "numeric_col_2",
        "numeric_col_3",
        "numeric_col_4",
        "numeric_col_5",
        "cat_col_1",
        "cat_col_2",
        "cat_col_3",
        "str_col_1",
        "str_col_2",
        "bool_complete",
        "bool_missings",
        "cat_missings",
        "cat_str_missings",
        "numeric_missings"
    ])
    return df

@pytest.fixture
def normality_pd_df():
    df = gen_data()
    instance = StatisticsNormalityPandasDF(df)
    return instance


def test_statistics_mean_pandas_df(normality_pd_df):
    df = normality_pd_df.data
    
    cols = normality_pd_df.disaggregate_to_df(df)
    assert df_structure_for_test().equals(cols)

    normality_pd_df.compute(grouping='cat_col_1')
    breakpoint()

    normality_pd_df.compute()


