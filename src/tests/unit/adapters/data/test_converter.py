import pytest

import io
import re

import pandas as pd
from pandas.testing import assert_frame_equal

from ntxter.adapters.data.converters import PandasTidyFrameRetriever, PandasTidyTable

@pytest.fixture
def mock_dataframe() -> pd.DataFrame:
    """Genera un DataFrame fake en memoria que simula datos reales de negocio."""
    csv_data = """
    tienda,region,2024_Q1,2024_Q2,manager,fecha,ventas_totales
    Sucursal_A,Norte,15000,18000,Alex,2024-01-15,33000
    Sucursal_B,Sur,12000,14000,Maria,2024-01-16,26000
    Sucursal_A,Norte,20000,22000,Alex,2024-02-15,42000
    Sucursal_C,Este,9000,11000,Pedro,2024-02-18,20000
    """

    return pd.read_csv(io.StringIO(csv_data.strip()))

def test_dataframe_retriever_init(mock_dataframe):
    rtv = PandasTidyFrameRetriever(
        data=mock_dataframe,
        query="region == 'Norte'",
        cols="2024_Q1",  
        id_vars=["tienda", "manager"]
    )
    
    assert rtv.query == "region == 'Norte'"
    assert isinstance(rtv.cols, list), "`cols` should be a list-type"
    assert rtv.cols == ["2024_Q1"]

    rtv = PandasTidyFrameRetriever(
        data=mock_dataframe,
        query="region == 'Norte'",
        cols=["2024_Q1", "2024_Q2"],  
        id_vars="manager"
    )


def test_dataframe_retriever_args_in_order(mock_dataframe):
    rtv = PandasTidyFrameRetriever(
        mock_dataframe,
        "2024_Q1",  
        ["tienda", "manager"],
        "region == 'Norte'",
    )
    
    assert rtv.query == "region == 'Norte'"
    assert isinstance(rtv.cols, list), "`cols` should be a list-type"
    assert rtv.cols == ["2024_Q1"]

    rtv = PandasTidyFrameRetriever(
        mock_dataframe,
        "2024_Q1",
        ["tienda", "manager"],
    )

    assert rtv.query == ""


def test_dataframe_retriever_args_wrong_order(mock_dataframe):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        PandasTidyFrameRetriever(
            "region == 'Norte'",
            mock_dataframe,
            ["tienda", "manager"],
            "2024_Q1"
        )


def test_dataframe_retriever_normalizes_str_args(mock_dataframe):
    rtv = PandasTidyFrameRetriever(
        data=mock_dataframe,
        query="region == 'Norte'",
        cols="2024_Q1",
        id_vars="tienda",
    )

    assert rtv.cols == ["2024_Q1"]
    assert rtv.id_vars == ["tienda"]


@pytest.mark.parametrize(
    ("invalid_arg", "expected_error"),
    [
        ({"cols": 1}, "cols must be a string or a list of strings."),
        ({"id_vars": 1}, "`id_vars` must be a string or a list of strings."),
        ({"query": ["region == 'Norte'"]}, "query must be a string."),
    ],
)
def test_dataframe_retriever_rejects_invalid_arg_types(
    mock_dataframe,
    invalid_arg,
    expected_error,
):
    args = {
        "data": mock_dataframe,
        "query": "region == 'Norte'",
        "cols": "2024_Q1",
        "id_vars": ["tienda", "manager"],
    }
    args.update(invalid_arg)

    with pytest.raises(TypeError, match=re.escape(expected_error)):
        PandasTidyFrameRetriever(
            **args,
        )

def test_dataframe_retriever_common_columns(mock_dataframe):
    expected_error = "There are common columns between `cols` and `id_vars should be mutually exclusive."
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        PandasTidyFrameRetriever(
            data=mock_dataframe,
            query="region == 'Norte'",
            cols=["2024_Q1", "manager"],  
            id_vars=["tienda", "manager"]
        )

def test_pandas_tidy_table__compose(mock_dataframe):
    expected_error = "Column specified in `cols` and `id_vars` must be valid colunm names in `data`. Missing columns: ['managers']"
    with pytest.raises(KeyError, match=re.escape(expected_error)):
        rtv = PandasTidyTable.compose(
            data=mock_dataframe,
            query="region == 'Norte'",
            cols="2024_Q1",  
            id_vars=["tienda", "managers"]
        )

    expected_error = "Column specified in `cols` and `id_vars` must be valid colunm names in `data`. Missing columns: ['2024_Q_1']"
    with pytest.raises(KeyError, match=re.escape(expected_error)):
        rtv = PandasTidyTable.compose(
            data=mock_dataframe,
            query="region == 'Norte'",
            cols="2024_Q_1",  
            id_vars=["tienda", "manager"]
        )
    
    df = PandasTidyTable.compose(
        data=mock_dataframe,
        query="region == 'Norte'",
        cols="2024_Q1",  
        id_vars=["tienda", "manager"]
    )

    expected =  (
        mock_dataframe.query("region=='Norte'")[['2024_Q1', 'tienda', 'manager']]
                      .melt(id_vars=['tienda', 'manager'])
        )
    
    assert_frame_equal(
        df.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
