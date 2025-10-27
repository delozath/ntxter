import pytest

import numpy as np
import pandas as pd

from pathlib import Path

from ntxter.core.data.loaders import DataLoader

data = pd.DataFrame(
    np.arange(1, 201).reshape(-1, 25).T
)

test_path = Path('.').absolute()
test_file = 'erase_tmp.CSV'

@pytest.fixture
def data_loader():
    class _Dummy(DataLoader):
        def load(self, *args, **kwargs):
            return data
    
    (test_path/test_file).touch()
    return _Dummy(test_path/test_file)

def test__file_check_valid(data_loader):
    assert data_loader._file_check(data_loader.pfname) == data_loader.pfname


def test__file_check_notvalid(data_loader):
    with pytest.raises(FileNotFoundError):    
        test_e_path = data_loader.pfname / 'kagoakfa74adg4s6s3dhg51a'
        data_loader._file_check(test_e_path)


def test__get_extension(data_loader):
    assert data_loader.ext == 'csv'

    fake_fname = Path('tmp/test2.PArqueT')
    assert data_loader._get_extension(fake_fname) == 'parquet'


(test_path/test_file).unlink()