import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ntxter.adapters.data.common_saver import PandasFramesSafeFactory


@pytest.fixture
def frames():
    return {
        'variables': pd.DataFrame(
            {'kind': ['numeric', 'categorical']},
            index=pd.Index(['age', 'group'], name='column'),
        ),
        'numeric': pd.DataFrame(
            {'mean': [30.0], 'median': [30.0]},
            index=pd.Index(['age'], name='column'),
        ),
        'categories': pd.DataFrame({
            'column': ['group', 'group'],
            'category': ['A', 'B'],
            'count': [2, 1],
        }),
        'information': pd.DataFrame(
            {'entropy': [0.64], 'dominant_ratio': [2 / 3]},
            index=pd.Index(['group'], name='column'),
        ),
    }


def test_collection_is_saved_in_one_workbook(tmp_path, frames):
    path = tmp_path / 'description.xlsx'

    written = PandasFramesSafeFactory.save(frames, path)

    assert written == {'all': path.resolve()}
    loaded = pd.read_excel(path, sheet_name=None, index_col=0)
    assert list(loaded) == list(frames)
    for name, expected in frames.items():
        assert_frame_equal(loaded[name], expected, check_dtype=False)


def test_collection_is_saved_to_exact_multiple_destinations(tmp_path, frames):
    paths = {
        name: tmp_path / f'{name}.csv'
        for name in frames
    }

    written = PandasFramesSafeFactory.save(frames, paths)

    assert written == {name: path.resolve() for name, path in paths.items()}
    for name, expected in frames.items():
        loaded = pd.read_csv(paths[name], index_col=0)
        assert_frame_equal(loaded, expected, check_dtype=False)


def test_multiple_destinations_must_match_frame_names(tmp_path, frames):
    paths = {'variables': tmp_path / 'variables.csv'}

    with pytest.raises(ValueError, match='missing=.*categories'):
        PandasFramesSafeFactory.save(frames, paths)

    assert not paths['variables'].exists()


def test_all_existing_files_are_detected_before_writing(tmp_path, frames):
    paths = {
        name: tmp_path / f'{name}.csv'
        for name in frames
    }
    paths['categories'].touch()

    with pytest.raises(FileExistsError):
        PandasFramesSafeFactory.save(frames, paths)

    assert not paths['variables'].exists()
    assert paths['categories'].read_bytes() == b''


def test_single_collection_requires_xlsx(tmp_path, frames):
    with pytest.raises(ValueError, match='must use .xlsx'):
        PandasFramesSafeFactory.save(frames, tmp_path / 'description.csv')
