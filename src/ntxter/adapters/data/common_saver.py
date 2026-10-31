from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Self
import pandas as pd


from ntxter.core.data.savers import DataSaver
from ntxter.core.utils import file_exists


class PandasFrameSafeFactory(DataSaver[pd.DataFrame]):
    def __init__(self, data: pd.DataFrame | dict | list, pth_fname: str, replace: bool):
        super().__init__(pth_fname, replace)
        self.data = data
        self.prepare()
    
    def prepare(self, *args, **kwargs) -> None:
        self._savers = {
            'sav' : self._write_spss,
            'xls' : self._write_excel,
            'xlsx': self._write_excel,
            'csv' : self._write_csv,
            'md': self._write_md,
            'parquet': self._write_parquet,
            'feather': self._write_feather
        }
    
    @classmethod
    def save(cls, data: pd.DataFrame | dict | list, pth_fname: str, replace: bool, *args, **kwargs) -> None:
        instance = cls(data, pth_fname, replace)
        ext = instance.pth_fname.suffix[1:].lower()
        
        func = instance._savers.get(ext, None)
        if func is not None:
            return func(**kwargs)
        else:
            raise NotImplementedError(f"Save manager for `{ext}` is not yet implemented")

    def _write_md(self, **kwargs):
        self.data.to_markdown(
            self.pth_fname,
            index=kwargs.get('index', True),
        )

    def _write_spss(self):
        raise NotImplementedError("SPSS saver is not implemented yet.")

    
    def _write_excel(self, **kwargs):
        index = kwargs.get('index', False)
        
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            sheets = {'Sheet': self.data}
        elif isinstance(self.data, list):
            sheets = {f'Sheet_{i}': df for i, df in enumerate(self.data)}
        elif isinstance(self.data, dict):
            sheets = self.data
        else:
            raise ValueError("Data must be a DataFrame, list of DataFrames, or dict of DataFrames.")
        
        with pd.ExcelWriter(self.pth_fname) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=index)
    
    def _write_csv(self, **kwargs):
        index = kwargs.get('index', True)
        self.data.to_csv(self.pth_fname, index=index)
    
    def _write_parquet(self, /, **kwargs):
        params = {
            'engine': "pyarrow", 
            'compression': "zstd",
            'partition_cols': kwargs.get('partition_cols', None)
        }
        self.data.to_parquet(
            self.pth_fname,
            **params
        )

    def _write_feather(self, **kwargs):
        self.data.to_feather(self.pth_fname)


class PandasFramesSafeFactory:
    """Save a named collection of DataFrames as one workbook or many files.

    A single path must end in ``.xlsx`` and receives one worksheet per mapping
    entry. A mapping of names to paths writes each DataFrame independently via
    :class:`PandasFrameSafeFactory`. In multiple-file mode, destination keys
    must exactly match the data keys.

    All destinations are validated before the first file is written. This
    prevents configuration errors, path collisions, or an existing protected
    file from leaving a partially written collection.
    """

    _SINGLE_EXTENSION = 'xlsx'
    _MULTIPLE_EXTENSIONS = frozenset({
        'csv', 'feather', 'md', 'parquet', 'xlsx',
    })

    @classmethod
    def save(
        cls,
        data: Mapping[str, pd.DataFrame],
        pth_names: str | PathLike[str] | Mapping[str, str | PathLike[str]],
        replace: bool = False,
        *args,
        **kwargs,
    ) -> dict[str, Path]:
        """Persist ``data`` and return the resolved paths that were written."""
        frames = cls._prepare_frames(data)
        multiple = isinstance(pth_names, Mapping)
        destinations = cls._prepare_destinations(frames, pth_names)
        cls._preflight(destinations, replace=replace)

        # Collection entries often use meaningful indexes. Keep them by default
        # in human-readable formats so a round-trip does not discard identity.
        kwargs.setdefault('index', True)
        if multiple:
            for name, path in destinations.items():
                PandasFrameSafeFactory.save(
                    frames[name], path, replace=True, **kwargs
                )
        else:
            PandasFrameSafeFactory.save(
                frames, destinations['all'], replace=True, **kwargs
            )

        return destinations

    @staticmethod
    def _prepare_frames(
        data: Mapping[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        if not isinstance(data, Mapping) or not data:
            raise TypeError('data must be a non-empty mapping of DataFrames.')
        if not all(isinstance(name, str) and name for name in data):
            raise TypeError('DataFrame names must be non-empty strings.')
        invalid = [
            name for name, frame in data.items()
            if not isinstance(frame, pd.DataFrame)
        ]
        if invalid:
            raise TypeError(f'Values for {invalid} are not pandas DataFrames.')
        return dict(data)

    @classmethod
    def _prepare_destinations(
        cls,
        frames: Mapping[str, pd.DataFrame],
        pth_names: str | PathLike[str] | Mapping[str, str | PathLike[str]],
    ) -> dict[str, Path]:
        if isinstance(pth_names, Mapping):
            missing = set(frames) - set(pth_names)
            extra = set(pth_names) - set(frames)
            if missing or extra:
                raise ValueError(
                    'Destination keys must match data keys exactly; '
                    f'missing={sorted(missing)}, extra={sorted(extra)}.'
                )
            destinations = {
                name: cls._path(path, name=name)
                for name, path in pth_names.items()
            }
            unsupported = {
                name: path.suffix.lower()
                for name, path in destinations.items()
                if path.suffix[1:].lower() not in cls._MULTIPLE_EXTENSIONS
            }
            if unsupported:
                raise ValueError(
                    f'Unsupported multiple-file destinations: {unsupported}.'
                )
        else:
            path = cls._path(pth_names, name='all')
            if path.suffix[1:].lower() != cls._SINGLE_EXTENSION:
                raise ValueError('A single collection file must use .xlsx.')
            cls._validate_sheet_names(frames)
            destinations = {'all': path}

        if len(set(destinations.values())) != len(destinations):
            raise ValueError('Every DataFrame must have a distinct destination path.')
        return destinations

    @staticmethod
    def _path(path: str | PathLike[str], *, name: str) -> Path:
        if not isinstance(path, (str, PathLike)):
            raise TypeError(f'Destination for {name!r} must be path-like.')
        return Path(path).expanduser().resolve()

    @staticmethod
    def _validate_sheet_names(frames: Mapping[str, pd.DataFrame]) -> None:
        invalid_characters = set('[]:*?/\\')
        invalid = [
            name for name in frames
            if len(name) > 31 or invalid_characters.intersection(name)
        ]
        if invalid:
            raise ValueError(f'Invalid Excel worksheet names: {invalid}.')
        folded = [name.casefold() for name in frames]
        if len(set(folded)) != len(folded):
            raise ValueError('Excel worksheet names must be unique ignoring case.')

    @staticmethod
    def _preflight(destinations: Mapping[str, Path], *, replace: bool) -> None:
        mode = 'create' if replace else 'raise'
        for path in destinations.values():
            # This creates only missing parent directories; files are written
            # after every destination has passed the same check.
            file_exists(path, mode=mode)


class StringStreamSafe(DataSaver[str]):
    content: str

    def __init__(self, pth_name: str, replace: bool=False) -> None:
        super().__init__(pth_name, replace)
    
    def prepare(self, data: str, *args, **kwargs) -> Self:
        if not isinstance(data, str):
            raise ValueError("Data must be a string containing markdown content.")
        
        self.content = data
        return self
    
    @classmethod
    def save(cls, data: str, pth_name: str, replace: bool, *args, **kwargs) -> None:
        instance = cls(pth_name, replace)
        instance.prepare(data)

        with open(instance.pth_fname, 'w') as f:
            f.write(instance.content)

import yaml


class YamlStreamSafe(DataSaver[dict]):
    content: dict

    def prepare(self, data: dict, *args, **kwargs) -> Self:
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")

        self.content = data
        return self

    @classmethod
    def save(
        cls,
        data: dict,
        pth_name: str,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> None:
        instance = cls(pth_name, replace)
        instance.prepare(data)

        with open(instance.pth_fname, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                instance.content,
                f,
                allow_unicode=True,
                sort_keys=False,
            )
