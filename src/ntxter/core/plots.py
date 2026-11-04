"""Backend-independent plot contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlotSpec:
    """Stable definition shared by every implementation of a plot."""

    name: str
    required_columns: tuple[str, ...] = ()
    title: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PlotSpec.name must be a non-empty string.")
        if (
            not isinstance(self.required_columns, tuple)
            or not all(isinstance(column, str) for column in self.required_columns)
        ):
            raise TypeError("required_columns must be a tuple of strings.")
        if len(self.required_columns) != len(set(self.required_columns)):
            raise ValueError("required_columns must not contain duplicates.")
        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping.")
        self.options = dict(self.options)


@dataclass(slots=True)
class PlotResult[T]:
    """Figure produced by a plot together with portable metadata."""

    spec: PlotSpec
    figure: T
    metadata: dict[str, Any] = field(default_factory=dict)
