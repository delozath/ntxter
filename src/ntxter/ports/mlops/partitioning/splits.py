from abc import ABC, abstractmethod
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np


FoldIndices = tuple[np.ndarray, np.ndarray]


@runtime_checkable
class SplitterProtocol(Protocol):
    """Structural contract for cross-validation splitters."""

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[FoldIndices]: ...


class BaseFoldFactory(ABC):
    """Contract for lazily creating train/test index folds."""

    @abstractmethod
    def create(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[FoldIndices]: ...
