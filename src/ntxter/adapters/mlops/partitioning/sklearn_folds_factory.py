from typing import Any, Iterator

from ntxter.ports.mlops.partitioning import (
    BaseFoldFactory,
    FoldIndices,
    SplitterProtocol,
)


class SklearnFoldsFactory(BaseFoldFactory):
    """Lazily expose folds from any scikit-learn compatible splitter."""

    def __init__(self, splitter: SplitterProtocol) -> None:
        if not isinstance(splitter, SplitterProtocol):
            raise TypeError("`splitter` must implement split(X, y=None, groups=None)")
        self._splitter = splitter

    def create(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[FoldIndices]:
        yield from self._splitter.split(X, y, groups)
