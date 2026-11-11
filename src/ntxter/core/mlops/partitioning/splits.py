"""Backward-compatible exports for partitioning contracts."""

from ntxter.ports.mlops.partitioning.splits import (
    BaseFoldFactory,
    FoldIndices,
    SplitterProtocol,
)

__all__ = [
    "BaseFoldFactory",
    "FoldIndices",
    "SplitterProtocol",
]
