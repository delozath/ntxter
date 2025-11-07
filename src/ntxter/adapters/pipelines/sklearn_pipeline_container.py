from typing import override


import numpy as np

from sklearn.pipeline import Pipeline


from ntxter.core import utils
from ntxter.core.pipelines.pipeline import BasePipelineContainer
from ntxter.core.data.types import PipelineProtocol


class SklearnPipelineContainer(BasePipelineContainer):
    def __init__(self) -> None:
        self.registry: dict[str, Pipeline] = {}
    
    