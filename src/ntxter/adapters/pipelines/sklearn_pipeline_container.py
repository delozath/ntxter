from typing import override


import numpy as np

from sklearn.pipeline import Pipeline


from ntxter.core import utils
from ntxter.core.pipelines.pipeline import BasePipelineContainer, PipelineStageConfig


class SklearnPipelineContainer(BasePipelineContainer[Pipeline]):
    def __init__(self) -> None:
        super().__init__()
    
    @override
    def build_pipeline_config(self, **kwargs) -> PipelineStageConfig:
        pipeline_cfg, _ = PipelineStageConfig(**kwargs)
        return pipeline_cfg
    """
    @override
    def _register(self, name: str, pipeline: Pipeline) -> None:
        self._pipelines[name] = pipeline

    @override
    def _fit(self, X, y):
        for name, stage in self._registry().items():
            stage.fit(X, y)
            yield name, stage
    
    @override
    def predict(self, X):
        for name, pipeline in self._pipelines.items():
            y = pipeline.predict(X)

            yield name, y
    
    @override
    def __str__(self) -> str:
        return f"SklearnPipelineContainer with {len(self._pipelines)} pipelines."
    """