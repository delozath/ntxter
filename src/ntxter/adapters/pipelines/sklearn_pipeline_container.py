from typing import override


import numpy as np

from sklearn.pipeline import Pipeline


from ntxter.core.pipelines.pipeline import BasePipelineContiner


class SklearnPipelineContainer(BasePipelineContiner):
    def __init__(self) -> None:
        self.pipelines = dict()

    @override
    def _register(self, name: str, pipeline: Pipeline) -> None:
        self._pipelines[name] = pipeline

    @override
    def fit(self, X, y):
        for name, pipeline in self._pipelines.items():
            pipeline.fit(X, y)

            yield name, pipeline
    
    @override
    def predict(self, X):
        for name, pipeline in self._pipelines.items():
            y = pipeline.predict(X)

            yield name, y