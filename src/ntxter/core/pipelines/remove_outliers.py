from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Self


import numpy as np


from ntxter.core.pipelines import BasePipeline


class BaseRemoveOutliers(ABC, BasePipeline):   
    def __init__(self, cfg_dataclass, **kwargs) -> None:
        config_fields = {f.name for f in fields(cfg_dataclass)}
        params = set(kwargs.keys())

        cfg_params = {k: kwargs[k] for k in params.intersection(config_fields)}
        self.cfg_model_ = cfg_dataclass(**cfg_params)

        self._params = {k: kwargs[k] for k in params - config_fields}
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Self:
        ...
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        ...

    @abstractmethod
    def is_outlier(self, *args, **kwargs) -> np.ndarray:
        ...