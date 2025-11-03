from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Self


import numpy as np


from ntxter.core.pipelines import BasePipeline
from ntxter.core import utils


class BaseRemoveOutliers(ABC, BasePipeline):   
    def __init__(self, cfg_dataclass, **kwargs) -> None:
        self.cfg_model_, self._params = utils.split_dataclass_kwargs(cfg_dataclass, **kwargs)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Self:
        ...
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        ...

    @abstractmethod
    def is_outlier(self, *args, **kwargs) -> np.ndarray:
        ...