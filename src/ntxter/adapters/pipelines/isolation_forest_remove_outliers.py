from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from dataclasses import dataclass, asdict
from typing import Optional, Type, Dict, override


from ntxter.core import utils
from ntxter.core.data.types import PipelineProtocol
from ntxter.core.pipelines import BaseRemoveOutliers
from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.data.types import BasePipelineStage

@dataclass
class BasePipelineStage:
    stage: str
    estimator: Type[EstimatorProtocol]
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.estimator(**self.params)


@dataclass
class IFConfig:
    contamination: float = 0.025
    n_estimators: int = 200
    max_samples: str | int = 'auto'
    max_features: float = 0.6
    random_state: Optional[int] = 42

    def __post_init__(self):
        if self.contamination <= 0:
            raise ValueError("contamination must be > 0")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("random_state must be >= 0 or None")

BasePipelineStage[IFConfig]

class IsolationForestRemoveOutliers(BaseRemoveOutliers):
    pipeline: Dict[str, Type[PipelineProtocol]] = SetterAndGetterType(dict)

    def __init__(self, **kwargs) -> None:
        safe_kwargs = utils.safe_init(IFConfig, **kwargs)
        BasePipelineStage
        self.isof_ = utils.safe_init(IFConfig, **kwargs)
        super().__init__()
"""
    def __init__(self, **kwargs) -> None:
        super().__init__(IFConfig, **kwargs)
        self.pipeline_ = Pipeline([
           ('isof', IsolationForest(**asdict(self.cfg_model_))) 
        ])
    
    @override
    def run(self, X, **run_kwargs):
        self.fit(X, **run_kwargs)
        self.outliers_ = self.is_outlier(X)
        
        return self
    
    @override
    def fit(self, X, scaler=None):
        if scaler is not None:
            self.insert_stage(0, scaler)
        
        self.model_ = self.pipeline_.fit(X)
        return self
    
    @override
    def is_outlier(self, X):
        mask = self.model_.predict(X) != 1
        
        return mask
    
    def predict(self, *args, **kwargs):
        pass
"""