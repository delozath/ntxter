from dataclasses import dataclass
from typing import Optional, Self, Type, Dict, override


import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


from ntxter.core import utils
from ntxter.core.pipelines import BaseRemoveOutliers
from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.data.types import BasePipelineStage, EstimatorProtocol


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


class IsolationForestRemoveOutliers(BaseRemoveOutliers):
    pipeline: Pipeline = SetterAndGetterType(Pipeline)

    def __init__(self, **kwargs) -> None:
        cls_kwargs, extra_kwargs = utils.safe_kwargs(IFConfig, **kwargs)
        isof = BasePipelineStage('Isolation Forest', IsolationForest, IFConfig(**cls_kwargs))
        self.pipeline = Pipeline([
            ('isof', isof.estimator)
        ])
    
    @override
    def predict(self, X: np.ndarray, scaler: EstimatorProtocol | None = None) -> np.ndarray:
        """
        Run the outlier removal process on the data X.
        Parameters
        ----------
        X : np.ndarray
            The input data to process.
        run_kwargs : Dict, optional
            Additional keyword arguments for the run method, by default {}
        
        Returns
        -------
        Self
            The instance of IsolationForestRemoveOutliers after processing.
        
        Notes
        -----
        This method always fits the model to the data X and then identifies outliers.
        """
        self.fit(X, scaler)
        outliers_ = self.is_outlier(X)
        
        return outliers_

    @override
    def fit(self, X, scaler=None, scaler_kwargs={}) -> Self:
        """
        Fit the Isolation Forest model to the data X.
        
        Parameters
        ----------
        X : np.ndarray
            The input data to fit the model.
        scaler : Type[EstimatorProtocol], optional
            An optional scaler to preprocess the data before fitting, by default None
        scaler_kwargs : Dict, optional
            The keyword arguments to initialize the scaler, by default {}
        
        Returns
        -------
        Self
            The fitted instance of IsolationForestRemoveOutliers.
        """
        if scaler is not None:
            self.add_scale_stage('scaler', scaler)

        self.model_ = self._pipeline.fit(X)
        return self
    
    @override
    def is_outlier(self, X):
        """
        Identify outliers in the data X using the fitted Isolation Forest model. The Isolation Forest model predicts -1 for outliers and 1 for inliers. This method converts these predictions into a boolean mask.

        Parameters
        ----------
        X : np.ndarray
            The input data to identify outliers.

        Returns
        -------
        np.ndarray
            A boolean mask indicating which samples are outliers. True for outliers, False for inliers.
        
        Notes
        -----
        Isolation Forest model's from sklearn is used here.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        """
        mask = self.model_.predict(X) != 1
        
        return mask

    def add_stage_at(
            self,
            name: str,
            index: int,
            estimator: EstimatorProtocol,
        ) -> None:
        """
        Insert a new stage into the pipeline at the specified index.

        Parameters
        ----------
        name : str
            The name of the new stage.
        index : int
            The index at which to insert the new stage.
        estimator : Type[EstimatorProtocol]
            The estimator class (Protocol) for the new stage. Defined in ntxter.core.data.types
        estimator_kwargs : Dict, optional
            The keyword arguments to initialize the estimator, by default {}
        
        Returns
        -------
        None

        Notes
        -----
        self._pipeline.steps is a list of tuples (name, estimator) is based on Pipeline from sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        """
        stage = BasePipelineStage(name, estimator)
        self._pipeline.steps.insert(index, (name, stage.estimator))

    def add_scale_stage(
            self,
            name: str,
            estimator: EstimatorProtocol,
        ) -> None:
        """
        Insert a new scaling stage into the pipeline at the beginning.
        This is useful for ensuring that the data is scaled before any other processing steps.

        Parameters
        ----------
        name : str
            The name of the new scaling stage.
        estimator : Type[EstimatorProtocol]
            The estimator class (Protocol) for the scaling stage. Defined in ntxter.core.data.types
        estimator_kwargs : Dict, optional
            The keyword arguments to initialize the estimator, by default {}
        
        Returns
        -------
        None
        """
        self.add_stage_at(name, 0, estimator)
