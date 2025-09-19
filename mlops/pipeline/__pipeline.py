from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

class AbstractModelPipeline(ABC):
    def __init__(self, params):
        self.params = params
        self.pipeline: Pipeline = self._build()
    
    @abstractmethod
    def _build(self, *args, **kwargs) -> Pipeline:
        ...
    
    def fit_predict_bundle(self, bdle):
        self.pipeline.fit(bdle.X_train, bdle.y_train)
        y_est = self.pipeline.predict(bdle.X_test2)
        return y_est
    
    def fit_predict(self, X_train, y_train, X_test):
        self.pipeline.fit(X_train, y_train)
        y_est = self.pipeline.predict(X_test)
        return y_est