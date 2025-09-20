from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

class AbstractModelPipeline(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.pipeline: Pipeline | None = None
    
    @abstractmethod
    def _build(self, **kwargs) -> Pipeline:
        ...

    def compose(self, **kwargs):
        self.pipeline = self._build(**kwargs)
        return self
    
    def fit_predict_bundle(self, bdle):
        #NOTE: implement if self.pipeline is None -> self.compose()?
        self.pipeline.fit(bdle.X_train, bdle.y_train)
        y_est = self.pipeline.predict(bdle.X_test)
        return y_est
    
    def fit_predict(self, X_train, y_train, X_test):
        self.pipeline.fit(X_train, y_train)
        y_est = self.pipeline.predict(X_test)
        return y_est

class PipelineFactory:
    def __init__(self) -> None:
        self._registry = {}

    def register(self, key):
        def deco(fn):
            if key in self._registry:
                raise KeyError(f"Builder name already registered '{key}'.")
            self._registry[key] = fn
            return fn
        return deco
    
    def build(self, key: str, *args, **kwargs):
        model = self._registry[key](*args, **kwargs)
        return model
    
    def __repr__(self):
        output = f"{type(self).__name__} registered:\n"
        output += '\n'.join(
            [f"{k}, {i.__name__}" for k, i in self._registry.items()]
         )
        output += '\n'
        return output