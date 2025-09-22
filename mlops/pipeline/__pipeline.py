from abc import ABC, abstractmethod
import trace


from sklearn.pipeline import Pipeline


from ntxter.validation import DuplicateKeyError


class AbstractModelPipeline(ABC):
    def __init__(self):
        self.params: dict = {}
        self.pipeline: Pipeline | None = None
    
    @abstractmethod
    def build(self, *build_args, **build_kwargs) -> Pipeline:
        # self.params and self.pipeline must be set
        ...
    
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

    def register(self, key: str, info: dict):
        def deco(fn):
            if key in self._registry:
                raise DuplicateKeyError(f"Attempt to register the key '{key}' that is already registered model", 100)
            if isinstance(info, dict):
                raise TypeError("info parameter must be dict type")
            reg = info | {'pipeline': fn}
            self._registry[key] = reg
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