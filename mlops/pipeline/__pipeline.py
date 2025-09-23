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

    def register(self, def_pipe, descriptor, model_params, id=None):
            key = (
                id if id else f"{def_pipe.__name__}: {descriptor['outcome']} ~ {descriptor['model']}"
            )
            if key in self._registry:
                raise DuplicateKeyError(f"Attempt to register the key '{key}' that is already registered model", 100)
            inst = def_pipe()
            inst.build(**model_params)
            reg = {'pipe_wrap': inst}
            reg |= descriptor
            self._registry[key] = reg

    def __repr__(self):
        output = f"{type(self).__name__} registered:\n"
        output += '\n'.join(
            [f"{k}, {i}" for k, i in self._registry.items()]
         )
        output += '\n'
        return output