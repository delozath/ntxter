from abc import ABC, abstractmethod

class BaseCallableClass(ABC):
    @abstractmethod
    def __call__(self, *args, **kargs):
        raise NotImplementedError(f"__call__ must be implemented")


class BaseRunnableClass(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError("run must be implemented")
