from abc import ABC, abstractmethod

class BaseCallableClass(ABC):
    @abstractmethod
    def __call__(self, *args, **kargs):
        raise NotImplementedError(f"__call__ must be implemented")