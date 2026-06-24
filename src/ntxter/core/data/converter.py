from abc import ABC, abstractmethod

from ntxter.core.data.types import TidyDataFrameRetriever

class TidyTable[T](ABC):
    df_rtv: TidyDataFrameRetriever

    @abstractmethod
    def compose(self, *args, **kwargs) -> T:
        raise NotImplementedError("Method `compose` must be implemented")
    
    def __getattr__(self, name):
        if hasattr(self.df_rtv, name):
            return getattr(self.df_rtv, name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute `{name}` associated to the composition into DataFrameRetriever") 