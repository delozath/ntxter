from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Callable


import pandas as pd


from ntxter.core.base.descriptors import SetterAndGetterType
from ntxter.core.base.errors import RegistryError


@dataclass
class Metric:
    """
    Dataclass representing a metric.

    Attributes
    ---------
    name (str): 
        The name of the metric.
    function (Callable): 
        The function to compute the metric.
    options (Dict): 
        Additional options for the metric function.
    """
    name: str
    function: Callable
    options: Dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization to validate the metric attributes. In particular, ensure that the function is callable.
        """
        if not callable(self.function):
            raise ValueError("The `function` attribute must be callable.")


@dataclass
class Reporting[T]():
    """
    Generic reporting dataclass for metrics reporting.

    Attributes:
    ---------
    identifier (int | str): 
        Unique identifier for the reporting instance.
    iteration (int): 
        The iteration, fold, bootstrap repetition number.
    performances (Dict[str, T]): 
        A dictionary mapping metric names to their computed values. T, in general should be a numeric type.
    optional (dict): 
        An optional dictionary for additional metadata, such as number of repetitions of a cross-validation, timestamps or configuration details.
    """
    identifier: int | str
    iteration: int
    performances: Dict[str, T]
    optional: dict = field(default_factory=dict)


class BaseMetricsContainter(ABC):
    """
    Abstract base class for a metrics container.
    
    Attributes:
    ---------
    registry (Dict[str, Metric]): 
        A dictionary mapping metric names to Metric instances.
    
    Methods:
    ---------
    register(name: str, /, **kwargs) -> None:
        Registers a new metric with the given name and parameters.
    compute(y_true, y_pred) -> pd.DataFrame:
        Computes all registered metrics given true and predicted values, returning the results as a DataFrame or dict.
    
    Raises:
    -------
    ValueError:
        If a metric with the given name is already registered.

    """
    registry = SetterAndGetterType(dict)

    def __init__(self) -> None:
        self._registry: dict[str, Metric] = {}

    @abstractmethod
    def register(self, name: str, /, **kwargs) -> None:
        if name in self._registry:
            raise RegistryError(f"Metric with name '{name}' is already registered.")
        
    @abstractmethod
    def compute(self, y_true, y_pred) -> Dict  | pd.DataFrame: ...


class BaseReportMetrics[T](ABC):
    registry = SetterAndGetterType(dict)
    """
    Abstract base class for reporting metrics.

    Methods:
    ---------
    add(report: Reporting[T]) -> None:
        Adds a new reporting instance.
    build() -> pd.DataFrame:
        Builds the report from data collected in the `registry` and returns a DataFrame.
    to_reporting(identifier: int | str, /, **kwargs) -> Reporting[T]:
        Converts given parameters into a Reporting instance for a complete compatibility.
    """
    def __init__(self) -> None:
        self._registry: dict[str, Reporting[T]] = {}
    
    @abstractmethod
    def add(self, report: Reporting[T]) -> None:
        if report.identifier in self._registry:
            raise RegistryError(f"Report with identifier `{report.identifier}` is already registered.")
        self._registry[report.identifier] = report

    @abstractmethod
    def build(self) -> pd.DataFrame: ...

    @abstractmethod
    def to_reporting(
        self,
        identifier: int | str, /,
        iteration: int = 0,
        performances: Dict[str, float] | None = None,
        optional: Dict[str, float] | None = None
     ) -> Reporting[T]: ...
    """
    Converts given parameters into a Reporting instance for a complete compatibility.
    
    identifier (int | str): 
        Unique identifier for the reporting instance.
    iteration (int): 
        The iteration, fold, bootstrap repetition number.
    performances (Dict[str, float] | None): 
        A dictionary mapping metric names to their computed values.
    optional (Dict[str, float] | None): 
        An optional dictionary for additional metadata.
    """