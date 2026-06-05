from abc import ABC, abstractmethod
from typing import Type, Mapping

from types import MappingProxyType


class Orchestrator(ABC):
    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError("Subclasses must implement the 'execute' method.")


class Factory[T](ABC):
    @abstractmethod
    def create(self) -> T:
        raise NotImplementedError("Subclasses must implement the 'create' method.")


class FactoryRegistry[T](ABC):
    @abstractmethod
    def register(self, name: str, cls: Type[T]) -> None:
        raise NotImplementedError("Subclasses must implement the 'register' method.")
    
    @abstractmethod
    def create(self, name: str, /, **kwargs) -> T:
        raise NotImplementedError("Subclasses must implement the 'create' method.")
    
    @abstractmethod
    def _get_registry(self) -> Mapping[str, Type[T]]:
        raise NotImplementedError("Subclasses must implement the '_get_registry' method.")


class RegistryDescriptor:
    def __get__(
        self,
        instance: FactoryRegistry,
        owner,
    ) -> Mapping[str, Type[Orchestrator]]:
        if instance is None:
            return MappingProxyType({})
        return MappingProxyType(instance._get_registry())

    def __set__(self, instance: FactoryRegistry, value) -> None:
        raise AttributeError("registry is read-only.")

    def __delete__(self, instance: FactoryRegistry) -> None:
        raise AttributeError("registry is read-only.")