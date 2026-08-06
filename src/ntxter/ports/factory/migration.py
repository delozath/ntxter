from typing import Dict, Type

from ntxter.ports.factory.base import FactoryRegistry, Orchestrator, RegistryDescriptor



#_____ Migration services and decorators _____


class _MigrationServiceFactory(FactoryRegistry[Orchestrator]):
    __slots__ = ("__registry",)

    registry = RegistryDescriptor()

    def __init__(self) -> None:
        self.__registry: Dict[str, Type[Orchestrator]] = {}

    def _get_registry(self) -> Dict[str, Type[Orchestrator]]:
        return self.__registry

    def register(self, name: str, cls: Type[Orchestrator]) -> None:
        if not issubclass(cls, Orchestrator):
            raise TypeError(f"{cls.__name__} must be a subclass of Orchestrator.")
        if name in self.__registry:
            raise ValueError(f"Migration {name} already registered.")
        self.__registry[name] = cls

    def create(self, name: str, /, **kwargs):
        migrator_class = self.registry.get(name)
        if migrator_class is None:
            raise KeyError(f"Migration {name} not found.")
        return migrator_class(**kwargs)


_MIGRATION_SERVICE_REGISTRY = _MigrationServiceFactory()

def register_migration_service(migration_name: str):
    def decorator(cls: Type[Orchestrator]):
        _MIGRATION_SERVICE_REGISTRY.register(migration_name, cls)
        return cls
    return decorator

def create_migrator(name: str, /, **kwargs):
    return _MIGRATION_SERVICE_REGISTRY.create(name, **kwargs)
