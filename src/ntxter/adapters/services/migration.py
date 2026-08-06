from ntxter.ports.factory.base import Orchestrator
from ntxter.ports.factory import migration
#from ntxter.ports.factory.migration import register_migration_service, create_migrator

from ntxter.adapters.data.db_connection import PostgreSQLocalConnection


@migration.register_migration_service("csv_to_local_postgresql")
class OrchestratorCSVToLocalPostgreSQL(Orchestrator):
    def __init__(self):
        self.db_conn = PostgreSQLocalConnection()

    def execute(self, /, **kwargs) -> None:
        self.db_conn.connect(**kwargs)

__all__ = [
    "migration",
    #"register_migration_service",
    #"create_migrator",
    "OrchestratorCSVToLocalPostgreSQL"
]