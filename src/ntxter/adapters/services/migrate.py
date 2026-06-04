from abc import ABC, abstractmethod

from ntxter.core.data.database import BaseDatabase
from ntxter.ports.factory import Orchestrator

from ntxter.adapters.data.db_connection import PostgreSQLocalConnection


class MigrationFactory(ABC):
    @abstractmethod
    def create_connection(self) -> BaseDatabase:
        raise NotImplementedError

    @abstractmethod
    def create_orchestrator(self, cfg=None) -> Orchestrator:
        raise NotImplementedError


class OrchestratorMigration(Orchestrator):
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
    
    def execute(self) -> None:
        self.orchestrator.execute()


class OrchestratorCSVToLocalPostgreSQL(Orchestrator):
    def __init__(self, db_conn: BaseDatabase, cfg=None):
        self.db_conn = db_conn
        self.cfg = cfg

    def execute(self) -> None:
        self.db_conn.connect()


class FactoryCSVToLocalPostgreSQL(MigrationFactory):
    def create_connection(self) -> BaseDatabase:
        return PostgreSQLocalConnection()

    def create_orchestrator(self, cfg=None) -> Orchestrator:
        return OrchestratorCSVToLocalPostgreSQL(
            db_conn=self.create_connection(),
            cfg=cfg,
        )


class CSVToLocalPostgreSQL:
    def __init__(self, cfg, factory: MigrationFactory | None = None):
        self.cfg = cfg
        self.factory = factory or FactoryCSVToLocalPostgreSQL()

    def run(self) -> None:
        orchestrator = self.factory.create_orchestrator(self.cfg)
        OrchestratorMigration(orchestrator).execute()
