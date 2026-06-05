from abc import ABC, abstractmethod


from ntxter.core.data.database import BaseDatabase
from ntxter.ports.factory.base import Orchestrator
from ntxter.ports.factory import migration
from ntxter.ports.factory.migration import register_migration_service, create_migrator

from ntxter.adapters.data.db_connection import PostgreSQLocalConnection


@register_migration_service("csv_to_local_postgresql")
class OrchestratorCSVToLocalPostgreSQL(Orchestrator):
    def __init__(self, db_conn: BaseDatabase, cfg=None):
        self.db_conn = db_conn
        self.cfg = cfg

    def execute(self) -> None:
        self.db_conn.connect()
