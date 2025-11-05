from abc import ABC, abstractmethod


import pandas as pd


from ntxter.core.base.errors import DatabaseConnectionError

class BaseDatabase(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database."""
        raise DatabaseConnectionError("Database connection method not implemented.")

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the database."""
        raise NotImplementedError
    
    @abstractmethod
    def update(self) -> None:
        #self._conn.commit()
        ...

    @abstractmethod
    def execute_query(self, query: str, *query_args) -> list | pd.DataFrame:
        """Execute a query against the database and return results."""
        raise NotImplementedError
    
    @abstractmethod
    def upsert(self, table: str, data: dict, primary_key) -> None:
        """Insert or update a record in the specified table."""
        raise NotImplementedError
    
    def _build_clauses(self, data: dict, uniques: list | str | int) -> tuple:
        if isinstance(uniques, (str, int)):
            uniques = [uniques]
        elif not isinstance(uniques, list):
            raise ValueError("uniques must be a list, str, or int")
        
        keys, placeholders = self.build_placesholders(data)
        updates = ", ".join(f"{k}=excluded.{k}" for k in keys if k != uniques)

        return keys, placeholders, updates
    
    def build_placesholders(self, data: dict) -> tuple:
        keys = data.keys()
        placeholders = ", ".join("?" for _ in keys)
        return keys, placeholders
    
    @abstractmethod
    def get_columns(self, table: str) -> list:
        ...