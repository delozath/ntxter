import os
from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import text as pgre_text

import sqlite3
import pandas as pd


from ntxter.core.data.database import BaseDatabase
from ntxter.core.base.errors import DatabaseConnectionError

class SQLiteConnection(BaseDatabase):
    def connect(self) -> None:
        load_dotenv()
        
        path = Path(os.getenv("DB_PATH"))
        if path.exists():
            self._conn = sqlite3.connect(path)
            self.cursor = self._conn.cursor()
        else:
            raise DatabaseConnectionError(f"Failed to connect to database")
    
    def update(self) -> None:
        self._conn.commit()

    def disconnect(self) -> None:
        self.cursor.close()

    def execute_query(self, table, query: str, fields: tuple) -> list | pd.DataFrame:
        self.cursor.execute(query, fields)
        res = list(self.cursor.fetchall())
        cols = self.get_columns(table)

        df = pd.DataFrame(res)
        df.columns = cols
        
        return df
    
    def get_columns(self, table: str) -> list:
        self.cursor.execute(f"PRAGMA table_info({table});")
        cols = [row[1] for row in self.cursor.fetchall()]
        return cols

    def upsert(self, table: str, data: dict, primary_key) -> None:
        if not data:
            raise ValueError("Empty data for UPSERT")

        keys, placeholders, updates = self._build_clauses(data, primary_key)

        sql = (
            f"INSERT INTO {table} ({', '.join(keys)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT({primary_key}) DO UPDATE SET {updates};"
        )

        self.cursor.execute(
            sql, 
            tuple(data.values())
         )
        self.update()


class PostgreSQLocalConnection(BaseDatabase):
    def connect(self, /, **kwargs) -> None:
        db_prefix = kwargs.pop('db_prefix', 'DB')
        load_dotenv()
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine import URL
        except ImportError as exc:
            raise DatabaseConnectionError("SQLAlchemy not installed") from exc

        try:
            url = URL.create(
                "postgresql+psycopg",
                host=os.getenv(f"{db_prefix}_HOST", "localhost"),
                port=int(os.getenv(f"{db_prefix}_PORT", "5432")),
                database=os.getenv(f"{db_prefix}_DB"),
                username=os.getenv(f"{db_prefix}_USER"),
                password=os.getenv(f"{db_prefix}_PASSWORD")
            )

            self._engine = create_engine(url, pool_pre_ping=True)
            self._conn = self._engine.connect()

        except Exception as exc:
            raise DatabaseConnectionError("Failed to connect to database") from exc

    def update(self) -> None:
        self._conn.commit()

    def disconnect(self) -> None:
        self._conn.close()
        self._engine.dispose()

    ##----------------- Under revision ----------------------##
    def execute_query(self, table, query: str, fields: tuple | dict = ()) -> list | pd.DataFrame:
        statement, params = self._prepare_query(query, fields)
        result = self._conn.execute(statement, params)

        if not result.returns_rows:
            return pd.DataFrame(columns=self.get_columns(table))

        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def get_table_names(self):
        statement = pgre_text(
            """
            SELECT schemaname, tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
            """
        )
        result = self._conn.execute(
            statement
        )
        breakpoint()
        return 1 

    def get_columns(self, table: str) -> list:

        schema, table_name = self._split_table_name(table)
        statement = pgre_text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :table_name
              AND (:schema IS NULL OR table_schema = :schema)
            ORDER BY ordinal_position;
            """
        )
        result = self._conn.execute(
            statement,
            {"schema": schema, "table_name": table_name},
        )
        return [row[0] for row in result.fetchall()]

    def upsert(self, table: str, data: dict, primary_key) -> None:
        if not data:
            raise ValueError("Empty data for UPSERT")

        keys, placeholders, updates = self._build_clauses(data, primary_key)
        conflict = self._build_conflict_target(primary_key)

        sql = (
            f"INSERT INTO {table} ({', '.join(keys)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {updates};"
        )

        statement, params = self._prepare_query(sql, data)
        self._conn.execute(statement, params)
        self.update()

    def _build_clauses(self, data: dict, uniques: list | str | int) -> tuple:
        if isinstance(uniques, (str, int)):
            uniques = [uniques]
        elif not isinstance(uniques, list):
            raise ValueError("uniques must be a list, str, or int")

        keys, placeholders = self.build_placesholders(data)
        updates = ", ".join(f"{k}=excluded.{k}" for k in keys if k not in uniques)

        return keys, placeholders, updates

    def build_placesholders(self, data: dict) -> tuple:
        keys = data.keys()
        placeholders = ", ".join(f":{k}" for k in keys)
        return keys, placeholders

    def _prepare_query(self, query: str, fields: tuple | dict):
        from sqlalchemy import text

        if isinstance(fields, dict):
            return text(query), fields

        fields = tuple(fields or ())
        if not fields:
            return text(query), {}

        parts = query.split("?")
        if len(parts) == 1:
            return text(query), {f"param_{i}": value for i, value in enumerate(fields)}

        query = "".join(
            part + (f":param_{i}" if i < len(fields) else "")
            for i, part in enumerate(parts)
        )
        return text(query), {f"param_{i}": value for i, value in enumerate(fields)}

    def _build_conflict_target(self, primary_key) -> str:
        if isinstance(primary_key, (str, int)):
            primary_key = [primary_key]
        elif not isinstance(primary_key, list):
            raise ValueError("primary_key must be a list, str, or int")

        return ", ".join(str(key) for key in primary_key)

    def _split_table_name(self, table: str) -> tuple[str | None, str]:
        parts = table.split(".", 1)
        if len(parts) == 1:
            return None, parts[0]

        return parts[0], parts[1]
