import os
from pathlib import Path
from dotenv import load_dotenv


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