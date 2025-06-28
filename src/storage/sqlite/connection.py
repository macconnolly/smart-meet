"""
SQLite database connection management.

Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
"""

import sqlite3
import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncGenerator, Generator, Dict, Any, List
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages SQLite database connections with async support.

    Uses ThreadPoolExecutor to run SQLite operations in separate threads
    since SQLite doesn't have native async support.
    """

    def __init__(self, db_path: str = "data/memories.db", pool_size: int = 5):
        """Initialize database connection manager."""
        self.db_path = Path(db_path)
        if db_path == ":memory:":
            self._db_url = db_path
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_url = str(self.db_path.absolute())
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=pool_size)

        # Enable foreign keys and WAL mode for better concurrency
        self._init_pragmas = [
            "PRAGMA foreign_keys = ON",
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA mmap_size = 30000000000",
        ]

    def _get_sync_connection(self) -> sqlite3.Connection:
        """Create a new synchronous database connection."""
        if self._db_url == ":memory:":
            if not hasattr(self, '_in_memory_conn') or self._in_memory_conn is None:
                self._in_memory_conn = sqlite3.connect(self._db_url, check_same_thread=False)
                self._in_memory_conn.row_factory = sqlite3.Row
                for pragma in self._init_pragmas:
                    self._in_memory_conn.execute(pragma)
            return self._in_memory_conn
        else:
            conn = sqlite3.connect(self._db_url, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable column access by name

            # Apply initialization pragmas
            for pragma in self._init_pragmas:
                conn.execute(pragma)

            return conn

    @contextmanager
    def get_sync_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a synchronous database connection with context manager."""
        conn = self._get_sync_connection()
        try:
            yield conn
        finally:
            if self._db_url != ":memory:": # Only close if not in-memory
                conn.close()

    async def execute_schema(self) -> None:
        """
        Execute the database schema.
        """
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        schema_sql = schema_file.read_text()

        def _execute():
            with self.get_sync_connection() as conn:
                # Execute the entire schema as a script
                conn.executescript(schema_sql)
                conn.commit()

                # Insert initial system metadata
                conn.execute(
                    "INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)",
                    ("schema_version", "2.0"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)",
                    ("schema_type", "consulting_enhanced"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)",
                    ("initialized_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                )
                conn.commit()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _execute)
        logger.info(f"Database schema executed successfully at {self._db_url}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        """
        Get a database connection with async context manager.

        Note: The connection is still synchronous, but we run it in a thread pool.
        """
        conn = await self._create_connection()
        try:
            yield conn
        finally:
            await self._close_connection(conn)

    async def _create_connection(self) -> sqlite3.Connection:
        """Create a connection in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_sync_connection)

    async def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a connection in the thread pool."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, conn.close)

    async def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dicts.
        """

        def _execute():
            with self.get_sync_connection() as conn:
                cursor = conn.execute(query, params or ())
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _execute)

    async def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query and return affected rows.
        """

        def _execute():
            with self.get_sync_connection() as conn:
                cursor = conn.execute(query, params or ())
                conn.commit()
                return cursor.rowcount

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _execute)

    async def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """
        Execute multiple queries in a transaction.
        """

        def _execute():
            with self.get_sync_connection() as conn:
                try:
                    conn.executemany(query, params_list)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise e

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _execute)

    async def transaction(self):
        """
        Create a transaction context for multiple operations.

        Usage:
            async with db.transaction() as conn:
                await db.execute_update("INSERT ...", conn=conn)
                await db.execute_update("UPDATE ...", conn=conn)
        """
        async with self._lock:
            conn = await self._create_connection()
            try:
                yield conn
                # Commit transaction
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, conn.commit)
            except Exception:
                # Rollback on error
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, conn.rollback)
                raise
            finally:
                await self._close_connection(conn)

    async def close(self) -> None:
        """Close all database connections and shutdown executor."""
        self._executor.shutdown(wait=True)
        logger.info("Database connection pool closed")


# Singleton instance
# TODO Day 1: Initialize in main application
_db_connection: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """Get the database connection instance."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection
