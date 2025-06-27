"""
API dependency injection utilities.

This module provides dependency functions for FastAPI endpoints.
"""

from typing import Generator, Optional
from ..storage.sqlite.connection import DatabaseConnection
from ..storage.qdrant.vector_store import QdrantVectorStore
from ..core.config import get_settings

# Global instances (set by main.py)
_db_connection: Optional[DatabaseConnection] = None
_vector_store: Optional[QdrantVectorStore] = None


def set_db_connection(db: DatabaseConnection) -> None:
    """Set the global database connection."""
    global _db_connection
    _db_connection = db


def set_vector_store(store: QdrantVectorStore) -> None:
    """Set the global vector store."""
    global _vector_store
    _vector_store = store


async def get_db_connection() -> DatabaseConnection:
    """Get database connection for dependency injection."""
    if _db_connection is None:
        raise RuntimeError("Database connection not initialized")
    return _db_connection


async def get_vector_store_instance() -> QdrantVectorStore:
    """Get vector store for dependency injection."""
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _vector_store