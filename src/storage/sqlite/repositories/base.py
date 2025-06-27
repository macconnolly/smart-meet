"""
Base repository pattern for database operations.

This module provides the abstract base class for all repositories,
implementing common CRUD operations and patterns.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import json
from datetime import datetime

from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository implementing common database operations.

    This class provides a foundation for all repositories with:
    - Connection management
    - Common CRUD operations
    - Error handling and logging
    - Transaction support
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize repository with database connection."""
        self.db = db_connection
        self._table_name = self.get_table_name()

    @abstractmethod
    def get_table_name(self) -> str:
        """Return the table name for this repository."""
        pass

    @abstractmethod
    def to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for storage."""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> T:
        """Convert dictionary from storage to entity."""
        pass

    async def create(self, entity: T) -> str:
        """
        Create a new entity in the database.

        Args:
            entity: The entity to create

        Returns:
            The ID of the created entity

        Raises:
            ValueError: If entity validation fails
            Exception: If database operation fails
        """
        try:
            data = self.to_dict(entity)

            # Build insert query dynamically
            columns = list(data.keys())
            placeholders = ["?" for _ in columns]

            query = f"""
                INSERT INTO {self._table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """

            values = [data[col] for col in columns]

            await self.db.execute_update(query, tuple(values))

            # Return the ID (assumes 'id' field exists)
            entity_id = data.get("id")
            logger.debug(f"Created {self._table_name} entity with ID: {entity_id}")

            return entity_id

        except Exception as e:
            logger.error(f"Failed to create {self._table_name} entity: {e}")
            raise

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            The entity if found, None otherwise
        """
        try:
            query = f"SELECT * FROM {self._table_name} WHERE id = ?"
            results = await self.db.execute_query(query, (entity_id,))

            if results:
                return self.from_dict(results[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get {self._table_name} by ID {entity_id}: {e}")
            raise

    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Retrieve all entities with optional pagination.

        Args:
            limit: Maximum number of entities to retrieve
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        try:
            query = f"SELECT * FROM {self._table_name}"
            params = []

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            results = await self.db.execute_query(query, tuple(params) if params else None)

            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get all {self._table_name} entities: {e}")
            raise

    async def update(self, entity: T) -> bool:
        """
        Update an existing entity.

        Args:
            entity: The entity to update

        Returns:
            True if update was successful, False otherwise
        """
        try:
            data = self.to_dict(entity)
            entity_id = data.pop("id")  # Remove ID from update fields

            # Build update query dynamically
            set_clauses = [f"{col} = ?" for col in data.keys()]
            query = f"""
                UPDATE {self._table_name}
                SET {', '.join(set_clauses)}
                WHERE id = ?
            """

            values = list(data.values()) + [entity_id]

            rows_affected = await self.db.execute_update(query, tuple(values))

            success = rows_affected > 0
            if success:
                logger.debug(f"Updated {self._table_name} entity with ID: {entity_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update {self._table_name} entity: {e}")
            raise

    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: The ID of the entity to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            query = f"DELETE FROM {self._table_name} WHERE id = ?"
            rows_affected = await self.db.execute_update(query, (entity_id,))

            success = rows_affected > 0
            if success:
                logger.debug(f"Deleted {self._table_name} entity with ID: {entity_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete {self._table_name} entity {entity_id}: {e}")
            raise

    async def exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists by ID.

        Args:
            entity_id: The ID to check

        Returns:
            True if entity exists, False otherwise
        """
        try:
            query = f"SELECT 1 FROM {self._table_name} WHERE id = ? LIMIT 1"
            results = await self.db.execute_query(query, (entity_id,))

            return len(results) > 0

        except Exception as e:
            logger.error(f"Failed to check existence of {self._table_name} {entity_id}: {e}")
            raise

    async def count(self, where_clause: str = "", params: Optional[Tuple] = None) -> int:
        """
        Count entities with optional filtering.

        Args:
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            Number of matching entities
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {self._table_name}"

            if where_clause:
                query += f" WHERE {where_clause}"

            results = await self.db.execute_query(query, params)

            return results[0]["count"] if results else 0

        except Exception as e:
            logger.error(f"Failed to count {self._table_name} entities: {e}")
            raise

    async def batch_create(self, entities: List[T]) -> int:
        """
        Create multiple entities in a single transaction.

        Args:
            entities: List of entities to create

        Returns:
            Number of entities created
        """
        if not entities:
            return 0

        try:
            # Convert all entities to dictionaries
            data_list = [self.to_dict(entity) for entity in entities]

            # Use first entity to determine columns
            columns = list(data_list[0].keys())
            placeholders = ["?" for _ in columns]

            query = f"""
                INSERT INTO {self._table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """

            # Prepare values for batch insert
            params_list = [tuple(data[col] for col in columns) for data in data_list]

            await self.db.execute_many(query, params_list)

            logger.debug(f"Batch created {len(entities)} {self._table_name} entities")
            return len(entities)

        except Exception as e:
            logger.error(f"Failed to batch create {self._table_name} entities: {e}")
            raise

    def _serialize_json_field(self, value: Any) -> Optional[str]:
        """Serialize a value to JSON string for storage."""
        if value is None:
            return None
        return json.dumps(value)

    def _deserialize_json_field(self, value: Optional[str], default: Any = None) -> Any:
        """Deserialize a JSON string from storage."""
        if value is None:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to deserialize JSON: {value}")
            return default

    def _serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format string."""
        if dt is None:
            return None
        return dt.isoformat()

    def _deserialize_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Deserialize ISO format string to datetime."""
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to deserialize datetime: {value}")
            return None
