"""
Memory connection repository for managing relationships between memories.

This module provides database operations for MemoryConnection entities,
enabling the graph-like structure of the cognitive system.
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime
import logging

from .base import BaseRepository
from ....models.entities import MemoryConnection, ConnectionType
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class MemoryConnectionRepository(BaseRepository[MemoryConnection]):
    """
    Repository for managing connections between memories.

    Provides specialized queries for:
    - Graph traversal operations
    - Connection strength analysis
    - Activation tracking
    - Connection type filtering
    - Batch connection creation
    - Network analysis
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize memory connection repository."""
        super().__init__(db_connection)

    def get_table_name(self) -> str:
        """Return the memory_connections table name."""
        return "memory_connections"

    def to_dict(self, connection: MemoryConnection) -> Dict[str, Any]:
        """Convert MemoryConnection entity to dictionary for storage."""
        return {
            "source_id": connection.source_id,
            "target_id": connection.target_id,
            "connection_strength": connection.connection_strength,
            "connection_type": connection.connection_type.value,
            "created_at": self._serialize_datetime(connection.created_at),
            "last_activated": self._serialize_datetime(connection.last_activated),
            "activation_count": connection.activation_count,
        }

    def from_dict(self, data: Dict[str, Any]) -> MemoryConnection:
        """Convert dictionary from storage to MemoryConnection entity."""
        return MemoryConnection(
            source_id=data["source_id"],
            target_id=data["target_id"],
            connection_strength=data.get("connection_strength", 0.5),
            connection_type=ConnectionType(data.get("connection_type", "sequential")),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
            last_activated=self._deserialize_datetime(data.get("last_activated")),
            activation_count=data.get("activation_count", 0),
        )

    # Override base methods for composite primary key

    async def create(self, connection: MemoryConnection) -> str:
        """
        Create a new connection between memories.

        Args:
            connection: The connection to create

        Returns:
            Composite key as "source_id:target_id"
        """
        try:
            data = self.to_dict(connection)

            query = f"""
                INSERT INTO {self._table_name} 
                (source_id, target_id, connection_strength, connection_type, 
                 created_at, last_activated, activation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            await self.db.execute_update(
                query,
                (
                    data["source_id"],
                    data["target_id"],
                    data["connection_strength"],
                    data["connection_type"],
                    data["created_at"],
                    data["last_activated"],
                    data["activation_count"],
                ),
            )

            composite_key = f"{data['source_id']}:{data['target_id']}"
            logger.debug(f"Created connection: {composite_key}")

            return composite_key

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise

    async def get_by_id(self, composite_key: str) -> Optional[MemoryConnection]:
        """
        Get connection by composite key.

        Args:
            composite_key: Format "source_id:target_id"

        Returns:
            Connection if found, None otherwise
        """
        try:
            source_id, target_id = composite_key.split(":", 1)
            return await self.get_connection(source_id, target_id)

        except ValueError:
            logger.error(f"Invalid composite key format: {composite_key}")
            return None
        except Exception as e:
            logger.error(f"Failed to get connection by ID: {e}")
            raise

    # Connection-specific queries

    async def get_connection(self, source_id: str, target_id: str) -> Optional[MemoryConnection]:
        """
        Get a specific connection between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID

        Returns:
            Connection if found, None otherwise
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE source_id = ? AND target_id = ?
            """

            results = await self.db.execute_query(query, (source_id, target_id))

            if results:
                return self.from_dict(results[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get connection {source_id}->{target_id}: {e}")
            raise

    async def get_outgoing_connections(
        self, source_id: str, connection_types: Optional[List[ConnectionType]] = None
    ) -> List[MemoryConnection]:
        """
        Get all outgoing connections from a memory.

        Args:
            source_id: Source memory ID
            connection_types: Optional filter by connection types

        Returns:
            List of outgoing connections
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE source_id = ?
            """

            params = [source_id]

            if connection_types:
                placeholders = ",".join("?" * len(connection_types))
                query += f" AND connection_type IN ({placeholders})"
                params.extend([ct.value for ct in connection_types])

            query += " ORDER BY connection_strength DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get outgoing connections from {source_id}: {e}")
            raise

    async def get_incoming_connections(
        self, target_id: str, connection_types: Optional[List[ConnectionType]] = None
    ) -> List[MemoryConnection]:
        """
        Get all incoming connections to a memory.

        Args:
            target_id: Target memory ID
            connection_types: Optional filter by connection types

        Returns:
            List of incoming connections
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE target_id = ?
            """

            params = [target_id]

            if connection_types:
                placeholders = ",".join("?" * len(connection_types))
                query += f" AND connection_type IN ({placeholders})"
                params.extend([ct.value for ct in connection_types])

            query += " ORDER BY connection_strength DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get incoming connections to {target_id}: {e}")
            raise

    async def get_bidirectional_connections(
        self, memory_id: str
    ) -> Dict[str, List[MemoryConnection]]:
        """
        Get both incoming and outgoing connections for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            Dictionary with 'incoming' and 'outgoing' connection lists
        """
        try:
            outgoing = await self.get_outgoing_connections(memory_id)
            incoming = await self.get_incoming_connections(memory_id)

            return {"outgoing": outgoing, "incoming": incoming}

        except Exception as e:
            logger.error(f"Failed to get bidirectional connections for {memory_id}: {e}")
            raise

    # Batch operations

    async def create_sequential_connections(
        self, memory_ids: List[str], connection_strength: float = 0.7
    ) -> int:
        """
        Create sequential connections between a list of memories.

        Args:
            memory_ids: List of memory IDs in sequence
            connection_strength: Strength for sequential connections

        Returns:
            Number of connections created
        """
        if len(memory_ids) < 2:
            return 0

        try:
            connections = []
            for i in range(len(memory_ids) - 1):
                connection = MemoryConnection(
                    source_id=memory_ids[i],
                    target_id=memory_ids[i + 1],
                    connection_strength=connection_strength,
                    connection_type=ConnectionType.SEQUENTIAL,
                    created_at=datetime.now(),
                )
                connections.append(connection)

            return await self.batch_create(connections)

        except Exception as e:
            logger.error(f"Failed to create sequential connections: {e}")
            raise

    async def batch_create_connections(
        self, connection_data: List[Tuple[str, str, float, ConnectionType]]
    ) -> int:
        """
        Create multiple connections in batch.

        Args:
            connection_data: List of (source_id, target_id, strength, type) tuples

        Returns:
            Number of connections created
        """
        if not connection_data:
            return 0

        try:
            connections = []
            for source_id, target_id, strength, conn_type in connection_data:
                connection = MemoryConnection(
                    source_id=source_id,
                    target_id=target_id,
                    connection_strength=strength,
                    connection_type=conn_type,
                    created_at=datetime.now(),
                )
                connections.append(connection)

            return await self.batch_create(connections)

        except Exception as e:
            logger.error(f"Failed to batch create connections: {e}")
            raise

    # Activation tracking

    async def update_activation(self, source_id: str, target_id: str) -> bool:
        """
        Update activation count and timestamp for a connection.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID

        Returns:
            True if update successful
        """
        try:
            query = f"""
                UPDATE {self._table_name}
                SET activation_count = activation_count + 1,
                    last_activated = ?
                WHERE source_id = ? AND target_id = ?
            """

            rows_affected = await self.db.execute_update(
                query, (datetime.now().isoformat(), source_id, target_id)
            )

            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to update activation for {source_id}->{target_id}: {e}")
            raise

    async def batch_update_activations(self, connection_pairs: List[Tuple[str, str]]) -> int:
        """
        Update activation for multiple connections.

        Args:
            connection_pairs: List of (source_id, target_id) tuples

        Returns:
            Number of connections updated
        """
        if not connection_pairs:
            return 0

        try:
            timestamp = datetime.now().isoformat()
            params_list = [
                (timestamp, source_id, target_id) for source_id, target_id in connection_pairs
            ]

            query = f"""
                UPDATE {self._table_name}
                SET activation_count = activation_count + 1,
                    last_activated = ?
                WHERE source_id = ? AND target_id = ?
            """

            await self.db.execute_many(query, params_list)
            return len(connection_pairs)

        except Exception as e:
            logger.error(f"Failed to batch update activations: {e}")
            raise

    # Network analysis

    async def get_strongly_connected_memories(
        self, memory_id: str, min_strength: float = 0.7, max_depth: int = 2
    ) -> Set[str]:
        """
        Get memories strongly connected within N hops.

        Args:
            memory_id: Starting memory ID
            min_strength: Minimum connection strength
            max_depth: Maximum hops to traverse

        Returns:
            Set of connected memory IDs
        """
        try:
            connected = {memory_id}
            current_level = {memory_id}

            for depth in range(max_depth):
                if not current_level:
                    break

                # Get next level connections
                placeholders = ",".join("?" * len(current_level))
                query = f"""
                    SELECT DISTINCT target_id
                    FROM {self._table_name}
                    WHERE source_id IN ({placeholders})
                    AND connection_strength >= ?
                    AND target_id NOT IN ({','.join('?' * len(connected))})
                """

                params = list(current_level) + [min_strength] + list(connected)
                results = await self.db.execute_query(query, tuple(params))

                next_level = {row["target_id"] for row in results}
                connected.update(next_level)
                current_level = next_level

            connected.remove(memory_id)  # Remove starting node
            return connected

        except Exception as e:
            logger.error(f"Failed to get strongly connected memories: {e}")
            raise

    async def get_connection_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about memory connections.

        Args:
            project_id: Optional project filter (requires join with memories)

        Returns:
            Dictionary with connection statistics
        """
        try:
            base_query = f"FROM {self._table_name} mc"
            join_clause = ""
            where_clause = ""
            params = []

            if project_id:
                join_clause = """
                    INNER JOIN memories m1 ON mc.source_id = m1.id
                    INNER JOIN memories m2 ON mc.target_id = m2.id
                """
                where_clause = "WHERE m1.project_id = ? AND m2.project_id = ?"
                params = [project_id, project_id]

            # Count by connection type
            type_query = f"""
                SELECT connection_type, COUNT(*) as count
                {base_query}
                {join_clause}
                {where_clause}
                GROUP BY connection_type
            """

            type_results = await self.db.execute_query(
                type_query, tuple(params) if params else None
            )

            # Overall statistics
            stats_query = f"""
                SELECT 
                    COUNT(*) as total_connections,
                    AVG(connection_strength) as avg_strength,
                    MAX(activation_count) as max_activations,
                    AVG(activation_count) as avg_activations
                {base_query}
                {join_clause}
                {where_clause}
            """

            stats_results = await self.db.execute_query(
                stats_query, tuple(params) if params else None
            )

            # Most connected memories
            connected_query = f"""
                SELECT memory_id, connection_count
                FROM (
                    SELECT source_id as memory_id, COUNT(*) as connection_count
                    {base_query}
                    {join_clause}
                    {where_clause}
                    GROUP BY source_id
                    
                    UNION ALL
                    
                    SELECT target_id as memory_id, COUNT(*) as connection_count
                    {base_query}
                    {join_clause}
                    {where_clause}
                    GROUP BY target_id
                ) combined
                GROUP BY memory_id
                ORDER BY SUM(connection_count) DESC
                LIMIT 10
            """

            # For the union query, we need to double the params
            connected_params = params + params if params else None
            connected_results = await self.db.execute_query(connected_query, connected_params)

            return {
                "by_type": {row["connection_type"]: row["count"] for row in type_results},
                "overall": stats_results[0] if stats_results else {},
                "most_connected": [
                    {"memory_id": row["memory_id"], "connections": row["connection_count"]}
                    for row in connected_results
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get connection statistics: {e}")
            raise

    async def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find a path between two memories using BFS.

        Args:
            source_id: Starting memory ID
            target_id: Target memory ID
            max_depth: Maximum path length

        Returns:
            List of memory IDs representing the path, or None if no path exists
        """
        try:
            if source_id == target_id:
                return [source_id]

            # BFS to find shortest path
            queue = [(source_id, [source_id])]
            visited = {source_id}

            for _ in range(max_depth):
                next_queue = []

                for current_id, path in queue:
                    # Get outgoing connections
                    connections = await self.get_outgoing_connections(current_id)

                    for conn in connections:
                        if conn.target_id == target_id:
                            return path + [target_id]

                        if conn.target_id not in visited:
                            visited.add(conn.target_id)
                            next_queue.append((conn.target_id, path + [conn.target_id]))

                queue = next_queue
                if not queue:
                    break

            return None

        except Exception as e:
            logger.error(f"Failed to find path from {source_id} to {target_id}: {e}")
            raise
