"""
Memory repository for managing cognitive memories.

This module provides database operations for Memory entities,
the core of the cognitive meeting intelligence system.
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
import logging
import json

from .base import BaseRepository
from ....models.entities import Memory, MemoryType, ContentType, Priority, Status, Vector
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class MemoryRepository(BaseRepository[Memory]):
    """
    Repository for managing memories.

    Provides specialized queries for:
    - Multi-level memory hierarchy (L0, L1, L2)
    - Content type and priority filtering
    - Project and meeting scoping
    - Lifecycle management (access, decay)
    - Parent-child relationships
    - Deliverable linkage
    - Performance-optimized batch operations
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize memory repository."""
        super().__init__(db_connection)

    def get_table_name(self) -> str:
        """Return the memories table name."""
        return "memories"

    def to_dict(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory entity to dictionary for storage."""
        return {
            "id": memory.id,
            "meeting_id": memory.meeting_id,
            "project_id": memory.project_id,
            "content": memory.content,
            "speaker": memory.speaker,
            "speaker_role": memory.speaker_role,
            "timestamp": float(memory.timestamp_ms / 1000) if memory.timestamp_ms else 0,
            "memory_type": memory.memory_type.value,
            "content_type": memory.content_type.value,
            "priority": memory.priority.value if memory.priority else None,
            "status": memory.status.value if memory.status else None,
            "due_date": self._serialize_datetime(memory.due_date),
            "owner": memory.owner,
            "level": memory.level,
            "importance_score": memory.importance_score,
            "decay_rate": memory.decay_rate,
            "access_count": memory.access_count,
            "last_accessed": self._serialize_datetime(memory.last_accessed),
            "created_at": self._serialize_datetime(memory.created_at),
            "parent_id": memory.parent_id,
            "deliverable_id": memory.deliverable_id,
            "qdrant_id": memory.qdrant_id,
            "dimensions_json": memory.dimensions_json,
        }

    def from_dict(self, data: Dict[str, Any]) -> Memory:
        """Convert dictionary from storage to Memory entity."""
        return Memory(
            id=data["id"],
            meeting_id=data["meeting_id"],
            project_id=data.get("project_id", ""),
            content=data["content"],
            speaker=data.get("speaker"),
            speaker_role=data.get("speaker_role"),
            timestamp_ms=int(data.get("timestamp", 0) * 1000),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            content_type=ContentType(data.get("content_type", "context")),
            priority=Priority(data["priority"]) if data.get("priority") else None,
            status=Status(data["status"]) if data.get("status") else None,
            due_date=self._deserialize_datetime(data.get("due_date")),
            owner=data.get("owner"),
            level=data.get("level", 2),
            importance_score=data.get("importance_score", 0.5),
            decay_rate=data.get("decay_rate", 0.1),
            access_count=data.get("access_count", 0),
            last_accessed=self._deserialize_datetime(data.get("last_accessed")),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
            parent_id=data.get("parent_id"),
            deliverable_id=data.get("deliverable_id"),
            qdrant_id=data.get("qdrant_id"),
            dimensions_json=data.get("dimensions_json"),
        )

    # Project and Meeting Scoped Queries

    async def get_by_project(
        self, project_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Memory]:
        """
        Get all memories for a project.

        Args:
            project_id: ID of the project
            limit: Maximum number of memories to retrieve
            offset: Number of memories to skip

        Returns:
            List of memories for the project
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                ORDER BY created_at DESC
            """

            params = [project_id]

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories for project {project_id}: {e}")
            raise

    async def get_by_meeting(
        self, meeting_id: str, order_by_timestamp: bool = True
    ) -> List[Memory]:
        """
        Get all memories from a specific meeting.

        Args:
            meeting_id: ID of the meeting
            order_by_timestamp: Whether to order by timestamp

        Returns:
            List of memories from the meeting
        """
        try:
            order_clause = "timestamp ASC" if order_by_timestamp else "created_at ASC"

            query = f"""
                SELECT * FROM {self._table_name}
                WHERE meeting_id = ?
                ORDER BY {order_clause}
            """

            results = await self.db.execute_query(query, (meeting_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories for meeting {meeting_id}: {e}")
            raise

    # Level-based Queries (3-tier system)

    async def get_by_level(
        self, level: int, project_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Get memories by hierarchy level (0=concepts, 1=contexts, 2=episodes).

        Args:
            level: Memory level (0, 1, or 2)
            project_id: Optional project filter
            limit: Maximum number of memories

        Returns:
            List of memories at the specified level
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE level = ?
            """

            params = [level]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY importance_score DESC, created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories by level {level}: {e}")
            raise

    # Content Type and Priority Queries

    async def get_by_content_type(
        self, content_type: ContentType, project_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Get memories by content type.

        Args:
            content_type: Type of content to filter by
            project_id: Optional project filter

        Returns:
            List of memories with the specified content type
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE content_type = ?
            """

            params = [content_type.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY created_at DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories by content type {content_type}: {e}")
            raise

    async def get_by_priority(
        self, priority: Priority, project_id: Optional[str] = None, include_higher: bool = True
    ) -> List[Memory]:
        """
        Get memories by priority level.

        Args:
            priority: Priority level to filter by
            project_id: Optional project filter
            include_higher: Include memories with higher priority

        Returns:
            List of memories with the specified priority
        """
        try:
            if include_higher:
                # Order priorities for comparison
                priority_order = {
                    Priority.LOW: 0,
                    Priority.MEDIUM: 1,
                    Priority.HIGH: 2,
                    Priority.CRITICAL: 3,
                }

                # Get all priorities at or above the specified level
                min_priority_value = priority_order[priority]
                valid_priorities = [
                    p.value for p, v in priority_order.items() if v >= min_priority_value
                ]

                placeholders = ",".join("?" * len(valid_priorities))
                query = f"""
                    SELECT * FROM {self._table_name}
                    WHERE priority IN ({placeholders})
                """

                params = list(valid_priorities)
            else:
                query = f"""
                    SELECT * FROM {self._table_name}
                    WHERE priority = ?
                """

                params = [priority.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY priority DESC, created_at DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories by priority {priority}: {e}")
            raise

    # Task Management Queries

    async def get_open_tasks(
        self, project_id: Optional[str] = None, owner: Optional[str] = None
    ) -> List[Memory]:
        """
        Get open action items and tasks.

        Args:
            project_id: Optional project filter
            owner: Optional owner filter

        Returns:
            List of open tasks
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE content_type IN ('action', 'commitment')
                AND status IN ('open', 'in_progress')
            """

            params = []

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            if owner:
                query += " AND owner = ?"
                params.append(owner)

            query += " ORDER BY priority DESC, due_date ASC NULLS LAST"

            results = await self.db.execute_query(query, tuple(params) if params else None)
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get open tasks: {e}")
            raise

    async def get_overdue_tasks(self, project_id: Optional[str] = None) -> List[Memory]:
        """
        Get overdue tasks.

        Args:
            project_id: Optional project filter

        Returns:
            List of overdue tasks
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE content_type IN ('action', 'commitment')
                AND status NOT IN ('completed', 'deferred')
                AND due_date < ?
            """

            params = [datetime.now().isoformat()]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY due_date ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get overdue tasks: {e}")
            raise

    # Lifecycle Management

    async def update_access_tracking(self, memory_id: str) -> bool:
        """
        Update access count and last accessed timestamp.

        Args:
            memory_id: ID of the memory accessed

        Returns:
            True if update successful
        """
        try:
            query = f"""
                UPDATE {self._table_name}
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """

            rows_affected = await self.db.execute_update(
                query, (datetime.now().isoformat(), memory_id)
            )

            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to update access tracking for {memory_id}: {e}")
            raise

    async def batch_update_access_tracking(self, memory_ids: List[str]) -> int:
        """
        Update access tracking for multiple memories.

        Args:
            memory_ids: List of memory IDs accessed

        Returns:
            Number of memories updated
        """
        if not memory_ids:
            return 0

        try:
            timestamp = datetime.now().isoformat()
            params_list = [(timestamp, memory_id) for memory_id in memory_ids]

            query = f"""
                UPDATE {self._table_name}
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """

            await self.db.execute_many(query, params_list)
            return len(memory_ids)

        except Exception as e:
            logger.error(f"Failed to batch update access tracking: {e}")
            raise

    async def apply_decay(self, decay_window_days: int = 30, batch_size: int = 1000) -> int:
        """
        Apply decay to memories based on age and access patterns.

        Args:
            decay_window_days: Only decay memories older than this
            batch_size: Process memories in batches

        Returns:
            Number of memories decayed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=decay_window_days)

            # Get memories to decay
            query = f"""
                SELECT id, importance_score, decay_rate, access_count, last_accessed
                FROM {self._table_name}
                WHERE created_at < ?
                AND importance_score > 0.1
                ORDER BY id
                LIMIT ?
            """

            total_decayed = 0

            while True:
                results = await self.db.execute_query(query, (cutoff_date.isoformat(), batch_size))

                if not results:
                    break

                # Calculate new importance scores
                updates = []
                for row in results:
                    # Apply decay based on time since last access
                    days_inactive = 0
                    if row["last_accessed"]:
                        last_accessed = datetime.fromisoformat(row["last_accessed"])
                        days_inactive = (datetime.now() - last_accessed).days

                    # Decay formula: importance * (1 - decay_rate * days_inactive / 365)
                    # But boost based on access count
                    decay_factor = max(0.1, 1 - (row["decay_rate"] * days_inactive / 365))
                    access_boost = min(1.5, 1 + row["access_count"] * 0.01)

                    new_score = min(1.0, row["importance_score"] * decay_factor * access_boost)

                    updates.append((new_score, row["id"]))

                # Apply updates
                update_query = f"""
                    UPDATE {self._table_name}
                    SET importance_score = ?
                    WHERE id = ?
                """

                await self.db.execute_many(update_query, updates)
                total_decayed += len(updates)

                if len(results) < batch_size:
                    break

            logger.info(f"Applied decay to {total_decayed} memories")
            return total_decayed

        except Exception as e:
            logger.error(f"Failed to apply decay: {e}")
            raise

    # Parent-Child Relationships

    async def get_children(self, parent_id: str) -> List[Memory]:
        """
        Get child memories of a parent (consolidated) memory.

        Args:
            parent_id: ID of the parent memory

        Returns:
            List of child memories
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE parent_id = ?
                ORDER BY created_at ASC
            """

            results = await self.db.execute_query(query, (parent_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get children of {parent_id}: {e}")
            raise

    # Deliverable Linkage

    async def get_by_deliverable(self, deliverable_id: str) -> List[Memory]:
        """
        Get all memories linked to a deliverable.

        Args:
            deliverable_id: ID of the deliverable

        Returns:
            List of memories linked to the deliverable
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE deliverable_id = ?
                ORDER BY created_at DESC
            """

            results = await self.db.execute_query(query, (deliverable_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get memories for deliverable {deliverable_id}: {e}")
            raise

    # Search and Analytics

    async def search_content(
        self,
        search_term: str,
        project_id: Optional[str] = None,
        content_types: Optional[List[ContentType]] = None,
    ) -> List[Memory]:
        """
        Search memories by content.

        Args:
            search_term: Term to search for
            project_id: Optional project filter
            content_types: Optional content type filters

        Returns:
            List of matching memories
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE content LIKE ?
            """

            params = [f"%{search_term}%"]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            if content_types:
                placeholders = ",".join("?" * len(content_types))
                query += f" AND content_type IN ({placeholders})"
                params.extend([ct.value for ct in content_types])

            query += " ORDER BY importance_score DESC, created_at DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search memories with term '{search_term}': {e}")
            raise

    async def get_memory_statistics(self, project_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about memories in a project.

        Args:
            project_id: ID of the project

        Returns:
            Dictionary with memory statistics
        """
        try:
            # Count by content type
            type_query = f"""
                SELECT content_type, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY content_type
            """

            type_results = await self.db.execute_query(type_query, (project_id,))

            # Count by level
            level_query = f"""
                SELECT level, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY level
            """

            level_results = await self.db.execute_query(level_query, (project_id,))

            # Task statistics
            task_query = f"""
                SELECT 
                    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_tasks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN due_date < datetime('now') 
                          AND status NOT IN ('completed', 'deferred') THEN 1 END) as overdue_tasks
                FROM {self._table_name}
                WHERE project_id = ?
                AND content_type IN ('action', 'commitment')
            """

            task_results = await self.db.execute_query(task_query, (project_id,))

            # Overall statistics
            overall_query = f"""
                SELECT 
                    COUNT(*) as total_memories,
                    AVG(importance_score) as avg_importance,
                    AVG(access_count) as avg_access_count,
                    COUNT(DISTINCT speaker) as unique_speakers
                FROM {self._table_name}
                WHERE project_id = ?
            """

            overall_results = await self.db.execute_query(overall_query, (project_id,))

            return {
                "by_content_type": {row["content_type"]: row["count"] for row in type_results},
                "by_level": {str(row["level"]): row["count"] for row in level_results},
                "tasks": task_results[0] if task_results else {},
                "overall": overall_results[0] if overall_results else {},
            }

        except Exception as e:
            logger.error(f"Failed to get memory statistics for project {project_id}: {e}")
            raise

    # Specialized Queries for Cognitive Features

    async def get_high_importance_memories(
        self, project_id: str, threshold: float = 0.7, limit: int = 100
    ) -> List[Memory]:
        """
        Get memories with high importance scores.

        Args:
            project_id: ID of the project
            threshold: Minimum importance score
            limit: Maximum number of memories

        Returns:
            List of high-importance memories
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                AND importance_score >= ?
                ORDER BY importance_score DESC
                LIMIT ?
            """

            results = await self.db.execute_query(query, (project_id, threshold, limit))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get high importance memories: {e}")
            raise

    async def get_frequently_accessed_memories(
        self, project_id: str, min_access_count: int = 5, limit: int = 50
    ) -> List[Memory]:
        """
        Get frequently accessed memories.

        Args:
            project_id: ID of the project
            min_access_count: Minimum access count
            limit: Maximum number of memories

        Returns:
            List of frequently accessed memories
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                AND access_count >= ?
                ORDER BY access_count DESC, last_accessed DESC
                LIMIT ?
            """

            results = await self.db.execute_query(query, (project_id, min_access_count, limit))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get frequently accessed memories: {e}")
            raise
