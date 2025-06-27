"""
Meeting repository for managing meeting records.

This module provides database operations for Meeting entities,
supporting consulting-specific meeting types and metadata.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from .base import BaseRepository
from ....models.entities import Meeting, MeetingType, MeetingCategory
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class MeetingRepository(BaseRepository[Meeting]):
    """
    Repository for managing meetings.

    Provides specialized queries for:
    - Meeting type filtering
    - Time-based queries
    - Participant tracking
    - Project-scoped meeting retrieval
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize meeting repository."""
        super().__init__(db_connection)

    def get_table_name(self) -> str:
        """Return the meetings table name."""
        return "meetings"

    def to_dict(self, meeting: Meeting) -> Dict[str, Any]:
        """Convert Meeting entity to dictionary for storage."""
        return {
            "id": meeting.id,
            "project_id": meeting.project_id,
            "title": meeting.title,
            "meeting_type": meeting.meeting_type.value,
            "meeting_category": meeting.meeting_category.value,
            "is_recurring": meeting.is_recurring,
            "recurring_series_id": meeting.recurring_series_id,
            "start_time": self._serialize_datetime(meeting.start_time),
            "end_time": self._serialize_datetime(meeting.end_time),
            "participants_json": self._serialize_json_field(meeting.participants),
            "transcript_path": meeting.transcript_path,
            "agenda_json": self._serialize_json_field(meeting.agenda),
            "key_decisions_json": self._serialize_json_field(meeting.key_decisions),
            "action_items_json": self._serialize_json_field(meeting.action_items),
            "metadata_json": self._serialize_json_field(meeting.metadata),
            "created_at": self._serialize_datetime(meeting.created_at),
            "processed_at": self._serialize_datetime(meeting.processed_at),
            "memory_count": meeting.memory_count,
        }

    def from_dict(self, data: Dict[str, Any]) -> Meeting:
        """Convert dictionary from storage to Meeting entity."""
        return Meeting(
            id=data["id"],
            project_id=data.get("project_id", ""),
            title=data["title"],
            meeting_type=MeetingType(data.get("meeting_type", "working_session")),
            meeting_category=MeetingCategory(data.get("meeting_category", "internal")),
            is_recurring=data.get("is_recurring", False),
            recurring_series_id=data.get("recurring_series_id"),
            start_time=self._deserialize_datetime(data["start_time"]) or datetime.now(),
            end_time=self._deserialize_datetime(data.get("end_time")),
            participants=self._deserialize_json_field(data.get("participants_json"), []),
            transcript_path=data.get("transcript_path"),
            agenda=self._deserialize_json_field(data.get("agenda_json"), {}),
            key_decisions=self._deserialize_json_field(data.get("key_decisions_json"), []),
            action_items=self._deserialize_json_field(data.get("action_items_json"), []),
            metadata=self._deserialize_json_field(data.get("metadata_json"), {}),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
            processed_at=self._deserialize_datetime(data.get("processed_at")),
            memory_count=data.get("memory_count", 0),
        )

    async def get_by_project(
        self, project_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Meeting]:
        """
        Get all meetings for a project.

        Args:
            project_id: ID of the project
            limit: Maximum number of meetings to retrieve
            offset: Number of meetings to skip

        Returns:
            List of meetings for the project
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                ORDER BY start_time DESC
            """

            params = [project_id]

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get meetings for project {project_id}: {e}")
            raise

    async def get_by_type(
        self, meeting_type: MeetingType, project_id: Optional[str] = None
    ) -> List[Meeting]:
        """
        Get meetings by type, optionally filtered by project.

        Args:
            meeting_type: Type of meeting to filter by
            project_id: Optional project ID filter

        Returns:
            List of meetings of the specified type
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE meeting_type = ?
            """

            params = [meeting_type.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY start_time DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get meetings by type {meeting_type}: {e}")
            raise

    async def get_by_category(
        self, category: MeetingCategory, project_id: Optional[str] = None
    ) -> List[Meeting]:
        """
        Get meetings by category (internal/external).

        Args:
            category: Meeting category
            project_id: Optional project ID filter

        Returns:
            List of meetings in the category
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE meeting_category = ?
            """

            params = [category.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY start_time DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get meetings by category {category}: {e}")
            raise

    async def get_by_time_range(
        self, start_date: datetime, end_date: datetime, project_id: Optional[str] = None
    ) -> List[Meeting]:
        """
        Get meetings within a time range.

        Args:
            start_date: Start of time range
            end_date: End of time range
            project_id: Optional project ID filter

        Returns:
            List of meetings in the time range
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE start_time >= ? AND start_time <= ?
            """

            params = [self._serialize_datetime(start_date), self._serialize_datetime(end_date)]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY start_time ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get meetings by time range: {e}")
            raise

    async def get_unprocessed_meetings(self) -> List[Meeting]:
        """
        Get meetings that haven't been processed yet.

        Returns:
            List of unprocessed meetings
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE processed_at IS NULL
                AND transcript_path IS NOT NULL
                ORDER BY start_time ASC
            """

            results = await self.db.execute_query(query)
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get unprocessed meetings: {e}")
            raise

    async def mark_as_processed(self, meeting_id: str, memory_count: int) -> bool:
        """
        Mark a meeting as processed and update memory count.

        Args:
            meeting_id: ID of the meeting
            memory_count: Number of memories extracted

        Returns:
            True if update successful, False otherwise
        """
        try:
            query = f"""
                UPDATE {self._table_name}
                SET processed_at = ?,
                    memory_count = ?
                WHERE id = ?
            """

            rows_affected = await self.db.execute_update(
                query, (datetime.now().isoformat(), memory_count, meeting_id)
            )

            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to mark meeting {meeting_id} as processed: {e}")
            raise

    async def get_recurring_meetings(self, series_id: str) -> List[Meeting]:
        """
        Get all meetings in a recurring series.

        Args:
            series_id: ID of the recurring series

        Returns:
            List of meetings in the series
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE recurring_series_id = ?
                ORDER BY start_time ASC
            """

            results = await self.db.execute_query(query, (series_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get recurring meetings for series {series_id}: {e}")
            raise

    async def search_meetings(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Meeting]:
        """
        Search meetings by title or participant names.

        Args:
            search_term: Term to search for
            project_id: Optional project ID filter

        Returns:
            List of matching meetings
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE (
                    title LIKE ? OR
                    participants_json LIKE ?
                )
            """

            search_pattern = f"%{search_term}%"
            params = [search_pattern, search_pattern]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY start_time DESC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search meetings with term '{search_term}': {e}")
            raise

    async def get_meeting_statistics(self, project_id: str) -> Dict[str, Any]:
        """
        Get statistics about meetings in a project.

        Args:
            project_id: ID of the project

        Returns:
            Dictionary with meeting statistics
        """
        try:
            # Count by type
            type_query = f"""
                SELECT meeting_type, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY meeting_type
            """

            type_results = await self.db.execute_query(type_query, (project_id,))

            # Count by category
            category_query = f"""
                SELECT meeting_category, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY meeting_category
            """

            category_results = await self.db.execute_query(category_query, (project_id,))

            # Total meetings and memories
            totals_query = f"""
                SELECT 
                    COUNT(*) as total_meetings,
                    SUM(memory_count) as total_memories,
                    AVG(memory_count) as avg_memories_per_meeting
                FROM {self._table_name}
                WHERE project_id = ?
            """

            totals = await self.db.execute_query(totals_query, (project_id,))

            return {
                "by_type": {row["meeting_type"]: row["count"] for row in type_results},
                "by_category": {row["meeting_category"]: row["count"] for row in category_results},
                "total_meetings": totals[0]["total_meetings"] if totals else 0,
                "total_memories": totals[0]["total_memories"] if totals else 0,
                "avg_memories_per_meeting": totals[0]["avg_memories_per_meeting"] if totals else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get meeting statistics for project {project_id}: {e}")
            raise
