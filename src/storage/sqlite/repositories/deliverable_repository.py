"""
Deliverable repository for managing project deliverables.

This module provides database operations for Deliverable entities,
supporting deliverable tracking in consulting projects.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .base import BaseRepository
from ....models.entities import Deliverable, DeliverableType, DeliverableStatus
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class DeliverableRepository(BaseRepository[Deliverable]):
    """
    Repository for managing project deliverables.

    Provides specialized queries for:
    - Status tracking and updates
    - Due date management
    - Dependency tracking
    - Owner and reviewer assignments
    - Version control
    - Approval workflows
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize deliverable repository."""
        super().__init__(db_connection)

    def get_table_name(self) -> str:
        """Return the deliverables table name."""
        return "deliverables"

    def to_dict(self, deliverable: Deliverable) -> Dict[str, Any]:
        """Convert Deliverable entity to dictionary for storage."""
        return {
            "id": deliverable.id,
            "project_id": deliverable.project_id,
            "name": deliverable.name,
            "deliverable_type": deliverable.deliverable_type.value,
            "status": deliverable.status.value,
            "due_date": self._serialize_datetime(deliverable.due_date),
            "owner": deliverable.owner,
            "reviewer": deliverable.reviewer,
            "version": deliverable.version,
            "file_path": deliverable.file_path,
            "description": deliverable.description,
            "acceptance_criteria_json": self._serialize_json_field(deliverable.acceptance_criteria),
            "dependencies_json": self._serialize_json_field(deliverable.dependencies),
            "created_at": self._serialize_datetime(deliverable.created_at),
            "delivered_at": self._serialize_datetime(deliverable.delivered_at),
            "approved_at": self._serialize_datetime(deliverable.approved_at),
        }

    def from_dict(self, data: Dict[str, Any]) -> Deliverable:
        """Convert dictionary from storage to Deliverable entity."""
        return Deliverable(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            deliverable_type=DeliverableType(data["deliverable_type"]),
            status=DeliverableStatus(data["status"]),
            due_date=self._deserialize_datetime(data.get("due_date")),
            owner=data["owner"],
            reviewer=data.get("reviewer"),
            version=data.get("version", 1.0),
            file_path=data.get("file_path"),
            description=data.get("description"),
            acceptance_criteria=self._deserialize_json_field(
                data.get("acceptance_criteria_json"), {}
            ),
            dependencies=self._deserialize_json_field(data.get("dependencies_json"), []),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
            delivered_at=self._deserialize_datetime(data.get("delivered_at")),
            approved_at=self._deserialize_datetime(data.get("approved_at")),
        )

    async def get_by_project(
        self, project_id: str, include_completed: bool = True
    ) -> List[Deliverable]:
        """
        Get all deliverables for a project.

        Args:
            project_id: ID of the project
            include_completed: Whether to include completed deliverables

        Returns:
            List of deliverables for the project
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
            """

            params = [project_id]

            if not include_completed:
                query += " AND status NOT IN ('delivered', 'approved')"

            query += " ORDER BY due_date ASC NULLS LAST, created_at ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get deliverables for project {project_id}: {e}")
            raise

    async def get_by_status(
        self, status: DeliverableStatus, project_id: Optional[str] = None
    ) -> List[Deliverable]:
        """
        Get deliverables by status.

        Args:
            status: Deliverable status to filter by
            project_id: Optional project filter

        Returns:
            List of deliverables with the specified status
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE status = ?
            """

            params = [status.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY due_date ASC NULLS LAST"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get deliverables by status {status}: {e}")
            raise

    async def get_by_owner(self, owner: str, include_completed: bool = False) -> List[Deliverable]:
        """
        Get deliverables owned by a specific person.

        Args:
            owner: Name of the owner
            include_completed: Whether to include completed deliverables

        Returns:
            List of deliverables owned by the person
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE owner = ?
            """

            params = [owner]

            if not include_completed:
                query += " AND status NOT IN ('delivered', 'approved')"

            query += " ORDER BY due_date ASC NULLS LAST"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get deliverables by owner {owner}: {e}")
            raise

    async def get_by_reviewer(self, reviewer: str, pending_only: bool = True) -> List[Deliverable]:
        """
        Get deliverables assigned to a reviewer.

        Args:
            reviewer: Name of the reviewer
            pending_only: Only show deliverables pending review

        Returns:
            List of deliverables for review
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE reviewer = ?
            """

            params = [reviewer]

            if pending_only:
                query += " AND status = 'review'"

            query += " ORDER BY due_date ASC NULLS LAST"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get deliverables by reviewer {reviewer}: {e}")
            raise

    async def get_upcoming_deliverables(
        self, days_ahead: int = 7, project_id: Optional[str] = None
    ) -> List[Deliverable]:
        """
        Get deliverables due in the next N days.

        Args:
            days_ahead: Number of days to look ahead
            project_id: Optional project filter

        Returns:
            List of upcoming deliverables
        """
        try:
            future_date = datetime.now() + timedelta(days=days_ahead)

            query = f"""
                SELECT * FROM {self._table_name}
                WHERE due_date BETWEEN ? AND ?
                AND status NOT IN ('delivered', 'approved')
            """

            params = [datetime.now().isoformat(), future_date.isoformat()]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY due_date ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get upcoming deliverables: {e}")
            raise

    async def get_overdue_deliverables(self, project_id: Optional[str] = None) -> List[Deliverable]:
        """
        Get overdue deliverables.

        Args:
            project_id: Optional project filter

        Returns:
            List of overdue deliverables
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE due_date < ?
                AND status NOT IN ('delivered', 'approved')
            """

            params = [datetime.now().isoformat()]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY due_date ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get overdue deliverables: {e}")
            raise

    async def get_dependencies(self, deliverable_id: str) -> List[Deliverable]:
        """
        Get deliverables that this deliverable depends on.

        Args:
            deliverable_id: ID of the deliverable

        Returns:
            List of dependency deliverables
        """
        try:
            # First get the deliverable to access its dependencies
            deliverable = await self.get_by_id(deliverable_id)
            if not deliverable or not deliverable.dependencies:
                return []

            # Get all dependency deliverables
            placeholders = ",".join("?" * len(deliverable.dependencies))
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE id IN ({placeholders})
                ORDER BY due_date ASC NULLS LAST
            """

            results = await self.db.execute_query(query, tuple(deliverable.dependencies))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get dependencies for {deliverable_id}: {e}")
            raise

    async def get_dependents(self, deliverable_id: str) -> List[Deliverable]:
        """
        Get deliverables that depend on this deliverable.

        Args:
            deliverable_id: ID of the deliverable

        Returns:
            List of dependent deliverables
        """
        try:
            # This requires checking the dependencies_json field
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE dependencies_json LIKE ?
                ORDER BY due_date ASC NULLS LAST
            """

            # Search for the ID in the JSON array
            search_pattern = f'%"{deliverable_id}"%'

            results = await self.db.execute_query(query, (search_pattern,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get dependents for {deliverable_id}: {e}")
            raise

    async def update_status(
        self, deliverable_id: str, new_status: DeliverableStatus, update_timestamp: bool = True
    ) -> bool:
        """
        Update deliverable status with optional timestamp updates.

        Args:
            deliverable_id: ID of the deliverable
            new_status: New status
            update_timestamp: Whether to update delivered/approved timestamps

        Returns:
            True if update successful
        """
        try:
            # Build update query based on status
            if new_status == DeliverableStatus.DELIVERED and update_timestamp:
                query = f"""
                    UPDATE {self._table_name}
                    SET status = ?, delivered_at = ?
                    WHERE id = ?
                """
                params = (new_status.value, datetime.now().isoformat(), deliverable_id)

            elif new_status == DeliverableStatus.APPROVED and update_timestamp:
                query = f"""
                    UPDATE {self._table_name}
                    SET status = ?, approved_at = ?
                    WHERE id = ?
                """
                params = (new_status.value, datetime.now().isoformat(), deliverable_id)

            else:
                query = f"""
                    UPDATE {self._table_name}
                    SET status = ?
                    WHERE id = ?
                """
                params = (new_status.value, deliverable_id)

            rows_affected = await self.db.execute_update(query, params)
            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to update status for {deliverable_id}: {e}")
            raise

    async def update_version(self, deliverable_id: str, new_version: float) -> bool:
        """
        Update deliverable version.

        Args:
            deliverable_id: ID of the deliverable
            new_version: New version number

        Returns:
            True if update successful
        """
        try:
            query = f"""
                UPDATE {self._table_name}
                SET version = ?
                WHERE id = ?
            """

            rows_affected = await self.db.execute_update(query, (new_version, deliverable_id))
            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to update version for {deliverable_id}: {e}")
            raise

    async def search_deliverables(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Deliverable]:
        """
        Search deliverables by name or description.

        Args:
            search_term: Term to search for
            project_id: Optional project filter

        Returns:
            List of matching deliverables
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE (
                    name LIKE ? OR
                    description LIKE ?
                )
            """

            search_pattern = f"%{search_term}%"
            params = [search_pattern, search_pattern]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY due_date ASC NULLS LAST"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search deliverables with term '{search_term}': {e}")
            raise

    async def get_deliverable_statistics(self, project_id: str) -> Dict[str, Any]:
        """
        Get statistics about deliverables in a project.

        Args:
            project_id: ID of the project

        Returns:
            Dictionary with deliverable statistics
        """
        try:
            # Count by type
            type_query = f"""
                SELECT deliverable_type, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY deliverable_type
            """

            type_results = await self.db.execute_query(type_query, (project_id,))

            # Count by status
            status_query = f"""
                SELECT status, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY status
            """

            status_results = await self.db.execute_query(status_query, (project_id,))

            # Timeline statistics
            timeline_query = f"""
                SELECT 
                    COUNT(*) as total_deliverables,
                    COUNT(CASE WHEN status IN ('delivered', 'approved') THEN 1 END) as completed_count,
                    COUNT(CASE WHEN due_date < datetime('now') 
                          AND status NOT IN ('delivered', 'approved') THEN 1 END) as overdue_count,
                    COUNT(CASE WHEN due_date BETWEEN datetime('now') 
                          AND datetime('now', '+7 days') 
                          AND status NOT IN ('delivered', 'approved') THEN 1 END) as due_this_week,
                    AVG(CASE 
                        WHEN delivered_at IS NOT NULL AND due_date IS NOT NULL
                        THEN julianday(delivered_at) - julianday(due_date)
                        ELSE NULL
                    END) as avg_days_delay
                FROM {self._table_name}
                WHERE project_id = ?
            """

            timeline_results = await self.db.execute_query(timeline_query, (project_id,))

            # Owner workload
            owner_query = f"""
                SELECT owner, COUNT(*) as deliverable_count
                FROM {self._table_name}
                WHERE project_id = ?
                AND status NOT IN ('delivered', 'approved')
                GROUP BY owner
                ORDER BY deliverable_count DESC
                LIMIT 10
            """

            owner_results = await self.db.execute_query(owner_query, (project_id,))

            return {
                "by_type": {row["deliverable_type"]: row["count"] for row in type_results},
                "by_status": {row["status"]: row["count"] for row in status_results},
                "timeline": timeline_results[0] if timeline_results else {},
                "owner_workload": [
                    {"owner": row["owner"], "count": row["deliverable_count"]}
                    for row in owner_results
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get deliverable statistics for project {project_id}: {e}")
            raise
