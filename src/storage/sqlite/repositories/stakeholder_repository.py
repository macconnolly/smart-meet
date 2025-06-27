"""
Stakeholder repository for managing project stakeholders.

This module provides database operations for Stakeholder entities,
supporting stakeholder management in consulting projects.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .base import BaseRepository
from ....models.entities import Stakeholder, StakeholderType, InfluenceLevel, EngagementLevel
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class StakeholderRepository(BaseRepository[Stakeholder]):
    """
    Repository for managing project stakeholders.

    Provides specialized queries for:
    - Stakeholder influence analysis
    - Engagement level tracking
    - Project-based filtering
    - Organization grouping
    - Stakeholder network analysis
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize stakeholder repository."""
        super().__init__(db_connection)

    def get_table_name(self) -> str:
        """Return the stakeholders table name."""
        return "stakeholders"

    def to_dict(self, stakeholder: Stakeholder) -> Dict[str, Any]:
        """Convert Stakeholder entity to dictionary for storage."""
        return {
            "id": stakeholder.id,
            "project_id": stakeholder.project_id,
            "name": stakeholder.name,
            "organization": stakeholder.organization,
            "role": stakeholder.role,
            "stakeholder_type": stakeholder.stakeholder_type.value,
            "influence_level": stakeholder.influence_level.value,
            "engagement_level": stakeholder.engagement_level.value,
            "email": stakeholder.email,
            "notes": stakeholder.notes,
            "created_at": self._serialize_datetime(stakeholder.created_at),
        }

    def from_dict(self, data: Dict[str, Any]) -> Stakeholder:
        """Convert dictionary from storage to Stakeholder entity."""
        return Stakeholder(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            organization=data["organization"],
            role=data["role"],
            stakeholder_type=StakeholderType(data["stakeholder_type"]),
            influence_level=InfluenceLevel(data["influence_level"]),
            engagement_level=EngagementLevel(data["engagement_level"]),
            email=data.get("email"),
            notes=data.get("notes"),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
        )

    async def get_by_project(self, project_id: str) -> List[Stakeholder]:
        """
        Get all stakeholders for a project.

        Args:
            project_id: ID of the project

        Returns:
            List of stakeholders in the project
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                ORDER BY influence_level DESC, name ASC
            """

            results = await self.db.execute_query(query, (project_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get stakeholders for project {project_id}: {e}")
            raise

    async def get_by_influence_level(
        self, influence_level: InfluenceLevel, project_id: Optional[str] = None
    ) -> List[Stakeholder]:
        """
        Get stakeholders by influence level.

        Args:
            influence_level: Level of influence to filter by
            project_id: Optional project filter

        Returns:
            List of stakeholders with the specified influence level
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE influence_level = ?
            """

            params = [influence_level.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY engagement_level DESC, name ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get stakeholders by influence {influence_level}: {e}")
            raise

    async def get_by_engagement_level(
        self, engagement_level: EngagementLevel, project_id: Optional[str] = None
    ) -> List[Stakeholder]:
        """
        Get stakeholders by engagement level.

        Args:
            engagement_level: Level of engagement to filter by
            project_id: Optional project filter

        Returns:
            List of stakeholders with the specified engagement level
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE engagement_level = ?
            """

            params = [engagement_level.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY influence_level DESC, name ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get stakeholders by engagement {engagement_level}: {e}")
            raise

    async def get_key_stakeholders(self, project_id: str) -> List[Stakeholder]:
        """
        Get key stakeholders (high influence or champions).

        Args:
            project_id: ID of the project

        Returns:
            List of key stakeholders
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                AND (
                    influence_level = 'high' OR
                    engagement_level = 'champion'
                )
                ORDER BY 
                    CASE influence_level 
                        WHEN 'high' THEN 1 
                        WHEN 'medium' THEN 2 
                        ELSE 3 
                    END,
                    CASE engagement_level
                        WHEN 'champion' THEN 1
                        WHEN 'supportive' THEN 2
                        WHEN 'neutral' THEN 3
                        WHEN 'skeptical' THEN 4
                        ELSE 5
                    END
            """

            results = await self.db.execute_query(query, (project_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get key stakeholders for project {project_id}: {e}")
            raise

    async def get_challenging_stakeholders(self, project_id: str) -> List[Stakeholder]:
        """
        Get challenging stakeholders (skeptical or resistant).

        Args:
            project_id: ID of the project

        Returns:
            List of challenging stakeholders
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_id = ?
                AND engagement_level IN ('skeptical', 'resistant')
                ORDER BY influence_level DESC
            """

            results = await self.db.execute_query(query, (project_id,))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get challenging stakeholders: {e}")
            raise

    async def get_by_organization(
        self, organization: str, project_id: Optional[str] = None
    ) -> List[Stakeholder]:
        """
        Get all stakeholders from an organization.

        Args:
            organization: Name of the organization
            project_id: Optional project filter

        Returns:
            List of stakeholders from the organization
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE organization = ?
            """

            params = [organization]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY influence_level DESC, name ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get stakeholders from {organization}: {e}")
            raise

    async def get_by_type(
        self, stakeholder_type: StakeholderType, project_id: Optional[str] = None
    ) -> List[Stakeholder]:
        """
        Get stakeholders by type.

        Args:
            stakeholder_type: Type of stakeholder
            project_id: Optional project filter

        Returns:
            List of stakeholders of the specified type
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE stakeholder_type = ?
            """

            params = [stakeholder_type.value]

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY influence_level DESC, name ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get stakeholders by type {stakeholder_type}: {e}")
            raise

    async def search_stakeholders(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Stakeholder]:
        """
        Search stakeholders by name, role, or organization.

        Args:
            search_term: Term to search for
            project_id: Optional project filter

        Returns:
            List of matching stakeholders
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE (
                    name LIKE ? OR
                    role LIKE ? OR
                    organization LIKE ? OR
                    notes LIKE ?
                )
            """

            search_pattern = f"%{search_term}%"
            params = [search_pattern] * 4

            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)

            query += " ORDER BY influence_level DESC, name ASC"

            results = await self.db.execute_query(query, tuple(params))
            return [self.from_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search stakeholders with term '{search_term}': {e}")
            raise

    async def get_stakeholder_statistics(self, project_id: str) -> Dict[str, Any]:
        """
        Get statistics about stakeholders in a project.

        Args:
            project_id: ID of the project

        Returns:
            Dictionary with stakeholder statistics
        """
        try:
            # Count by type
            type_query = f"""
                SELECT stakeholder_type, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY stakeholder_type
            """

            type_results = await self.db.execute_query(type_query, (project_id,))

            # Count by influence level
            influence_query = f"""
                SELECT influence_level, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY influence_level
            """

            influence_results = await self.db.execute_query(influence_query, (project_id,))

            # Count by engagement level
            engagement_query = f"""
                SELECT engagement_level, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY engagement_level
            """

            engagement_results = await self.db.execute_query(engagement_query, (project_id,))

            # Organization distribution
            org_query = f"""
                SELECT organization, COUNT(*) as count
                FROM {self._table_name}
                WHERE project_id = ?
                GROUP BY organization
                ORDER BY count DESC
                LIMIT 10
            """

            org_results = await self.db.execute_query(org_query, (project_id,))

            # Overall counts
            overall_query = f"""
                SELECT 
                    COUNT(*) as total_stakeholders,
                    COUNT(DISTINCT organization) as unique_organizations,
                    COUNT(CASE WHEN influence_level = 'high' THEN 1 END) as high_influence_count,
                    COUNT(CASE WHEN engagement_level IN ('skeptical', 'resistant') THEN 1 END) as challenging_count
                FROM {self._table_name}
                WHERE project_id = ?
            """

            overall = await self.db.execute_query(overall_query, (project_id,))

            return {
                "by_type": {row["stakeholder_type"]: row["count"] for row in type_results},
                "by_influence": {row["influence_level"]: row["count"] for row in influence_results},
                "by_engagement": {
                    row["engagement_level"]: row["count"] for row in engagement_results
                },
                "top_organizations": [
                    {"organization": row["organization"], "count": row["count"]}
                    for row in org_results
                ],
                "overall": overall[0] if overall else {},
            }

        except Exception as e:
            logger.error(f"Failed to get stakeholder statistics for project {project_id}: {e}")
            raise

    async def update_engagement_level(
        self,
        stakeholder_id: str,
        new_engagement_level: EngagementLevel,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update a stakeholder's engagement level.

        Args:
            stakeholder_id: ID of the stakeholder
            new_engagement_level: New engagement level
            notes: Optional notes about the change

        Returns:
            True if update successful
        """
        try:
            if notes:
                # Get existing notes
                existing = await self.get_by_id(stakeholder_id)
                if existing and existing.notes:
                    # Append to existing notes with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d")
                    notes = f"{existing.notes}\n[{timestamp}] Engagement changed to {new_engagement_level.value}: {notes}"

                query = f"""
                    UPDATE {self._table_name}
                    SET engagement_level = ?, notes = ?
                    WHERE id = ?
                """

                params = (new_engagement_level.value, notes, stakeholder_id)
            else:
                query = f"""
                    UPDATE {self._table_name}
                    SET engagement_level = ?
                    WHERE id = ?
                """

                params = (new_engagement_level.value, stakeholder_id)

            rows_affected = await self.db.execute_update(query, params)
            return rows_affected > 0

        except Exception as e:
            logger.error(f"Failed to update engagement level for {stakeholder_id}: {e}")
            raise
