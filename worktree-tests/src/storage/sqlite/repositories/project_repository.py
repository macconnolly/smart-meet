"""
Project repository for managing consulting projects.

This module provides database operations for Project entities,
supporting the consulting-specific features of the system.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .base import BaseRepository
from ....models.entities import Project, ProjectType, ProjectStatus
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository[Project]):
    """
    Repository for managing consulting projects.
    
    Provides specialized queries for:
    - Active project retrieval
    - Client-based filtering
    - Budget tracking
    - Project lifecycle management
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        """Initialize project repository."""
        super().__init__(db_connection)
    
    def get_table_name(self) -> str:
        """Return the projects table name."""
        return "projects"
    
    def to_dict(self, project: Project) -> Dict[str, Any]:
        """Convert Project entity to dictionary for storage."""
        return {
            "id": project.id,
            "name": project.name,
            "client_name": project.client_name,
            "project_type": project.project_type.value,
            "status": project.status.value,
            "start_date": self._serialize_datetime(project.start_date),
            "end_date": self._serialize_datetime(project.end_date),
            "project_manager": project.project_manager,
            "engagement_code": project.engagement_code,
            "budget_hours": project.budget_hours,
            "consumed_hours": project.consumed_hours,
            "metadata_json": self._serialize_json_field(project.metadata),
            "created_at": self._serialize_datetime(project.created_at),
            "updated_at": self._serialize_datetime(project.updated_at),
        }
    
    def from_dict(self, data: Dict[str, Any]) -> Project:
        """Convert dictionary from storage to Project entity."""
        return Project(
            id=data["id"],
            name=data["name"],
            client_name=data["client_name"],
            project_type=ProjectType(data["project_type"]),
            status=ProjectStatus(data["status"]),
            start_date=self._deserialize_datetime(data["start_date"]) or datetime.now(),
            end_date=self._deserialize_datetime(data.get("end_date")),
            project_manager=data.get("project_manager"),
            engagement_code=data.get("engagement_code"),
            budget_hours=data.get("budget_hours"),
            consumed_hours=data.get("consumed_hours", 0),
            metadata=self._deserialize_json_field(data.get("metadata_json"), {}),
            created_at=self._deserialize_datetime(data["created_at"]) or datetime.now(),
            updated_at=self._deserialize_datetime(data["updated_at"]) or datetime.now(),
        )
    
    async def get_active_projects(self) -> List[Project]:
        """
        Get all active projects.
        
        Returns:
            List of active projects
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE status = ?
                ORDER BY start_date DESC
            """
            
            results = await self.db.execute_query(query, (ProjectStatus.ACTIVE.value,))
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get active projects: {e}")
            raise
    
    async def get_by_client(self, client_name: str) -> List[Project]:
        """
        Get all projects for a specific client.
        
        Args:
            client_name: Name of the client
            
        Returns:
            List of projects for the client
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE client_name = ?
                ORDER BY start_date DESC
            """
            
            results = await self.db.execute_query(query, (client_name,))
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get projects for client {client_name}: {e}")
            raise
    
    async def get_by_engagement_code(self, engagement_code: str) -> Optional[Project]:
        """
        Get project by engagement code.
        
        Args:
            engagement_code: Unique engagement code
            
        Returns:
            Project if found, None otherwise
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE engagement_code = ?
            """
            
            results = await self.db.execute_query(query, (engagement_code,))
            
            if results:
                return self.from_dict(results[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get project by engagement code {engagement_code}: {e}")
            raise
    
    async def update_consumed_hours(self, project_id: str, hours_to_add: int) -> bool:
        """
        Update consumed hours for a project.
        
        Args:
            project_id: ID of the project
            hours_to_add: Hours to add to consumed_hours
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            query = f"""
                UPDATE {self._table_name}
                SET consumed_hours = consumed_hours + ?,
                    updated_at = ?
                WHERE id = ?
            """
            
            rows_affected = await self.db.execute_update(
                query, 
                (hours_to_add, datetime.now().isoformat(), project_id)
            )
            
            return rows_affected > 0
            
        except Exception as e:
            logger.error(f"Failed to update consumed hours for project {project_id}: {e}")
            raise
    
    async def get_projects_by_status(self, status: ProjectStatus) -> List[Project]:
        """
        Get projects by status.
        
        Args:
            status: Project status to filter by
            
        Returns:
            List of projects with the specified status
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE status = ?
                ORDER BY start_date DESC
            """
            
            results = await self.db.execute_query(query, (status.value,))
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get projects by status {status}: {e}")
            raise
    
    async def get_projects_by_manager(self, manager_name: str) -> List[Project]:
        """
        Get projects managed by a specific person.
        
        Args:
            manager_name: Name of the project manager
            
        Returns:
            List of projects managed by the person
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE project_manager = ?
                ORDER BY start_date DESC
            """
            
            results = await self.db.execute_query(query, (manager_name,))
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get projects by manager {manager_name}: {e}")
            raise
    
    async def get_overbudget_projects(self) -> List[Project]:
        """
        Get projects that have exceeded their budget.
        
        Returns:
            List of overbudget projects
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE budget_hours IS NOT NULL
                AND consumed_hours > budget_hours
                ORDER BY (consumed_hours - budget_hours) DESC
            """
            
            results = await self.db.execute_query(query)
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get overbudget projects: {e}")
            raise
    
    async def search_projects(
        self, 
        search_term: str,
        include_completed: bool = False
    ) -> List[Project]:
        """
        Search projects by name, client, or engagement code.
        
        Args:
            search_term: Term to search for
            include_completed: Whether to include completed projects
            
        Returns:
            List of matching projects
        """
        try:
            status_filter = "" if include_completed else "AND status != 'completed'"
            
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE (
                    name LIKE ? OR
                    client_name LIKE ? OR
                    engagement_code LIKE ?
                )
                {status_filter}
                ORDER BY start_date DESC
            """
            
            search_pattern = f"%{search_term}%"
            params = (search_pattern, search_pattern, search_pattern)
            
            results = await self.db.execute_query(query, params)
            return [self.from_dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to search projects with term '{search_term}': {e}")
            raise
