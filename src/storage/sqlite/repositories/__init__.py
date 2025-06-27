"""
SQLite repository implementations for the Cognitive Meeting Intelligence system.

This module provides database access layers for all entities in the system,
implementing the repository pattern for clean separation of concerns.
"""

from .base import BaseRepository
from .project_repository import ProjectRepository
from .meeting_repository import MeetingRepository
from .memory_repository import MemoryRepository
from .memory_connection_repository import MemoryConnectionRepository
from .stakeholder_repository import StakeholderRepository
from .deliverable_repository import DeliverableRepository

__all__ = [
    "BaseRepository",
    "ProjectRepository",
    "MeetingRepository", 
    "MemoryRepository",
    "MemoryConnectionRepository",
    "StakeholderRepository",
    "DeliverableRepository",
]

# Repository factory functions for dependency injection

def get_project_repository(db_connection) -> ProjectRepository:
    """Create a project repository instance."""
    return ProjectRepository(db_connection)

def get_meeting_repository(db_connection) -> MeetingRepository:
    """Create a meeting repository instance."""
    return MeetingRepository(db_connection)

def get_memory_repository(db_connection) -> MemoryRepository:
    """Create a memory repository instance."""
    return MemoryRepository(db_connection)

def get_memory_connection_repository(db_connection) -> MemoryConnectionRepository:
    """Create a memory connection repository instance."""
    return MemoryConnectionRepository(db_connection)

def get_stakeholder_repository(db_connection) -> StakeholderRepository:
    """Create a stakeholder repository instance."""
    return StakeholderRepository(db_connection)

def get_deliverable_repository(db_connection) -> DeliverableRepository:
    """Create a deliverable repository instance."""
    return DeliverableRepository(db_connection)
