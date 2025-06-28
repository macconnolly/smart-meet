import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from src.models.entities import (
    Memory,
    Meeting,
    MemoryConnection,
    Project,
    Stakeholder,
    Deliverable,
    MeetingSeries,
    MemoryType,
    ContentType,
    ConnectionType,
    Priority,
    Status,
    ProjectType,
    ProjectStatus,
    MeetingType,
    MeetingCategory,
    DeliverableType,
    DeliverableStatus,
    StakeholderType,
    InfluenceLevel,
    EngagementLevel,
    MeetingFrequency,
)
from src.storage.sqlite.connection import DatabaseConnection


class BaseRepository:
    def __init__(self, db_connection: DatabaseConnection, table_name: str):
        self.db = db_connection
        self.table_name = table_name

    async def _execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        return await self.db.execute_query(query, params)

    async def _execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        return await self.db.execute_update(query, params)

    async def _fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        results = await self._execute_query(query, params)
        return results[0] if results else None

    async def get_by_id(self, id: str) -> Optional[Any]:
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        data = await self._fetch_one(query, (id,))
        if data:
            return self._from_dict(data)
        return None

    async def delete(self, id: str) -> bool:
        query = f"DELETE FROM {self.table_name} WHERE id = ?"
        rows_affected = await self._execute_update(query, (id,))
        return rows_affected > 0

    def _to_dict(self, entity: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def _from_dict(self, data: Dict[str, Any]) -> Any:
        raise NotImplementedError


class ProjectRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "projects")

    def _to_dict(self, project: Project) -> Dict[str, Any]:
        return project.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> Project:
        return Project.from_dict(data)

    async def create(self, project: Project) -> str:
        data = self._to_dict(project)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return project.id

    async def update(self, project: Project) -> bool:
        data = self._to_dict(project)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (project.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_all(self) -> List[Project]:
        query = f"SELECT * FROM {self.table_name}"
        results = await self._execute_query(query)
        return [self._from_dict(row) for row in results]


class StakeholderRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "stakeholders")

    def _to_dict(self, stakeholder: Stakeholder) -> Dict[str, Any]:
        return stakeholder.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> Stakeholder:
        return Stakeholder.from_dict(data)

    async def create(self, stakeholder: Stakeholder) -> str:
        data = self._to_dict(stakeholder)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return stakeholder.id

    async def update(self, stakeholder: Stakeholder) -> bool:
        data = self._to_dict(stakeholder)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (stakeholder.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_by_project(self, project_id: str) -> List[Stakeholder]:
        query = f"SELECT * FROM {self.table_name} WHERE project_id = ?"
        results = await self._execute_query(query, (project_id,))
        return [self._from_dict(row) for row in results]


class DeliverableRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "deliverables")

    def _to_dict(self, deliverable: Deliverable) -> Dict[str, Any]:
        return deliverable.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> Deliverable:
        return Deliverable.from_dict(data)

    async def create(self, deliverable: Deliverable) -> str:
        data = self._to_dict(deliverable)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return deliverable.id

    async def update(self, deliverable: Deliverable) -> bool:
        data = self._to_dict(deliverable)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (deliverable.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_by_project(self, project_id: str) -> List[Deliverable]:
        query = f"SELECT * FROM {self.table_name} WHERE project_id = ?"
        results = await self._execute_query(query, (project_id,))
        return [self._from_dict(row) for row in results]


class MeetingSeriesRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "meeting_series")

    def _to_dict(self, series: MeetingSeries) -> Dict[str, Any]:
        return series.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> MeetingSeries:
        return MeetingSeries.from_dict(data)

    async def create(self, series: MeetingSeries) -> str:
        data = self._to_dict(series)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return series.id

    async def update(self, series: MeetingSeries) -> bool:
        data = self._to_dict(series)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (series.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_by_project(self, project_id: str) -> List[MeetingSeries]:
        query = f"SELECT * FROM {self.table_name} WHERE project_id = ?"
        results = await self._execute_query(query, (project_id,))
        return [self._from_dict(row) for row in results]


class MeetingRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "meetings")

    def _to_dict(self, meeting: Meeting) -> Dict[str, Any]:
        return meeting.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> Meeting:
        return Meeting.from_dict(data)

    async def create(self, meeting: Meeting) -> str:
        data = self._to_dict(meeting)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return meeting.id

    async def update(self, meeting: Meeting) -> bool:
        data = self._to_dict(meeting)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (meeting.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_by_project(self, project_id: str) -> List[Meeting]:
        query = f"SELECT * FROM {self.table_name} WHERE project_id = ?"
        results = await self._execute_query(query, (project_id,))
        return [self._from_dict(row) for row in results]

    async def mark_processed(self, meeting_id: str, memory_count: int) -> bool:
        query = f"UPDATE {self.table_name} SET processed_at = ?, memory_count = ? WHERE id = ?"
        rows_affected = await self._execute_update(query, (datetime.now().isoformat(), memory_count, meeting_id))
        return rows_affected > 0


class MemoryRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "memories")

    def _to_dict(self, memory: Memory) -> Dict[str, Any]:
        return memory.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> Memory:
        return Memory.from_dict(data)

    async def create(self, memory: Memory) -> str:
        data = self._to_dict(memory)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        await self._execute_update(query, tuple(data.values()))
        return memory.id

    async def update(self, memory: Memory) -> bool:
        data = self._to_dict(memory)
        set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'id'])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(value for key, value in data.items() if key != 'id') + (memory.id,)
        rows_affected = await self._execute_update(query, params)
        return rows_affected > 0

    async def get_by_meeting(self, meeting_id: str) -> List[Memory]:
        query = f"SELECT * FROM {self.table_name} WHERE meeting_id = ? ORDER BY timestamp"
        results = await self._execute_query(query, (meeting_id,))
        return [self._from_dict(row) for row in results]

    async def get_by_project(self, project_id: str) -> List[Memory]:
        query = f"SELECT * FROM {self.table_name} WHERE project_id = ? ORDER BY created_at"
        results = await self._execute_query(query, (project_id,))
        return [self._from_dict(row) for row in results]


class ConnectionRepository(BaseRepository):
    def __init__(self, db_connection: DatabaseConnection):
        super().__init__(db_connection, "memory_connections")

    def _to_dict(self, connection: MemoryConnection) -> Dict[str, Any]:
        return connection.to_dict()

    def _from_dict(self, data: Dict[str, Any]) -> MemoryConnection:
        return MemoryConnection.from_dict(data)

    async def create(self, connection: MemoryConnection) -> bool:
        data = self._to_dict(connection)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        rows_affected = await self._execute_update(query, tuple(data.values()))
        return rows_affected > 0

    async def get_connections_for_memory(self, memory_id: str) -> List[MemoryConnection]:
        query = f"SELECT * FROM {self.table_name} WHERE source_id = ? OR target_id = ?"
        results = await self._execute_query(query, (memory_id, memory_id))
        return [self._from_dict(row) for row in results]

    async def delete_by_source_target(self, source_id: str, target_id: str) -> bool:
        query = f"DELETE FROM {self.table_name} WHERE source_id = ? AND target_id = ?"
        rows_affected = await self._execute_update(query, (source_id, target_id))
        return rows_affected > 0