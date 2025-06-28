
import pytest
import asyncio
from datetime import datetime, timedelta
import json
import uuid

from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.sql_repositories import (
    ProjectRepository,
    StakeholderRepository,
    DeliverableRepository,
    MeetingSeriesRepository,
    MeetingRepository,
    MemoryRepository,
    ConnectionRepository,
)
from src.models.entities import (
    Project,
    Stakeholder,
    Deliverable,
    MeetingSeries,
    Meeting,
    Memory,
    MemoryConnection,
    ProjectType,
    ProjectStatus,
    StakeholderType,
    DeliverableType,
    MeetingType,
    MeetingCategory,
    MemoryType,
    ContentType,
    ConnectionType,
    Priority,
    Status,
)


db_instance = None


@pytest_asyncio.fixture(scope="session")
async def db_connection():
    global db_instance
    if db_instance is None:
        db_instance = DatabaseConnection(db_path=":memory:")
        await db_instance.execute_schema()
    yield db_instance


@pytest_asyncio.fixture(autouse=True)
async def clear_db(db_connection):
    # Clear all tables before each test
    tables_data = await db_connection.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in tables_data if row["name"] != "sqlite_sequence"]
    for table in tables:
        await db_connection.execute_update(f"DELETE FROM {table};")


@pytest.fixture
def project_repo(db_connection):
    return ProjectRepository(db_connection)


@pytest.fixture
def stakeholder_repo(db_connection):
    return StakeholderRepository(db_connection)


@pytest.fixture
def deliverable_repo(db_connection):
    return DeliverableRepository(db_connection)


@pytest.fixture
def meeting_series_repo(db_connection):
    return MeetingSeriesRepository(db_connection)


@pytest.fixture
def meeting_repo(db_connection):
    return MeetingRepository(db_connection)


@pytest.fixture
def memory_repo(db_connection):
    return MemoryRepository(db_connection)


@pytest.fixture
def connection_repo(db_connection):
    return ConnectionRepository(db_connection)


class TestProjectRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_project(self, project_repo):
        project = Project(name="Test Project", client_name="Test Client")
        project_id = await project_repo.create(project)
        assert project_id == project.id

        retrieved_project = await project_repo.get_by_id(project_id)
        assert retrieved_project.name == "Test Project"
        assert retrieved_project.client_name == "Test Client"

    @pytest.mark.asyncio
    async def test_update_project(self, project_repo):
        project = Project(name="Old Name", client_name="Old Client")
        await project_repo.create(project)

        project.name = "New Name"
        updated = await project_repo.update(project)
        assert updated is True

        retrieved_project = await project_repo.get_by_id(project.id)
        assert retrieved_project.name == "New Name"

    @pytest.mark.asyncio
    async def test_delete_project(self, project_repo):
        project = Project(name="To Delete", client_name="Client")
        await project_repo.create(project)

        deleted = await project_repo.delete(project.id)
        assert deleted is True

        retrieved_project = await project_repo.get_by_id(project.id)
        assert retrieved_project is None

    @pytest.mark.asyncio
    async def test_get_all_projects(self, project_repo):
        project1 = Project(name="P1", client_name="C1")
        project2 = Project(name="P2", client_name="C2")
        await project_repo.create(project1)
        await project_repo.create(project2)

        all_projects = await project_repo.get_all()
        assert len(all_projects) == 2
        assert any(p.name == "P1" for p in all_projects)
        assert any(p.name == "P2" for p in all_projects)


class TestStakeholderRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_stakeholder(self, project_repo, stakeholder_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)

        stakeholder = Stakeholder(project_id=project.id, name="John Doe", organization="Org", role="Role")
        stakeholder_id = await stakeholder_repo.create(stakeholder)
        assert stakeholder_id == stakeholder.id

        retrieved_stakeholder = await stakeholder_repo.get_by_id(stakeholder_id)
        assert retrieved_stakeholder.name == "John Doe"

    @pytest.mark.asyncio
    async def test_get_by_project(self, project_repo, stakeholder_repo):
        project1 = Project(name="P1", client_name="C1")
        project2 = Project(name="P2", client_name="C2")
        await project_repo.create(project1)
        await project_repo.create(project2)

        s1 = Stakeholder(project_id=project1.id, name="S1", organization="O1", role="R1")
        s2 = Stakeholder(project_id=project1.id, name="S2", organization="O2", role="R2")
        s3 = Stakeholder(project_id=project2.id, name="S3", organization="O3", role="R3")
        await stakeholder_repo.create(s1)
        await stakeholder_repo.create(s2)
        await stakeholder_repo.create(s3)

        project1_stakeholders = await stakeholder_repo.get_by_project(project1.id)
        assert len(project1_stakeholders) == 2
        assert any(s.name == "S1" for s in project1_stakeholders)


class TestDeliverableRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_deliverable(self, project_repo, deliverable_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)

        deliverable = Deliverable(project_id=project.id, name="Report", owner="Me")
        deliverable_id = await deliverable_repo.create(deliverable)
        assert deliverable_id == deliverable.id

        retrieved_deliverable = await deliverable_repo.get_by_id(deliverable_id)
        assert retrieved_deliverable.name == "Report"

    @pytest.mark.asyncio
    async def test_get_by_project(self, project_repo, deliverable_repo):
        project1 = Project(name="P1", client_name="C1")
        project2 = Project(name="P2", client_name="C2")
        await project_repo.create(project1)
        await project_repo.create(project2)

        d1 = Deliverable(project_id=project1.id, name="D1", owner="O1")
        d2 = Deliverable(project_id=project1.id, name="D2", owner="O2")
        d3 = Deliverable(project_id=project2.id, name="D3", owner="O3")
        await deliverable_repo.create(d1)
        await deliverable_repo.create(d2)
        await deliverable_repo.create(d3)

        project1_deliverables = await deliverable_repo.get_by_project(project1.id)
        assert len(project1_deliverables) == 2
        assert any(d.name == "D1" for d in project1_deliverables)


class TestMeetingSeriesRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_meeting_series(self, project_repo, meeting_series_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)

        series = MeetingSeries(project_id=project.id, series_name="Weekly Sync")
        series_id = await meeting_series_repo.create(series)
        assert series_id == series.id

        retrieved_series = await meeting_series_repo.get_by_id(series_id)
        assert retrieved_series.series_name == "Weekly Sync"

    @pytest.mark.asyncio
    async def test_get_by_project(self, project_repo, meeting_series_repo):
        project1 = Project(name="P1", client_name="C1")
        project2 = Project(name="P2", client_name="C2")
        await project_repo.create(project1)
        await project_repo.create(project2)

        ms1 = MeetingSeries(project_id=project1.id, series_name="MS1")
        ms2 = MeetingSeries(project_id=project1.id, series_name="MS2")
        ms3 = MeetingSeries(project_id=project2.id, series_name="MS3")
        await meeting_series_repo.create(ms1)
        await meeting_series_repo.create(ms2)
        await meeting_series_repo.create(ms3)

        project1_series = await meeting_series_repo.get_by_project(project1.id)
        assert len(project1_series) == 2
        assert any(ms.series_name == "MS1" for ms in project1_series)


class TestMeetingRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_meeting(self, project_repo, meeting_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)

        meeting = Meeting(project_id=project.id, title="Daily Standup")
        meeting_id = await meeting_repo.create(meeting)
        assert meeting_id == meeting.id

        retrieved_meeting = await meeting_repo.get_by_id(meeting_id)
        assert retrieved_meeting.title == "Daily Standup"

    @pytest.mark.asyncio
    async def test_mark_processed(self, project_repo, meeting_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)

        meeting = Meeting(project_id=project.id, title="Unprocessed Meeting")
        await meeting_repo.create(meeting)

        assert meeting.processed_at is None
        assert meeting.memory_count == 0

        updated = await meeting_repo.mark_processed(meeting.id, 10)
        assert updated is True

        retrieved_meeting = await meeting_repo.get_by_id(meeting.id)
        assert retrieved_meeting.processed_at is not None
        assert retrieved_meeting.memory_count == 10


class TestMemoryRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_memory(self, project_repo, meeting_repo, memory_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)
        meeting = Meeting(project_id=project.id, title="Test Meeting")
        await meeting_repo.create(meeting)

        memory = Memory(
            meeting_id=meeting.id,
            project_id=project.id,
            content="This is a test memory.",
            timestamp_ms=1000,
            qdrant_id=str(uuid.uuid4()),
            dimensions_json=json.dumps([0.5]*16)
        )
        memory_id = await memory_repo.create(memory)
        assert memory_id == memory.id

        retrieved_memory = await memory_repo.get_by_id(memory_id)
        assert retrieved_memory.content == "This is a test memory."
        assert retrieved_memory.timestamp_ms == 1000

    @pytest.mark.asyncio
    async def test_get_by_meeting(self, project_repo, meeting_repo, memory_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)
        meeting1 = Meeting(project_id=project.id, title="Meeting 1")
        meeting2 = Meeting(project_id=project.id, title="Meeting 2")
        await meeting_repo.create(meeting1)
        await meeting_repo.create(meeting2)

        mem1 = Memory(meeting_id=meeting1.id, project_id=project.id, content="M1-1", timestamp_ms=100, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        mem2 = Memory(meeting_id=meeting1.id, project_id=project.id, content="M1-2", timestamp_ms=200, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        mem3 = Memory(meeting_id=meeting2.id, project_id=project.id, content="M2-1", timestamp_ms=300, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        await memory_repo.create(mem1)
        await memory_repo.create(mem2)
        await memory_repo.create(mem3)

        meeting1_memories = await memory_repo.get_by_meeting(meeting1.id)
        assert len(meeting1_memories) == 2
        assert meeting1_memories[0].content == "M1-1"
        assert meeting1_memories[1].content == "M1-2"


class TestConnectionRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_connection(self, project_repo, meeting_repo, memory_repo, connection_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)
        meeting = Meeting(project_id=project.id, title="Test Meeting")
        await meeting_repo.create(meeting)

        mem1 = Memory(meeting_id=meeting.id, project_id=project.id, content="Mem1", timestamp_ms=100, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        mem2 = Memory(meeting_id=meeting.id, project_id=project.id, content="Mem2", timestamp_ms=200, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        await memory_repo.create(mem1)
        await memory_repo.create(mem2)

        connection = MemoryConnection(source_id=mem1.id, target_id=mem2.id, connection_type=ConnectionType.SEQUENTIAL)
        created = await connection_repo.create(connection)
        assert created is True

        retrieved_connections = await connection_repo.get_connections_for_memory(mem1.id)
        assert len(retrieved_connections) == 1
        assert retrieved_connections[0].target_id == mem2.id

    @pytest.mark.asyncio
    async def test_delete_connection(self, project_repo, meeting_repo, memory_repo, connection_repo):
        project = Project(name="Test Project", client_name="Test Client")
        await project_repo.create(project)
        meeting = Meeting(project_id=project.id, title="Test Meeting")
        await meeting_repo.create(meeting)

        mem1 = Memory(meeting_id=meeting.id, project_id=project.id, content="Mem1", timestamp_ms=100, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        mem2 = Memory(meeting_id=meeting.id, project_id=project.id, content="Mem2", timestamp_ms=200, qdrant_id=str(uuid.uuid4()), dimensions_json=json.dumps([0.5]*16))
        await memory_repo.create(mem1)
        await memory_repo.create(mem2)

        connection = MemoryConnection(source_id=mem1.id, target_id=mem2.id, connection_type=ConnectionType.SEQUENTIAL)
        await connection_repo.create(connection)

        deleted = await connection_repo.delete_by_source_target(mem1.id, mem2.id)
        assert deleted is True

        retrieved_connections = await connection_repo.get_connections_for_memory(mem1.id)
        assert len(retrieved_connections) == 0
