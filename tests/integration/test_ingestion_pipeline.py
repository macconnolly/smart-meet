
import pytest
import asyncio
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path

from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.models.entities import Meeting, Memory, ContentType, MemoryType
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.sql_repositories import (
    MemoryRepository,
    MeetingRepository,
    ConnectionRepository,
)
from src.storage.qdrant.vector_store import QdrantVectorStore


# Mock Qdrant and Encoder for integration tests
@pytest.fixture(scope="session")
def mock_qdrant_client_for_ingestion(mocker):
    client = mocker.Mock()
    client.upsert.return_value = mocker.Mock(status="completed")
    client.get_collections.return_value = mocker.Mock(collections=[mocker.Mock(name="L2_cognitive_episodes")])
    return client

@pytest.fixture(scope="session")
def mock_encoder_for_ingestion(mocker):
    import numpy as np
    encoder = mocker.Mock()
    encoder.encode_batch.return_value = np.random.rand(10, 384).astype(np.float32)
    encoder.get_embedding_dimension.return_value = 384
    return encoder

@pytest.fixture(scope="session")
def mock_vector_manager_for_ingestion(mocker):
    manager = mocker.patch('src.embedding.vector_manager.VectorManager')
    manager.compose_vector.return_value = mocker.Mock(full_vector=mocker.Mock(tolist=lambda: [0.5]*400))
    return manager

@pytest_asyncio.fixture(scope="session")
async def ingestion_db_connection():
    db = DatabaseConnection(db_path=":memory:")
    await db.execute_schema()
    yield db

@pytest_asyncio.fixture(autouse=True)
async def clear_ingestion_db(ingestion_db_connection):
    tables_data = await ingestion_db_connection.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in tables_data if row["name"] != "sqlite_sequence"]
    for table in tables:
        await ingestion_db_connection.execute_update(f"DELETE FROM {table};")


@pytest.fixture
def ingestion_pipeline(ingestion_db_connection, mock_qdrant_client_for_ingestion, mock_encoder_for_ingestion, mock_vector_manager_for_ingestion):
    # Patch get_db and get_vector_store to return our mocked/test instances
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.storage.sqlite.connection.get_db", lambda: ingestion_db_connection)
        mp.setattr("src.storage.qdrant.vector_store.get_vector_store", lambda host=None, port=None, api_key=None: QdrantVectorStore(client=mock_qdrant_client_for_ingestion))
        mp.setattr("src.embedding.onnx_encoder.get_encoder", lambda config=None: mock_encoder_for_ingestion)
        mp.setattr("src.embedding.vector_manager.get_vector_manager", lambda: mock_vector_manager_for_ingestion)

        pipeline = IngestionPipeline()
        yield pipeline


@pytest.fixture
def sample_transcript_file(tmp_path):
    transcript_content = """
Alice: Hello everyone. We need to decide on the new project timeline.
Bob: I think we should aim for a 3-month delivery.
Alice: That sounds like a good idea. Let's make it an action item to finalize the dates.
[00:01:30] Charlie: I have a question about resource allocation.
"""
    file_path = tmp_path / "sample_transcript.txt"
    file_path.write_text(transcript_content)
    return file_path


@pytest.mark.integration
@pytest.mark.asyncio
class TestIngestionPipeline:
    async def test_ingest_meeting_full_pipeline(self, ingestion_pipeline, sample_transcript_file, ingestion_db_connection):
        meeting = Meeting(
            id="test-meeting-123",
            project_id="test-project-abc",
            title="Project Timeline Discussion",
            transcript_path=str(sample_transcript_file),
            start_time=datetime.now()
        )

        stats = await ingestion_pipeline.ingest_meeting(meeting, str(sample_transcript_file))

        assert stats["memories_extracted"] > 0
        assert stats["vectors_stored"] == stats["memories_extracted"]
        assert stats["connections_created"] > 0
        assert not stats["errors"]

        # Verify data in SQLite
        memory_repo = MemoryRepository(ingestion_db_connection)
        memories_in_db = await memory_repo.get_by_meeting(meeting.id)
        assert len(memories_in_db) == stats["memories_extracted"]

        meeting_repo = MeetingRepository(ingestion_db_connection)
        retrieved_meeting = await meeting_repo.get_by_id(meeting.id)
        assert retrieved_meeting.processed_at is not None
        assert retrieved_meeting.memory_count == stats["memories_extracted"]

        connection_repo = ConnectionRepository(ingestion_db_connection)
        connections_in_db = await connection_repo.get_connections_for_memory(memories_in_db[0].id)
        assert len(connections_in_db) == stats["connections_created"]

        # Verify Qdrant interactions (mocked)
        ingestion_pipeline.vector_store.client.upsert.assert_called()
        # Verify that dimensions are stored in Qdrant payload
        upsert_calls = ingestion_pipeline.vector_store.client.upsert.call_args_list
        assert len(upsert_calls) > 0
        # Check the payload of the first upserted point
        first_call_points = upsert_calls[0].kwargs['points']
        assert len(first_call_points) > 0
        first_point_payload = first_call_points[0].payload
        assert 'dim_temporal_urgency' in first_point_payload
        assert 'dim_emotional_polarity' in first_point_payload
        assert 'dim_social_authority' in first_point_payload
        assert 'dim_causal_impact' in first_point_payload
        assert 'dim_evolutionary_change_rate' in first_point_payload

    async def test_ingest_meeting_non_existent_transcript(self, ingestion_pipeline, ingestion_db_connection):
        meeting = Meeting(
            id="test-meeting-456",
            project_id="test-project-def",
            title="Non Existent Transcript",
            transcript_path="/path/to/non_existent_file.txt",
            start_time=datetime.now()
        )

        stats = await ingestion_pipeline.ingest_meeting(meeting, meeting.transcript_path)

        assert stats["memories_extracted"] == 0
        assert stats["vectors_stored"] == 0
        assert stats["connections_created"] == 0
        assert any("FileNotFoundError" in error for error in stats["errors"])

        # Verify no data in SQLite
        memory_repo = MemoryRepository(ingestion_db_connection)
        memories_in_db = await memory_repo.get_by_meeting(meeting.id)
        assert len(memories_in_db) == 0

        meeting_repo = MeetingRepository(ingestion_db_connection)
        retrieved_meeting = await meeting_repo.get_by_id(meeting.id)
        assert retrieved_meeting.processed_at is None

