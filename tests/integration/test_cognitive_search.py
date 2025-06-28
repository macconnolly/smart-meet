import pytest
import asyncio
from datetime import datetime
import json
import numpy as np

from src.api.routers.cognitive import CognitiveQueryRequest, cognitive_query
from src.models.entities import Memory, MemoryType, ContentType
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.repositories import MemoryRepository, MemoryConnectionRepository
from src.storage.qdrant.vector_store import QdrantVectorStore
from src.embedding.onnx_encoder import ONNXEncoder
from src.embedding.vector_manager import VectorManager
from src.extraction.dimensions.dimension_analyzer import CognitiveDimensions

# Mock dependencies for integration tests
@pytest.fixture(scope="session")
def mock_qdrant_client_for_cognitive_search(mocker):
    client = mocker.Mock()
    # Mock search method to return predefined results
    async def mock_search_all_levels(*args, **kwargs):
        # Simulate some search results with payloads containing dimensions
        mock_payload_1 = {
            "memory_id": "mem-1", "project_id": "proj-1", "meeting_id": "meet-1",
            "memory_type": "episodic", "content_type": "decision", "importance_score": 0.8,
            "created_at": int(datetime.now().timestamp()),
            "dim_temporal_urgency": 0.9, "dim_causal_impact": 0.8
        }
        mock_payload_2 = {
            "memory_id": "mem-2", "project_id": "proj-1", "meeting_id": "meet-1",
            "memory_type": "episodic", "content_type": "context", "importance_score": 0.6,
            "created_at": int(datetime.now().timestamp()),
            "dim_temporal_urgency": 0.2, "dim_causal_impact": 0.3
        }
        mock_vector_1 = mocker.Mock(full_vector=np.random.rand(400).tolist())
        mock_vector_2 = mocker.Mock(full_vector=np.random.rand(400).tolist())

        mock_result_1 = mocker.Mock(payload=mock_payload_1, score=0.9, vector=mock_vector_1.full_vector)
        mock_result_2 = mocker.Mock(payload=mock_payload_2, score=0.7, vector=mock_vector_2.full_vector)

        return {0: [mock_result_1], 1: [], 2: [mock_result_2]} # Simulate results from L0 and L2

    client.search_all_levels = mock_search_all_levels
    return client

@pytest.fixture(scope="session")
def mock_encoder_for_cognitive_search(mocker):
    encoder = mocker.Mock(spec=ONNXEncoder)
    encoder.encode.return_value = np.random.rand(384).astype(np.float32)
    return encoder

@pytest.fixture(scope="session")
def mock_vector_manager_for_cognitive_search(mocker):
    manager = mocker.Mock(spec=VectorManager)
    manager.compose_vector.return_value = np.random.rand(400).astype(np.float32) # Returns numpy array
    return manager

@pytest.fixture(scope="session")
def mock_dimension_analyzer_for_cognitive_search(mocker):
    analyzer = mocker.Mock()
    # Mock analyze to return a CognitiveDimensions object
    analyzer.analyze.return_value = CognitiveDimensions(
        temporal=mocker.Mock(urgency=0.8, deadline_proximity=0.5, sequence_position=0.5, duration_relevance=0.5),
        emotional=mocker.Mock(polarity=0.5, intensity=0.5, confidence=0.5),
        social=mocker.Mock(authority=0.5, influence=0.5, team_dynamics=0.5),
        causal=mocker.Mock(dependencies=0.5, impact=0.7, risk_factors=0.5),
        strategic=mocker.Mock(alignment=0.5, innovation=0.5, value=0.5)
    )
    return analyzer

@pytest_asyncio.fixture(scope="session")
async def cognitive_search_db_connection():
    db = DatabaseConnection(db_path=":memory:")
    await db.execute_schema()
    yield db

@pytest_asyncio.fixture(autouse=True)
async def clear_cognitive_search_db(cognitive_search_db_connection):
    tables_data = await cognitive_search_db_connection.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in tables_data if row["name"] != "sqlite_sequence"]
    for table in tables:
        await cognitive_search_db_connection.execute_update(f"DELETE FROM {table};")

@pytest.fixture
def cognitive_search_dependencies(cognitive_search_db_connection, mock_qdrant_client_for_cognitive_search, mock_encoder_for_cognitive_search, mock_vector_manager_for_cognitive_search, mock_dimension_analyzer_for_cognitive_search):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.dependencies.get_db_connection", lambda: cognitive_search_db_connection)
        mp.setattr("src.api.dependencies.get_vector_store_instance", lambda: mock_qdrant_client_for_cognitive_search)
        mp.setattr("src.embedding.onnx_encoder.get_encoder", lambda: mock_encoder_for_cognitive_search)
        mp.setattr("src.embedding.vector_manager.get_vector_manager", lambda: mock_vector_manager_for_cognitive_search)
        mp.setattr("src.extraction.dimensions.dimension_analyzer.get_dimension_analyzer", lambda: mock_dimension_analyzer_for_cognitive_search)
        yield

@pytest.mark.integration
@pytest.mark.asyncio
class TestCognitiveSearch:
    async def test_cognitive_query_endpoint(self, cognitive_search_dependencies, cognitive_search_db_connection):
        # Setup: Add some memories to the mocked DB
        memory_repo = MemoryRepository(cognitive_search_db_connection)
        mem1 = Memory(
            id="mem-1", meeting_id="meet-1", project_id="proj-1", content="This is a critical decision about project timeline.",
            memory_type=MemoryType.EPISODIC, content_type=ContentType.DECISION, created_at=datetime.now()
        )
        mem2 = Memory(
            id="mem-2", meeting_id="meet-1", project_id="proj-1", content="Discussion about resource allocation.",
            memory_type=MemoryType.EPISODIC, content_type=ContentType.CONTEXT, created_at=datetime.now()
        )
        await memory_repo.create(mem1)
        await memory_repo.create(mem2)

        # Create a request
        request = CognitiveQueryRequest(
            query="urgent project decisions",
            project_id="proj-1",
            max_activations=10,
            activation_threshold=0.5,
            core_threshold=0.7,
            peripheral_threshold=0.4,
            decay_factor=0.8
        )

        # Call the endpoint
        response = await cognitive_query(request, Depends(lambda: cognitive_search_db_connection), Depends(lambda: mock_qdrant_client_for_cognitive_search))

        # Assertions
        assert response.query == request.query
        assert response.status == "success"
        assert response.total_activated_memories > 0
        assert len(response.core_memories) >= 1 # Expect at least one core memory due to high urgency
        assert any(m.memory_id == "mem-1" for m in response.core_memories) # mem-1 should be core
        assert any(m.memory_id == "mem-2" for m in response.peripheral_memories) # mem-2 should be peripheral

        # Verify explanations contain cognitive dimension details
        core_mem_explanation = next(m.explanation for m in response.core_memories if m.memory_id == "mem-1")
        assert "highly urgent memory" in core_mem_explanation
        assert "significant impact" in core_mem_explanation

        # Verify Qdrant search was called with correct parameters
        mock_qdrant_client_for_cognitive_search.search_all_levels.assert_called_once()
        call_args, call_kwargs = mock_qdrant_client_for_cognitive_search.search_all_levels.call_args
        assert isinstance(call_kwargs['query_vector'], np.ndarray) # Should be numpy array from vector_manager.compose_vector
        assert call_kwargs['filters'].project_id == "proj-1"

    async def test_cognitive_query_no_results(self, cognitive_search_dependencies, cognitive_search_db_connection, mock_qdrant_client_for_cognitive_search):
        # Mock search to return no results
        async def mock_empty_search(*args, **kwargs):
            return {0: [], 1: [], 2: []}
        mock_qdrant_client_for_cognitive_search.search_all_levels = mock_empty_search

        request = CognitiveQueryRequest(
            query="non-existent query",
            project_id="proj-non-existent",
            max_activations=10,
            activation_threshold=0.5,
            core_threshold=0.7,
            peripheral_threshold=0.4,
            decay_factor=0.8
        )

        response = await cognitive_query(request, Depends(lambda: cognitive_search_db_connection), Depends(lambda: mock_qdrant_client_for_cognitive_search))

        assert response.query == request.query
        assert response.status == "success"
        assert response.total_activated_memories == 0
        assert len(response.core_memories) == 0
        assert len(response.peripheral_memories) == 0

