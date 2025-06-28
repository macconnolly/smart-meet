import pytest
import asyncio
from datetime import datetime
import json
import numpy as np

from src.api.routers.bridges import BridgeDiscoveryRequest, discover_bridges
from src.models.entities import Memory, MemoryType, ContentType
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.repositories import MemoryRepository, MemoryConnectionRepository
from src.storage.qdrant.vector_store import QdrantVectorStore
from src.embedding.onnx_encoder import ONNXEncoder
from src.embedding.vector_manager import VectorManager
from src.extraction.dimensions.dimension_analyzer import CognitiveDimensions

# Mock dependencies for integration tests
@pytest.fixture(scope="session")
def mock_qdrant_client_for_bridge_discovery(mocker):
    client = mocker.Mock()
    # Mock search_all_levels to return predefined results
    async def mock_search_all_levels(*args, **kwargs):
        # Simulate some search results with payloads containing dimensions
        # Memory 1: Highly relevant to query, but also has high novelty potential
        mock_payload_1 = {
            "memory_id": "mem-1", "project_id": "proj-1", "meeting_id": "meet-1",
            "memory_type": "episodic", "content_type": "insight", "importance_score": 0.9,
            "created_at": int(datetime.now().timestamp()),
            "dim_temporal_urgency": 0.1, "dim_causal_impact": 0.9, "dim_strategic_innovation": 0.8
        }
        # Memory 2: Less relevant to query, but strongly connected to mem-1 (simulated)
        mock_payload_2 = {
            "memory_id": "mem-2", "project_id": "proj-1", "meeting_id": "meet-2",
            "memory_type": "episodic", "content_type": "decision", "importance_score": 0.7,
            "created_at": int(datetime.now().timestamp()),
            "dim_temporal_urgency": 0.8, "dim_causal_impact": 0.2, "dim_strategic_innovation": 0.1
        }
        mock_vector_1 = mocker.Mock(full_vector=np.random.rand(400).tolist())
        mock_vector_2 = mocker.Mock(full_vector=np.random.rand(400).tolist())

        mock_result_1 = mocker.Mock(payload=mock_payload_1, score=0.85, vector=mock_vector_1.full_vector)
        mock_result_2 = mocker.Mock(payload=mock_payload_2, score=0.6, vector=mock_vector_2.full_vector)

        return {0: [mock_result_1], 1: [], 2: [mock_result_2]} # Simulate results from L0 and L2

    client.search_all_levels = mock_search_all_levels

    # Mock search for SimpleBridgeDiscovery.find_candidate_bridges
    async def mock_search_candidates(*args, **kwargs):
        # Simulate a candidate bridge memory (e.g., mem-3)
        mock_payload_3 = {
            "memory_id": "mem-3", "project_id": "proj-2", "meeting_id": "meet-3",
            "memory_type": "semantic", "content_type": "finding", "importance_score": 0.95,
            "created_at": int(datetime.now().timestamp()),
            "dim_temporal_urgency": 0.1, "dim_causal_impact": 0.1, "dim_strategic_innovation": 0.9
        }
        mock_vector_3 = mocker.Mock(full_vector=np.random.rand(400).tolist())
        mock_result_3 = mocker.Mock(payload=mock_payload_3, score=0.1, vector=mock_vector_3.full_vector) # Low semantic similarity to query
        return [mock_result_3]

    client.search = mock_search_candidates
    return client

@pytest.fixture(scope="session")
def mock_encoder_for_bridge_discovery(mocker):
    encoder = mocker.Mock(spec=ONNXEncoder)
    encoder.encode.return_value = np.random.rand(384).astype(np.float32)
    return encoder

@pytest.fixture(scope="session")
def mock_vector_manager_for_bridge_discovery(mocker):
    manager = mocker.Mock(spec=VectorManager)
    manager.compose_vector.return_value = np.random.rand(400).astype(np.float32) # Returns numpy array
    return manager

@pytest.fixture(scope="session")
def mock_dimension_analyzer_for_bridge_discovery(mocker):
    analyzer = mocker.Mock()
    # Mock analyze to return a CognitiveDimensions object
    analyzer.analyze.return_value = CognitiveDimensions(
        temporal=mocker.Mock(urgency=0.1, deadline_proximity=0.1, sequence_position=0.1, duration_relevance=0.1),
        emotional=mocker.Mock(polarity=0.5, intensity=0.5, confidence=0.5),
        social=mocker.Mock(authority=0.5, influence=0.5, team_dynamics=0.5),
        causal=mocker.Mock(dependencies=0.1, impact=0.1, risk_factors=0.1),
        strategic=mocker.Mock(alignment=0.9, innovation=0.9, value=0.9)
    )
    return analyzer

@pytest_asyncio.fixture(scope="session")
async def bridge_discovery_db_connection():
    db = DatabaseConnection(db_path=":memory:")
    await db.execute_schema()
    yield db

@pytest_asyncio.fixture(autouse=True)
async def clear_bridge_discovery_db(bridge_discovery_db_connection):
    tables_data = await bridge_discovery_db_connection.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in tables_data if row["name"] != "sqlite_sequence"]
    for table in tables:
        await bridge_discovery_db_connection.execute_update(f"DELETE FROM {table};")

@pytest.fixture
def bridge_discovery_dependencies(bridge_discovery_db_connection, mock_qdrant_client_for_bridge_discovery, mock_encoder_for_bridge_discovery, mock_vector_manager_for_bridge_discovery, mock_dimension_analyzer_for_bridge_discovery):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.dependencies.get_db_connection", lambda: bridge_discovery_db_connection)
        mp.setattr("src.api.dependencies.get_vector_store_instance", lambda: mock_qdrant_client_for_bridge_discovery)
        mp.setattr("src.embedding.onnx_encoder.get_encoder", lambda: mock_encoder_for_bridge_discovery)
        mp.setattr("src.embedding.vector_manager.get_vector_manager", lambda: mock_vector_manager_for_bridge_discovery)
        mp.setattr("src.extraction.dimensions.dimension_analyzer.get_dimension_analyzer", lambda: mock_dimension_analyzer_for_bridge_discovery)
        yield

@pytest.mark.integration
@pytest.mark.asyncio
class TestBridgeDiscovery:
    async def test_discover_bridges_endpoint(self, bridge_discovery_dependencies, bridge_discovery_db_connection):
        # Setup: Add some memories to the mocked DB
        memory_repo = MemoryRepository(bridge_discovery_db_connection)
        mem1 = Memory(
            id="mem-1", meeting_id="meet-1", project_id="proj-1", content="Insight on new marketing strategy.",
            memory_type=MemoryType.EPISODIC, content_type=ContentType.INSIGHT, created_at=datetime.now(),
            dimensions_json=json.dumps({
                "temporal": {"urgency": 0.1, "deadline_proximity": 0.1, "sequence_position": 0.1, "duration_relevance": 0.1},
                "emotional": {"polarity": 0.5, "intensity": 0.5, "confidence": 0.5},
                "social": {"authority": 0.5, "influence": 0.5, "team_dynamics": 0.5},
                "causal": {"dependencies": 0.1, "impact": 0.9, "risk_factors": 0.1},
                "strategic": {"alignment": 0.8, "innovation": 0.8, "value": 0.8}
            })
        )
        mem2 = Memory(
            id="mem-2", meeting_id="meet-2", project_id="proj-1", content="Decision to re-architect backend.",
            memory_type=MemoryType.EPISODIC, content_type=ContentType.DECISION, created_at=datetime.now(),
            dimensions_json=json.dumps({
                "temporal": {"urgency": 0.8, "deadline_proximity": 0.8, "sequence_position": 0.8, "duration_relevance": 0.8},
                "emotional": {"polarity": 0.5, "intensity": 0.5, "confidence": 0.5},
                "social": {"authority": 0.5, "influence": 0.5, "team_dynamics": 0.5},
                "causal": {"dependencies": 0.9, "impact": 0.1, "risk_factors": 0.1},
                "strategic": {"alignment": 0.1, "innovation": 0.1, "value": 0.1}
            })
        )
        mem3 = Memory(
            id="mem-3", meeting_id="meet-3", project_id="proj-2", content="New finding about customer behavior.",
            memory_type=MemoryType.SEMANTIC, content_type=ContentType.FINDING, created_at=datetime.now(),
            dimensions_json=json.dumps({
                "temporal": {"urgency": 0.1, "deadline_proximity": 0.1, "sequence_position": 0.1, "duration_relevance": 0.1},
                "emotional": {"polarity": 0.5, "intensity": 0.5, "confidence": 0.5},
                "social": {"authority": 0.5, "influence": 0.5, "team_dynamics": 0.5},
                "causal": {"dependencies": 0.1, "impact": 0.1, "risk_factors": 0.1},
                "strategic": {"alignment": 0.9, "innovation": 0.9, "value": 0.9}
            })
        )
        await memory_repo.create(mem1)
        await memory_repo.create(mem2)
        await memory_repo.create(mem3)

        # Create a request
        request = BridgeDiscoveryRequest(
            query="How can we innovate our marketing to impact customer behavior?",
            project_id="proj-1", # Querying within proj-1, but mem-3 is in proj-2
            max_bridges=1,
            novelty_weight=0.6,
            connection_weight=0.4,
            min_bridge_score=0.5,
            search_expansion=100
        )

        # Call the endpoint
        response = await discover_bridges(request, Depends(lambda: bridge_discovery_db_connection), Depends(lambda: mock_qdrant_client_for_bridge_discovery))

        # Assertions
        assert response.query == request.query
        assert response.status == "success"
        assert len(response.discovered_bridges) == 1

        bridge = response.discovered_bridges[0]
        assert bridge.memory_id == "mem-3" # mem-3 should be the discovered bridge
        assert bridge.novelty_score > 0.5 # Should be novel due to different project_id
        assert bridge.connection_potential > 0.5 # Should be connected due to high strategic innovation
        assert bridge.bridge_score > 0.5
        assert "unexpected connection" in bridge.explanation

        # Verify Qdrant search was called with correct parameters
        mock_qdrant_client_for_bridge_discovery.search_all_levels.assert_called_once()
        mock_qdrant_client_for_bridge_discovery.search.assert_called_once()

    async def test_discover_bridges_no_results(self, bridge_discovery_dependencies, bridge_discovery_db_connection, mock_qdrant_client_for_bridge_discovery):
        # Mock search to return no results
        async def mock_empty_search(*args, **kwargs):
            return {0: [], 1: [], 2: []}
        mock_qdrant_client_for_bridge_discovery.search_all_levels = mock_empty_search
        mock_qdrant_client_for_bridge_discovery.search.return_value = []

        request = BridgeDiscoveryRequest(
            query="non-existent query",
            project_id="proj-non-existent",
            max_bridges=1,
            novelty_weight=0.6,
            connection_weight=0.4,
            min_bridge_score=0.5,
            search_expansion=100
        )

        response = await discover_bridges(request, Depends(lambda: bridge_discovery_db_connection), Depends(lambda: mock_qdrant_client_for_bridge_discovery))

        assert response.query == request.query
        assert response.status == "success"
        assert len(response.discovered_bridges) == 0
