"""
Basic API tests to verify the API is functional.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Mock the dependencies before importing the app
@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with patch('src.api.dependencies.get_db_connection') as mock_db, \
         patch('src.api.dependencies.get_vector_store_instance') as mock_vector, \
         patch('src.storage.sqlite.connection.DatabaseConnection') as mock_db_conn, \
         patch('src.storage.qdrant.vector_store.QdrantVectorStore') as mock_qdrant:
        
        # Mock database connection
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query = MagicMock(return_value=[])
        mock_db_conn.return_value = mock_db_instance
        mock_db.return_value = mock_db_instance
        
        # Mock vector store
        mock_vector_instance = MagicMock()
        mock_vector_instance.get_collection_stats = MagicMock(return_value={"vectors_count": 0})
        mock_qdrant.return_value = mock_vector_instance
        mock_vector.return_value = mock_vector_instance
        
        yield {
            'db': mock_db_instance,
            'vector_store': mock_vector_instance
        }


@pytest.fixture
def test_client(mock_dependencies):
    """Create test client with mocked dependencies."""
    from src.api.main import app
    return TestClient(app)


class TestBasicAPI:
    """Basic API functionality tests."""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Cognitive Meeting Intelligence API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert "endpoints" in data
        
    def test_health_endpoint(self, test_client, mock_dependencies):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        assert "vector_store" in data["components"]
        
    def test_openapi_endpoint(self, test_client):
        """Test OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Cognitive Meeting Intelligence API"
        
    def test_docs_endpoint(self, test_client):
        """Test that docs endpoint is available."""
        response = test_client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
        
    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set."""
        response = test_client.options("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        
    @pytest.mark.parametrize("endpoint", [
        "/api/v2/memories/search",
        "/api/v2/cognitive/query", 
        "/api/v2/discover-bridges"
    ])
    def test_api_endpoints_exist(self, test_client, endpoint):
        """Test that key API endpoints exist (even if they return errors without proper data)."""
        response = test_client.post(endpoint, json={})
        # We expect 422 (validation error) since we're not sending proper data
        # But this confirms the endpoint exists
        assert response.status_code in [422, 400, 500]  # Any of these indicate the endpoint exists