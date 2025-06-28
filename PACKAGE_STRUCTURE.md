# Package Structure for Component Separation

## Current Monolithic Structure
```
src/
├── __init__.py          # Package metadata
├── api/                 # API Layer
│   ├── __init__.py
│   ├── main.py         # FastAPI app
│   ├── dependencies.py # DI setup
│   └── routers/        # API endpoints
├── cognitive/          # ML Algorithms
│   ├── activation/     # Spreading activation
│   ├── bridges/        # Bridge discovery
│   └── consolidation/  # Memory consolidation
├── embedding/          # Vector operations
│   ├── engine.py       # ONNX encoder
│   └── vector_manager.py
├── extraction/         # Feature extraction
│   └── dimensions/     # 16D extractors
├── models/            # Data models
├── pipeline/          # Processing pipelines
├── storage/           # Storage backends
│   ├── sqlite/        # Metadata storage
│   └── qdrant/        # Vector storage
└── utils/             # Shared utilities
```

## Target Microservices Structure

### 1. cognitive-meeting-api/
```
cognitive-meeting-api/
├── pyproject.toml
├── src/
│   └── cognitive_meeting_api/
│       ├── __init__.py
│       ├── main.py           # FastAPI app
│       ├── dependencies.py   # DI setup
│       ├── routers/         # API endpoints
│       ├── services/        # Business logic
│       ├── clients/         # ML service clients
│       └── models/          # API models
└── tests/
```

### 2. cognitive-meeting-ml/
```
cognitive-meeting-ml/
├── pyproject.toml
├── src/
│   └── cognitive_meeting_ml/
│       ├── __init__.py
│       ├── server.py         # gRPC/REST server
│       ├── cognitive/        # Algorithms
│       ├── embedding/        # Embeddings
│       ├── extraction/       # Extractors
│       ├── pipeline/         # Pipelines
│       └── storage/          # Storage
└── tests/
```

### 3. cognitive-meeting-core/
```
cognitive-meeting-core/
├── pyproject.toml
├── src/
│   └── cognitive_meeting_core/
│       ├── __init__.py
│       ├── models/          # Shared data models
│       ├── interfaces/      # Service interfaces
│       └── utils/           # Shared utilities
└── tests/
```

## Migration Steps

### Step 1: Create Interface Layer
```python
# src/interfaces/ml_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MLServiceInterface(ABC):
    @abstractmethod
    async def extract_memories(self, transcript: str) -> List[Memory]:
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    async def spread_activation(self, query: str) -> List[Memory]:
        pass
```

### Step 2: Implement Service Adapters
```python
# src/api/services/ml_client.py
class MLServiceClient(MLServiceInterface):
    """Client for remote ML service"""
    def __init__(self, ml_service_url: str):
        self.url = ml_service_url
    
    async def extract_memories(self, transcript: str) -> List[Memory]:
        # HTTP/gRPC call to ML service
        pass

# src/api/services/ml_local.py  
class MLServiceLocal(MLServiceInterface):
    """Local implementation (current monolith)"""
    def __init__(self):
        self.extractor = MemoryExtractor()
        self.encoder = ONNXEncoder()
    
    async def extract_memories(self, transcript: str) -> List[Memory]:
        # Direct call to local implementation
        return await self.extractor.extract(transcript)
```

### Step 3: Dependency Injection Setup
```python
# src/api/dependencies.py
from functools import lru_cache
from src.interfaces import MLServiceInterface

@lru_cache()
def get_ml_service() -> MLServiceInterface:
    if settings.ML_SERVICE_MODE == "remote":
        return MLServiceClient(settings.ML_SERVICE_URL)
    else:
        return MLServiceLocal()

# Usage in routers
@router.post("/process")
async def process_transcript(
    transcript: str,
    ml_service: MLServiceInterface = Depends(get_ml_service)
):
    memories = await ml_service.extract_memories(transcript)
    return memories
```

## Benefits of This Structure

1. **Gradual Migration**: Can start with local implementation, switch to remote later
2. **Testing**: Easy to mock interfaces for testing
3. **Flexibility**: Can run as monolith or microservices
4. **Clear Boundaries**: Well-defined interfaces between components
5. **Independent Development**: Teams can work on different services

## Environment Variables for Configuration

```env
# Monolithic mode (default)
ML_SERVICE_MODE=local

# Microservices mode
ML_SERVICE_MODE=remote
ML_SERVICE_URL=http://ml-service:8001
```

## Docker Compose for Microservices

```yaml
version: '3.8'

services:
  api:
    build:
      context: ./cognitive-meeting-api
      dockerfile: Dockerfile
    environment:
      - ML_SERVICE_MODE=remote
      - ML_SERVICE_URL=http://ml:8001
    ports:
      - "8000:8000"
    depends_on:
      - ml
      - qdrant
      - postgres

  ml:
    build:
      context: ./cognitive-meeting-ml
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    depends_on:
      - qdrant
      - postgres

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: meetings
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
```

## Next Steps

1. Implement interface layer in current monolith
2. Add service adapter pattern
3. Test with local implementation
4. Create separate package repositories
5. Implement remote service communication
6. Deploy as microservices