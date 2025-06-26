# Cognitive Meeting Intelligence MVP - Technical Specification

## 1. System Overview

### 1.1 Purpose
The Cognitive Meeting Intelligence System transforms organizational meetings into a queryable, thinking memory network. It ingests meeting transcripts, extracts multi-dimensional memories, and provides intelligent retrieval through activation spreading and bridge discovery.

### 1.2 Core Capabilities
- **Meeting Ingestion**: Process transcripts into structured memories
- **Cognitive Memory**: Dual episodic/semantic system with decay
- **Intelligent Retrieval**: Activation spreading and bridge discovery
- **Multi-dimensional Analysis**: 400D vectors (384D semantic + 16D features)
- **Hierarchical Storage**: 3-tier Qdrant collections (L0/L1/L2)

### 1.3 Technical Stack
- **Vector Database**: Qdrant (local mode)
- **ML Runtime**: ONNX Runtime with all-MiniLM-L6-v2
- **Persistence**: SQLite for metadata and relationships
- **API Framework**: FastAPI
- **Dimensional Analysis**: VADER, regex patterns, rule-based extractors

## 2. Data Models

### 2.1 Meeting Model
```python
@dataclass
class Meeting:
    id: str  # UUID
    title: str
    start_time: datetime
    end_time: datetime
    participants: List[str]
    transcript_path: str
    metadata: Dict[str, Any]  # Platform, series_id, etc.
    created_at: datetime
    processed_at: Optional[datetime]
    memory_count: int = 0
```

### 2.2 Memory Model
```python
@dataclass
class Memory:
    id: str  # UUID
    meeting_id: str  # Reference to source meeting
    content: str
    speaker: Optional[str]
    timestamp: float  # Seconds from meeting start
    
    # Classification
    memory_type: MemoryType  # episodic, semantic
    content_type: ContentType  # decision, action, commitment, question, insight
    level: int  # 0=concept, 1=context, 2=episode
    
    # Vectors and dimensions
    qdrant_id: str  # Reference to vector storage
    embedding: Optional[np.ndarray]  # 400D vector (not stored in DB)
    dimensions: Dict[str, float]  # 16D features
    
    # Lifecycle
    importance_score: float = 0.5
    decay_rate: float = 0.1  # 0.1 for episodic, 0.01 for semantic
    access_count: int = 0
    last_accessed: datetime
    created_at: datetime
    
    # Hierarchy
    parent_id: Optional[str]  # For L0/L1 relationships
```

### 2.3 Connection Model
```python
@dataclass
class MemoryConnection:
    source_id: str
    target_id: str
    connection_strength: float = 0.5
    connection_type: str = "associative"
    created_at: datetime
    last_activated: Optional[datetime]
    activation_count: int = 0
```

## 3. Database Schema

### 3.1 SQLite Schema
```sql
-- Meetings table
CREATE TABLE meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    participants_json TEXT NOT NULL,  -- JSON array
    transcript_path TEXT NOT NULL,
    metadata_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    memory_count INTEGER DEFAULT 0
);

-- Memories table (full schema from Phase 1)
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    content TEXT NOT NULL,
    speaker TEXT,
    timestamp REAL,  -- Seconds from meeting start
    
    -- Classification
    memory_type TEXT DEFAULT 'episodic' CHECK(memory_type IN ('episodic', 'semantic')),
    content_type TEXT CHECK(content_type IN ('decision', 'action', 'commitment', 'question', 'insight')),
    level INTEGER NOT NULL CHECK(level IN (0,1,2)),
    
    -- Vectors (stored in Qdrant, referenced here)
    qdrant_id TEXT NOT NULL UNIQUE,
    dimensions_json TEXT NOT NULL,  -- 16D features as JSON
    
    -- Lifecycle
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Hierarchy
    parent_id TEXT,
    
    FOREIGN KEY (meeting_id) REFERENCES meetings(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id)
);

-- Connection graph
CREATE TABLE memory_connections (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    connection_strength REAL NOT NULL DEFAULT 0.5,
    connection_type TEXT DEFAULT 'associative',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activated DATETIME,
    activation_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

-- Bridge cache for performance
CREATE TABLE bridge_cache (
    query_hash TEXT NOT NULL,
    bridge_memory_id TEXT NOT NULL,
    bridge_score REAL NOT NULL,
    novelty_score REAL NOT NULL,
    connection_potential REAL NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    PRIMARY KEY (query_hash, bridge_memory_id),
    FOREIGN KEY (bridge_memory_id) REFERENCES memories(id)
);

-- Retrieval statistics for meta-learning
CREATE TABLE retrieval_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    retrieval_type TEXT NOT NULL CHECK(retrieval_type IN ('core','contextual','peripheral','bridge')),
    success_score REAL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Indexes
CREATE INDEX idx_memories_meeting ON memories(meeting_id);
CREATE INDEX idx_memories_type ON memories(memory_type, content_type);
CREATE INDEX idx_memories_level ON memories(level);
CREATE INDEX idx_memories_importance ON memories(importance_score DESC);
CREATE INDEX idx_connections_strength ON memory_connections(connection_strength DESC);
CREATE INDEX idx_bridge_expires ON bridge_cache(expires_at);
```

### 3.2 Qdrant Collections
```yaml
# L0: Cognitive Concepts (highest abstraction)
cognitive_concepts:
  vectors:
    size: 400
    distance: Cosine
  payload_schema:
    memory_id: keyword
    importance_score: float
    child_count: integer
  hnsw_config:
    m: 32
    ef_construct: 400
    full_scan_threshold: 5000

# L1: Cognitive Contexts (patterns)
cognitive_contexts:
  vectors:
    size: 400
    distance: Cosine
  payload_schema:
    memory_id: keyword
    parent_concept_id: keyword
    relevance_score: float
  hnsw_config:
    m: 24
    ef_construct: 300
    full_scan_threshold: 10000

# L2: Cognitive Episodes (raw memories)
cognitive_episodes:
  vectors:
    size: 400
    distance: Cosine
  payload_schema:
    memory_id: keyword
    meeting_id: keyword
    content_type: keyword
    importance_score: float
  hnsw_config:
    m: 16
    ef_construct: 200
    full_scan_threshold: 20000
```

## 4. Meeting Ingestion Pipeline

### 4.1 Ingestion Flow
```python
async def ingest_meeting(transcript: str, metadata: Dict) -> Meeting:
    # 1. Create meeting record
    meeting = Meeting(
        id=str(uuid4()),
        title=metadata.get('title', 'Untitled Meeting'),
        start_time=metadata['start_time'],
        end_time=metadata['end_time'],
        participants=metadata.get('participants', []),
        transcript_path=save_transcript(transcript),
        metadata=metadata
    )
    await meeting_repo.create(meeting)
    
    # 2. Extract memories
    memories = await extract_memories_from_transcript(transcript, meeting.id)
    
    # 3. Generate embeddings and dimensions
    for memory in memories:
        # Generate 384D semantic embedding
        semantic_embedding = onnx_encoder.encode(memory.content)
        
        # Extract 16D dimensional features
        dimensions = dimension_analyzer.analyze(memory.content)
        
        # Create 400D vector
        memory.embedding = np.concatenate([semantic_embedding, dimensions])
        
        # Store in Qdrant (L2 by default)
        memory.qdrant_id = await vector_store.store(
            collection="cognitive_episodes",
            vector=memory.embedding,
            metadata={
                "memory_id": memory.id,
                "meeting_id": meeting.id,
                "content_type": memory.content_type
            }
        )
        
        # Store in SQLite
        await memory_repo.create(memory)
    
    # 4. Create initial connections
    await create_memory_connections(memories)
    
    # 5. Update meeting
    meeting.processed_at = datetime.now()
    meeting.memory_count = len(memories)
    await meeting_repo.update(meeting)
    
    return meeting
```

### 4.2 Memory Extraction
```python
class MemoryExtractor:
    def extract_memories_from_transcript(self, transcript: str, meeting_id: str) -> List[Memory]:
        memories = []
        
        # 1. Split into utterances
        utterances = self.split_utterances(transcript)
        
        # 2. Group into semantic chunks
        chunks = self.group_semantic_chunks(utterances)
        
        # 3. Extract memories from chunks
        for chunk in chunks:
            # Detect memory type
            content_type = self.classify_content_type(chunk.text)
            
            if content_type:
                memory = Memory(
                    id=str(uuid4()),
                    meeting_id=meeting_id,
                    content=chunk.text,
                    speaker=chunk.speaker,
                    timestamp=chunk.timestamp,
                    content_type=content_type,
                    memory_type="episodic",  # All start as episodic
                    level=2,  # Start at L2
                    created_at=datetime.now(),
                    last_accessed=datetime.now()
                )
                memories.append(memory)
        
        return memories
    
    def classify_content_type(self, text: str) -> Optional[ContentType]:
        """Rule-based classification for MVP"""
        patterns = {
            'decision': r'\b(decided?|agreed?|will do|going to|conclusion)\b',
            'action': r'\b(will|going to|need to|should|must)\s+\w+',
            'commitment': r'\b(commit|promise|deliver|by when|deadline)\b',
            'question': r'[?]|^(what|why|how|when|where|who)\b',
            'insight': r'\b(realized|learned|discovered|found out|insight)\b'
        }
        
        for content_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return ContentType(content_type)
        
        return None
```

### 4.3 Connection Creation
```python
async def create_memory_connections(memories: List[Memory]):
    """Create initial connections between memories from same meeting"""
    
    for i, memory1 in enumerate(memories):
        for j, memory2 in enumerate(memories[i+1:], i+1):
            # Temporal proximity
            time_diff = abs(memory1.timestamp - memory2.timestamp)
            if time_diff < 60:  # Within 1 minute
                strength = 1.0 - (time_diff / 60)
                
                # Semantic similarity
                similarity = cosine_similarity(
                    memory1.embedding.reshape(1, -1),
                    memory2.embedding.reshape(1, -1)
                )[0][0]
                
                # Combined strength
                connection_strength = (0.3 * strength) + (0.7 * similarity)
                
                if connection_strength > 0.5:
                    await connection_repo.create(
                        MemoryConnection(
                            source_id=memory1.id,
                            target_id=memory2.id,
                            connection_strength=connection_strength,
                            connection_type="temporal-semantic"
                        )
                    )
```

## 5. Dimensional Analysis

### 5.1 Dimension Extractors (16D Total)

```python
class DimensionAnalyzer:
    def __init__(self):
        self.extractors = {
            'temporal': TemporalExtractor(),      # 4D
            'causal': CausalExtractor(),          # 3D
            'social': SocialExtractor(),          # 3D
            'emotional': EmotionalExtractor(),    # 3D
            'evolutionary': EvolutionaryExtractor(), # 3D
        }
    
    def analyze(self, text: str) -> np.ndarray:
        """Extract 16D feature vector"""
        features = []
        
        for name, extractor in self.extractors.items():
            dimension_features = extractor.extract(text)
            features.extend(dimension_features)
        
        return np.array(features)
```

### 5.2 Temporal Dimension (4D)
```python
class TemporalExtractor:
    def extract(self, text: str) -> np.ndarray:
        features = np.zeros(4)
        
        # 1. Urgency score
        urgency_patterns = r'(urgent|asap|immediately|critical|deadline|now)'
        urgency_matches = len(re.findall(urgency_patterns, text, re.I))
        features[0] = min(urgency_matches / 10.0, 1.0)
        
        # 2. Deadline proximity
        deadline_patterns = r'(by|before|until|due)\s+(\w+\s+\d+|\d+\s+days?|next\s+\w+)'
        deadline_match = re.search(deadline_patterns, text, re.I)
        if deadline_match:
            features[1] = self.parse_deadline_proximity(deadline_match.group(2))
        
        # 3. Sequence position (would be set during extraction)
        features[2] = 0.0  # Placeholder
        
        # 4. Duration relevance
        duration_patterns = r'(\d+)\s*(hours?|days?|weeks?|months?)'
        duration_match = re.search(duration_patterns, text, re.I)
        if duration_match:
            features[3] = self.normalize_duration(duration_match.group(0))
        
        return features
```

### 5.3 Emotional Dimension (3D) - VADER
```python
class EmotionalExtractor:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
    
    def extract(self, text: str) -> np.ndarray:
        scores = self.analyzer.polarity_scores(text)
        
        return np.array([
            scores['compound'],           # Overall sentiment (-1 to 1)
            scores['pos'] - scores['neg'], # Polarity
            abs(scores['compound'])       # Intensity (0 to 1)
        ])
```

### 5.4 Social Dimension (3D)
```python
class SocialExtractor:
    def extract(self, text: str) -> np.ndarray:
        features = np.zeros(3)
        
        # 1. Speaker authority (would be enhanced with speaker role)
        authority_markers = r'\b(decide|approve|authorize|direct|mandate)\b'
        features[0] = len(re.findall(authority_markers, text, re.I)) / 10.0
        
        # 2. Audience relevance
        mention_patterns = r'@\w+|\b(you|your|team|everyone)\b'
        features[1] = len(re.findall(mention_patterns, text, re.I)) / 10.0
        
        # 3. Interaction score
        interaction_patterns = r'\b(agree|disagree|suggest|propose|think)\b'
        features[2] = len(re.findall(interaction_patterns, text, re.I)) / 10.0
        
        return np.clip(features, 0, 1)
```

## 6. Cognitive Features

### 6.1 Activation Spreading
```python
class ActivationEngine:
    def __init__(self, memory_repo, vector_store):
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        self.activation_threshold = 0.7
        self.decay_factor = 0.8
        self.max_activations = 50
    
    async def spread_activation(self, query_vector: np.ndarray) -> ActivationResult:
        """Two-phase BFS activation spreading"""
        
        # Phase 1: Query L0 concepts
        l0_results = await self.vector_store.search(
            collection="cognitive_concepts",
            query_vector=query_vector,
            limit=5
        )
        
        # Initialize activation
        activated = {}
        queue = deque()
        
        for result in l0_results:
            memory_id = result.payload['memory_id']
            activated[memory_id] = ActivatedMemory(
                memory_id=memory_id,
                activation_level=result.score,
                depth=0,
                path=[memory_id]
            )
            queue.append((memory_id, result.score, 0))
        
        # Phase 2: BFS through connection graph
        visited = set(activated.keys())
        
        while queue and len(activated) < self.max_activations:
            current_id, current_activation, depth = queue.popleft()
            
            # Skip if activation too low
            if current_activation < self.activation_threshold:
                continue
            
            # Get connections
            connections = await self.memory_repo.get_connections(current_id)
            
            for conn in connections:
                if conn.target_id in visited:
                    continue
                
                # Calculate new activation
                new_activation = current_activation * conn.connection_strength * self.decay_factor
                
                if new_activation >= self.activation_threshold:
                    activated[conn.target_id] = ActivatedMemory(
                        memory_id=conn.target_id,
                        activation_level=new_activation,
                        depth=depth + 1,
                        path=activated[current_id].path + [conn.target_id]
                    )
                    
                    queue.append((conn.target_id, new_activation, depth + 1))
                    visited.add(conn.target_id)
        
        # Classify results
        return self.classify_results(activated)
    
    def classify_results(self, activated: Dict[str, ActivatedMemory]) -> ActivationResult:
        core = []
        contextual = []
        peripheral = []
        
        for memory in activated.values():
            if memory.activation_level >= 0.9:
                core.append(memory)
            elif memory.activation_level >= 0.7:
                contextual.append(memory)
            else:
                peripheral.append(memory)
        
        return ActivationResult(
            core_memories=core,
            contextual_memories=contextual,
            peripheral_memories=peripheral,
            activation_paths={m.memory_id: m.path for m in activated.values()}
        )
```

### 6.2 Bridge Discovery
```python
class BridgeDiscoveryEngine:
    def __init__(self, vector_store, memory_repo):
        self.vector_store = vector_store
        self.memory_repo = memory_repo
        self.novelty_weight = 0.6
        self.connection_weight = 0.4
        self.bridge_threshold = 0.7
    
    async def discover_bridges(self, query_vector: np.ndarray, 
                              activated_memories: Set[str]) -> List[BridgeMemory]:
        """Distance inversion algorithm for serendipitous discovery"""
        
        # Search for memories with low similarity to query
        all_results = await self.vector_store.search(
            collection="cognitive_episodes",
            query_vector=query_vector,
            limit=1000
        )
        
        bridges = []
        activated_vectors = {}
        
        # Cache activated memory vectors
        for memory_id in activated_memories:
            memory = await self.memory_repo.get_memory(memory_id)
            if memory and memory.embedding is not None:
                activated_vectors[memory_id] = memory.embedding
        
        for result in all_results:
            memory_id = result.payload['memory_id']
            
            # Skip if already activated
            if memory_id in activated_memories:
                continue
            
            # Skip if too similar (not novel)
            if result.score > 0.5:
                continue
            
            # Calculate connection potential to activated set
            max_connection = 0.0
            best_connection_id = None
            
            memory = await self.memory_repo.get_memory(memory_id)
            if not memory or memory.embedding is None:
                continue
            
            for activated_id, activated_vector in activated_vectors.items():
                similarity = cosine_similarity(
                    memory.embedding.reshape(1, -1),
                    activated_vector.reshape(1, -1)
                )[0][0]
                
                if similarity > max_connection:
                    max_connection = similarity
                    best_connection_id = activated_id
            
            # Calculate bridge score
            novelty_score = 1.0 - result.score
            bridge_score = (self.novelty_weight * novelty_score) + \
                          (self.connection_weight * max_connection)
            
            if bridge_score > self.bridge_threshold:
                bridges.append(BridgeMemory(
                    memory=memory,
                    bridge_score=bridge_score,
                    novelty_score=novelty_score,
                    connection_potential=max_connection,
                    connection_path=[best_connection_id, memory_id]
                ))
        
        # Return top 5 bridges
        bridges.sort(key=lambda b: b.bridge_score, reverse=True)
        return bridges[:5]
```

### 6.3 Memory Consolidation
```python
class ConsolidationEngine:
    def __init__(self, memory_repo, vector_store):
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        self.access_threshold = 5
        self.time_window_days = 7
        self.cluster_min_size = 3
    
    async def consolidate_memories(self):
        """Promote frequently accessed episodic memories to semantic"""
        
        # Find consolidation candidates
        candidates = await self.memory_repo.get_consolidation_candidates(
            access_count=self.access_threshold,
            days_old=self.time_window_days
        )
        
        # Cluster similar memories
        clusters = self.cluster_memories(candidates)
        
        for cluster in clusters:
            if len(cluster) >= self.cluster_min_size:
                # Create semantic memory from cluster
                semantic_memory = await self.create_semantic_memory(cluster)
                
                # Promote to L1
                semantic_memory.level = 1
                semantic_memory.memory_type = "semantic"
                semantic_memory.decay_rate = 0.01
                
                # Store in L1 collection
                await self.vector_store.store(
                    collection="cognitive_contexts",
                    vector=semantic_memory.embedding,
                    metadata={
                        "memory_id": semantic_memory.id,
                        "parent_concept_id": None,
                        "relevance_score": semantic_memory.importance_score
                    }
                )
                
                # Update database
                await self.memory_repo.update(semantic_memory)
                
                # Link episodic memories to semantic parent
                for memory in cluster:
                    memory.parent_id = semantic_memory.id
                    await self.memory_repo.update(memory)
```

## 7. API Specification

### 7.1 Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Cognitive Meeting Intelligence API")

# Request/Response Models
class IngestRequest(BaseModel):
    transcript: str
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    use_activation: bool = True
    include_bridges: bool = True
    limit: int = 20

class QueryResponse(BaseModel):
    results: List[MemoryResult]
    activated: Optional[List[ActivatedMemory]]
    bridges: Optional[List[BridgeResult]]
    processing_time_ms: float

# Endpoints
@app.post("/api/v1/meetings/ingest", response_model=Meeting)
async def ingest_meeting(request: IngestRequest):
    """Ingest a meeting transcript and extract memories"""
    try:
        meeting = await ingest_meeting(request.transcript, request.metadata)
        return meeting
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_memories(request: QueryRequest):
    """Query memories with cognitive features"""
    start_time = time.time()
    
    # Generate query embedding
    query_embedding = onnx_encoder.encode(request.query)
    query_dimensions = dimension_analyzer.analyze(request.query)
    query_vector = np.concatenate([query_embedding, query_dimensions])
    
    # Basic search
    results = await search_memories(query_vector, request.limit)
    
    # Activation spreading
    activated = None
    if request.use_activation and results:
        activation_result = await activation_engine.spread_activation(query_vector)
        activated = activation_result.all_memories()
    
    # Bridge discovery
    bridges = None
    if request.include_bridges and activated:
        activated_ids = {m.memory_id for m in activated}
        bridges = await bridge_engine.discover_bridges(query_vector, activated_ids)
    
    processing_time = (time.time() - start_time) * 1000
    
    return QueryResponse(
        results=results,
        activated=activated,
        bridges=bridges,
        processing_time_ms=processing_time
    )

@app.get("/api/v1/meetings/{meeting_id}/memories")
async def get_meeting_memories(meeting_id: str):
    """Get all memories from a specific meeting"""
    memories = await memory_repo.get_by_meeting(meeting_id)
    return {"memories": memories, "count": len(memories)}

@app.post("/api/v1/consolidate")
async def trigger_consolidation():
    """Manually trigger memory consolidation"""
    result = await consolidation_engine.consolidate_memories()
    return {"consolidated": result.consolidated_count}
```

## 8. Performance Specifications

### 8.1 Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Memory extraction | 10-15/second | Per meeting hour |
| Embedding generation | <100ms | Single memory |
| Vector storage | <50ms | Single vector |
| Similarity search | <200ms | 10K memories |
| Activation spreading | <500ms | 50 activations |
| Bridge discovery | <1s | 5 bridges |
| Full query pipeline | <2s | End-to-end |

### 8.2 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8GB | 16GB |
| Storage | 50GB | 200GB |
| ONNX Runtime | 100MB | 100MB |
| Model size | 100MB | 100MB |

## 9. Configuration

### 9.1 Configuration Schema
```yaml
# config/default.yaml
app:
  name: "Cognitive Meeting Intelligence"
  version: "1.0.0"
  environment: "development"

storage:
  qdrant:
    path: "./data/qdrant"  # Local storage for MVP
    collection_configs:
      l0_size: 5000
      l1_size: 10000
      l2_size: 50000
  
  sqlite:
    path: "./data/memories.db"
    pool_size: 5

ml:
  model_path: "./models/all-MiniLM-L6-v2/model.onnx"
  batch_size: 32
  cache_size: 10000

cognitive:
  activation:
    threshold: 0.7
    max_activations: 50
    decay_factor: 0.8
    max_depth: 5
  
  bridges:
    novelty_weight: 0.6
    connection_weight: 0.4
    threshold: 0.7
    max_bridges: 5
    cache_ttl: 3600
  
  consolidation:
    access_threshold: 5
    time_window_days: 7
    cluster_min_size: 3
    
  decay:
    episodic_rate: 0.1
    semantic_rate: 0.01

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
```

## 10. Deployment

### 10.1 Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ONNX model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    model.save('./models/all-MiniLM-L6-v2')"

# Convert to ONNX (if needed)
COPY scripts/convert_to_onnx.py .
RUN python convert_to_onnx.py

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create data directories
RUN mkdir -p data/qdrant data/transcripts

# Run application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.2 Docker Compose
```yaml
version: '3.8'

services:
  cognitive-meeting:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 11. Testing Strategy

### 11.1 Unit Tests
```python
# tests/test_memory_extraction.py
def test_memory_classification():
    extractor = MemoryExtractor()
    
    assert extractor.classify_content_type("We decided to use Python") == "decision"
    assert extractor.classify_content_type("I will send the report tomorrow") == "action"
    assert extractor.classify_content_type("What should we do about the bug?") == "question"

# tests/test_dimensions.py
def test_temporal_extraction():
    extractor = TemporalExtractor()
    features = extractor.extract("This is urgent and due by Friday")
    
    assert features[0] > 0.5  # High urgency
    assert features[1] > 0.0  # Has deadline
```

### 11.2 Integration Tests
```python
# tests/test_integration.py
async def test_full_pipeline():
    # Ingest meeting
    meeting = await ingest_meeting(
        transcript="John: We decided to implement caching. Mary: I'll do it by Friday.",
        metadata={"title": "Tech Planning", "participants": ["John", "Mary"]}
    )
    
    assert meeting.memory_count == 2
    
    # Query with activation
    response = await query_memories(
        query="What did John decide?",
        use_activation=True
    )
    
    assert len(response.results) > 0
    assert response.results[0].content_type == "decision"
```
