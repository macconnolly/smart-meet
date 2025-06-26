# Phase 4: Memory Consolidation & Lifecycle (Week 4)

Implement automated memory consolidation, clustering, and lifecycle management for scalability.

## Day 1-2: Consolidation Models & Architecture

### Task 1: Consolidation Data Models
- **Where**: `src/models/consolidation.py`
- **Create**:
  ```python
  @dataclass
  class MemoryCluster:
      cluster_id: str
      memory_ids: List[str]
      centroid_vector: np.ndarray
      theme: str  # Extracted common theme
      size: int
      coherence_score: float  # How well memories fit together
      
  @dataclass
  class SemanticMemory:
      id: str
      content: str  # Generated summary/abstraction
      source_memories: List[str]  # Original L2 memories
      level: int = 1  # L1 by default
      importance: float
      created_from: str  # 'consolidation' or 'promotion'
      
  @dataclass
  class ConsolidationResult:
      clusters_found: int
      semantic_memories_created: int
      memories_promoted: int
      processing_time_ms: int
  ```

### Task 2: Design Consolidation Strategy
- **Criteria for Consolidation**:
  - Memories accessed together frequently
  - High semantic similarity (>0.8)
  - Same meeting or time period
  - Similar importance scores
- **Promotion Rules**:
  - L2→L1: After 5+ accesses or in 3+ queries
  - L1→L0: Manual curation or extreme importance

### Task 3: Update Schema for Parent-Child
- **Where**: Update memory model and repository
- **Add**: Parent-child relationship tracking
- **Purpose**: Link semantic memories to their sources

## Day 3-4: Clustering Implementation

### Task 1: Memory Clusterer
- **Where**: `src/cognitive/consolidation/memory_clusterer.py`
- **Implementation**:
  ```python
  from sklearn.cluster import DBSCAN
  import numpy as np
  
  class MemoryClusterer:
      def __init__(self, 
                   eps: float = 0.2,  # Maximum distance between samples
                   min_samples: int = 3):  # Minimum cluster size
          self.eps = eps
          self.min_samples = min_samples
          
      async def find_clusters(self, 
                            memory_vectors: Dict[str, np.ndarray],
                            metadata: Dict[str, Any]) -> List[MemoryCluster]:
          """
          Find clusters of related memories using DBSCAN
          """
          # Prepare data
          memory_ids = list(memory_vectors.keys())
          vectors = np.array([memory_vectors[id] for id in memory_ids])
          
          # Run DBSCAN
          clustering = DBSCAN(
              eps=self.eps,
              min_samples=self.min_samples,
              metric='cosine'
          ).fit(vectors)
          
          # Process clusters
          clusters = []
          for cluster_id in set(clustering.labels_):
              if cluster_id == -1:  # Noise points
                  continue
                  
              # Get cluster members
              mask = clustering.labels_ == cluster_id
              cluster_memory_ids = [memory_ids[i] for i in range(len(memory_ids)) if mask[i]]
              cluster_vectors = vectors[mask]
              
              # Calculate centroid
              centroid = np.mean(cluster_vectors, axis=0)
              centroid = centroid / np.linalg.norm(centroid)
              
              # Extract theme (simplified - could use LLM in future)
              theme = self._extract_theme(cluster_memory_ids, metadata)
              
              clusters.append(MemoryCluster(
                  cluster_id=f"cluster-{uuid.uuid4()}",
                  memory_ids=cluster_memory_ids,
                  centroid_vector=centroid,
                  theme=theme,
                  size=len(cluster_memory_ids),
                  coherence_score=self._calculate_coherence(cluster_vectors)
              ))
              
          return clusters
  ```

### Task 2: Theme Extraction
- **Simple Rule-Based Approach**:
  ```python
  def _extract_theme(self, memory_ids: List[str], metadata: Dict[str, Any]) -> str:
      # Get memory contents
      contents = [metadata[id]['content'] for id in memory_ids]
      
      # Extract common words (excluding stopwords)
      word_freq = Counter()
      for content in contents:
          words = content.lower().split()
          word_freq.update(w for w in words if w not in STOPWORDS)
      
      # Get top 3 words as theme
      top_words = word_freq.most_common(3)
      return " ".join([word for word, _ in top_words])
  ```

### Task 3: Coherence Calculation
- **Measure Cluster Quality**:
  ```python
  def _calculate_coherence(self, cluster_vectors: np.ndarray) -> float:
      """Calculate average pairwise similarity within cluster"""
      if len(cluster_vectors) < 2:
          return 1.0
          
      similarities = []
      for i in range(len(cluster_vectors)):
          for j in range(i + 1, len(cluster_vectors)):
              sim = np.dot(cluster_vectors[i], cluster_vectors[j])
              similarities.append(sim)
              
      return np.mean(similarities)
  ```

## Day 5: Consolidation Engine

### Task 1: Consolidation Engine Core
- **Where**: `src/cognitive/consolidation/consolidation_engine.py`
- **Implementation**:
  ```python
  class ConsolidationEngine:
      def __init__(self,
                   memory_repo: MemoryRepository,
                   vector_store: QdrantVectorStore,
                   clusterer: MemoryClusterer):
          self.memory_repo = memory_repo
          self.vector_store = vector_store
          self.clusterer = clusterer
          
      async def consolidate_memories(self, 
                                   time_window: timedelta = timedelta(days=7),
                                   min_accesses: int = 3) -> ConsolidationResult:
          """
          Main consolidation process:
          1. Select candidate memories
          2. Cluster similar memories
          3. Create semantic memories
          4. Promote frequently accessed memories
          """
  ```

### Task 2: Candidate Selection
- **Select Memories for Consolidation**:
  ```python
  async def _select_candidates(self, time_window: timedelta, min_accesses: int):
      # Get L2 memories that are:
      # - Accessed at least min_accesses times
      # - Created within time_window
      # - Not already consolidated
      
      candidates = await self.memory_repo.get_consolidation_candidates(
          level=2,
          min_accesses=min_accesses,
          since=datetime.now() - time_window
      )
      
      # Load vectors
      memory_vectors = {}
      for memory in candidates:
          vector_data = await self.vector_store.get_by_id(memory.id, level=2)
          if vector_data:
              memory_vectors[memory.id] = vector_data['vector']
              
      return candidates, memory_vectors
  ```

### Task 3: Semantic Memory Generation
- **Create Abstract Memories from Clusters**:
  ```python
  async def _create_semantic_memory(self, cluster: MemoryCluster, 
                                  source_memories: List[Memory]) -> SemanticMemory:
      # Generate consolidated content
      contents = [m.content for m in source_memories]
      
      # Simple summarization (in production, use LLM)
      consolidated_content = f"[{cluster.theme}] " + self._summarize_contents(contents)
      
      # Calculate importance as max of sources
      importance = max(m.importance for m in source_memories)
      
      # Create semantic memory
      semantic_memory = SemanticMemory(
          content=consolidated_content,
          source_memories=[m.id for m in source_memories],
          level=1,
          importance=importance * 1.2,  # Boost importance
          created_from='consolidation'
      )
      
      # Store in L1
      await self.memory_repo.create(semantic_memory)
      await self.vector_store.store_memory(
          semantic_memory.id,
          cluster.centroid_vector,
          level=1,
          metadata={'type': 'semantic', 'theme': cluster.theme}
      )
      
      return semantic_memory
  ```

### Task 4: Individual Memory Promotion
- **Promote High-Value L2 Memories**:
  ```python
  async def _promote_individual_memories(self, min_accesses: int = 5):
      # Find L2 memories accessed frequently
      promotion_candidates = await self.memory_repo.get_promotion_candidates(
          level=2,
          min_accesses=min_accesses
      )
      
      promoted = []
      for memory in promotion_candidates:
          # Move to L1
          await self.memory_repo.update_level(memory.id, new_level=1)
          
          # Move vector to L1 collection
          vector_data = await self.vector_store.get_by_id(memory.id, level=2)
          if vector_data:
              await self.vector_store.store_memory(
                  memory.id,
                  vector_data['vector'],
                  level=1,
                  metadata={**vector_data['payload'], 'promoted': True}
              )
              # Optionally remove from L2
              
          promoted.append(memory.id)
          
      return promoted
  ```

## Day 6: Lifecycle Management & Scheduling

### Task 1: Memory Lifecycle Manager
- **Where**: `src/cognitive/consolidation/lifecycle_manager.py`
- **Implementation**:
  ```python
  class MemoryLifecycleManager:
      def __init__(self, memory_repo: MemoryRepository):
          self.memory_repo = memory_repo
          
      async def update_importance_decay(self, decay_rate: float = 0.95):
          """
          Decay importance of memories not accessed recently
          """
          stale_memories = await self.memory_repo.get_stale_memories(
              days_inactive=7
          )
          
          for memory in stale_memories:
              new_importance = memory.importance * decay_rate
              if new_importance < 0.1:  # Threshold for removal
                  await self._archive_memory(memory)
              else:
                  await self.memory_repo.update_importance(
                      memory.id, 
                      new_importance
                  )
                  
      async def reinforce_accessed_memories(self):
          """
          Boost importance of recently accessed memories
          """
          recent_accesses = await self.memory_repo.get_recent_accesses(hours=24)
          
          for memory_id, access_count in recent_accesses.items():
              boost = 1.0 + (0.1 * access_count)  # 10% per access
              await self.memory_repo.boost_importance(memory_id, boost)
  ```

### Task 2: Consolidation Scheduler
- **Where**: `src/cognitive/consolidation/scheduler.py`
- **Background Task Implementation**:
  ```python
  import asyncio
  from datetime import datetime, time
  
  class ConsolidationScheduler:
      def __init__(self, 
                   consolidation_engine: ConsolidationEngine,
                   lifecycle_manager: MemoryLifecycleManager):
          self.consolidation_engine = consolidation_engine
          self.lifecycle_manager = lifecycle_manager
          self.running = False
          
      async def start(self):
          """Start background consolidation tasks"""
          self.running = True
          
          # Schedule daily consolidation at 2 AM
          asyncio.create_task(self._daily_consolidation())
          
          # Schedule hourly importance updates
          asyncio.create_task(self._hourly_importance_update())
          
      async def _daily_consolidation(self):
          while self.running:
              # Wait until 2 AM
              now = datetime.now()
              next_run = now.replace(hour=2, minute=0, second=0)
              if next_run < now:
                  next_run += timedelta(days=1)
                  
              wait_seconds = (next_run - now).total_seconds()
              await asyncio.sleep(wait_seconds)
              
              # Run consolidation
              try:
                  result = await self.consolidation_engine.consolidate_memories()
                  logger.info(f"Consolidation complete: {result}")
              except Exception as e:
                  logger.error(f"Consolidation failed: {e}")
  ```

### Task 3: Manual Trigger API
- **Where**: `src/api/routers/consolidation.py`
- **Endpoint for Manual Consolidation**:
  ```python
  @router.post("/api/v2/consolidate")
  async def trigger_consolidation(
      background_tasks: BackgroundTasks,
      time_window_days: int = 7,
      min_accesses: int = 3
  ):
      """Manually trigger memory consolidation"""
      
      # Run in background
      background_tasks.add_task(
          consolidation_engine.consolidate_memories,
          time_window=timedelta(days=time_window_days),
          min_accesses=min_accesses
      )
      
      return {"status": "Consolidation started in background"}
  ```

## Day 7: Testing & Integration

### Task 1: Clustering Tests
- **Where**: `tests/unit/test_clustering.py`
- **Test Cases**:
  ```python
  async def test_dbscan_clustering():
      # Create test memories with known clusters
      memories = [
          # Cluster 1: Technical decisions
          ("We decided to use Python", [0.9, 0.1, ...]),
          ("Python was chosen for the backend", [0.85, 0.15, ...]),
          ("The team agreed on Python", [0.88, 0.12, ...]),
          
          # Cluster 2: Timeline discussions  
          ("The deadline is next Friday", [0.1, 0.9, ...]),
          ("We need to finish by Friday", [0.15, 0.85, ...]),
          
          # Outlier
          ("Lunch will be provided", [0.5, 0.5, ...])
      ]
      
      clusters = await clusterer.find_clusters(memories)
      assert len(clusters) == 2
      assert all(c.coherence_score > 0.8 for c in clusters)
  ```

### Task 2: Consolidation Integration Test
- **Where**: `tests/integration/test_consolidation.py`
- **End-to-End Test**:
  ```python
  async def test_consolidation_pipeline():
      # 1. Create memories with high similarity
      meeting_id = await create_test_meeting()
      memory_ids = await create_similar_memories(meeting_id, count=10)
      
      # 2. Simulate accesses
      for memory_id in memory_ids[:5]:
          for _ in range(4):
              await memory_repo.update_access(memory_id)
              
      # 3. Run consolidation
      result = await consolidation_engine.consolidate_memories(
          time_window=timedelta(days=1),
          min_accesses=3
      )
      
      # 4. Verify semantic memory created
      assert result.semantic_memories_created >= 1
      
      # 5. Verify L1 has new semantic memory
      l1_memories = await vector_store.search(
          query_vector=test_vector,
          level=1
      )
      assert any('semantic' in m[2].get('type', '') for m in l1_memories)
  ```

### Task 3: Lifecycle Tests
- **Test Decay and Reinforcement**:
  ```python
  async def test_importance_decay():
      # Create memory with high importance
      memory = Memory(content="Important decision", importance=0.9)
      memory_id = await memory_repo.create(memory)
      
      # Simulate no access for 7 days
      await memory_repo.update_last_accessed(memory_id, datetime.now() - timedelta(days=8))
      
      # Run decay
      await lifecycle_manager.update_importance_decay()
      
      # Check importance decreased
      updated = await memory_repo.get_by_id(memory_id)
      assert updated.importance < 0.9
      assert updated.importance == pytest.approx(0.9 * 0.95)
  ```

### Task 4: Performance Validation
- **Consolidation Performance**:
  - Process 1000 memories in <5s
  - Cluster finding scales linearly
  - Background tasks don't block API

## Success Criteria

### Functionality
- ✅ DBSCAN clustering identifies related memories
- ✅ Semantic memories created from clusters
- ✅ Individual memory promotion working
- ✅ Importance decay/reinforcement implemented
- ✅ Background scheduling operational

### Performance
- ✅ Consolidation <5s for 1000 memories
- ✅ Clustering scales to 10K memories
- ✅ Background tasks non-blocking
- ✅ Memory footprint remains stable

### Quality
- ✅ Meaningful clusters formed
- ✅ Semantic memories capture essence
- ✅ No memory loss during consolidation
- ✅ Parent-child relationships maintained

## Configuration

```python
# config/consolidation.yaml
consolidation:
  clustering:
    eps: 0.2
    min_samples: 3
  promotion:
    min_accesses: 5
    boost_factor: 1.2
  lifecycle:
    decay_rate: 0.95
    decay_after_days: 7
    archive_threshold: 0.1
  scheduling:
    consolidation_hour: 2
    importance_update_interval_hours: 1
```

## Next Phase Preview
Phase 5 will focus on:
- Production hardening
- Performance optimization
- Security implementation
- Deployment readiness