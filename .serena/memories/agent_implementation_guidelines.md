# Agent Implementation Guidelines

## Philosophy
We provide specifications, not implementations. This allows agents to:
- Make appropriate technology choices
- Follow project patterns they discover
- Write code that fits the codebase
- Learn from tests and iterate

## What We Provide to Agents

### 1. Clear Specifications
```python
# Example: Vector Manager Spec
class VectorManager:
    """Manages 400D vector composition"""
    
    def compose_vector(self, semantic: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        """
        Compose 384D semantic + 16D features into 400D vector
        
        Args:
            semantic: 384D normalized embedding from ONNX
            dimensions: 16D feature vector from analyzers
            
        Returns:
            400D vector suitable for Qdrant storage
            
        Constraints:
            - Semantic portion must remain normalized
            - Dimensions must be in [0, 1] range
            - Output shape must be exactly (400,)
            - Performance: <50ms
            
        Example:
            >>> semantic = encoder.encode("Hello world")  # (384,)
            >>> dims = analyzer.extract("Hello world")    # (16,)
            >>> vector = manager.compose_vector(semantic, dims)
            >>> assert vector.shape == (400,)
        """
```

### 2. Test Cases
```python
# tests/test_vector_manager.py
def test_vector_composition():
    """Agent can run this to verify implementation"""
    manager = VectorManager()
    
    # Test inputs
    semantic = np.random.randn(384)
    semantic = semantic / np.linalg.norm(semantic)  # Normalize
    dimensions = np.random.rand(16)  # Already in [0,1]
    
    # Test composition
    result = manager.compose_vector(semantic, dimensions)
    
    # Verify constraints
    assert result.shape == (400,), f"Expected (400,), got {result.shape}"
    assert np.allclose(np.linalg.norm(result[:384]), 1.0), "Semantic not normalized"
    assert np.all(result[384:] >= 0) and np.all(result[384:] <= 1), "Dims out of range"
```

### 3. Integration Points
```python
# How components connect
pipeline = MeetingIngestionPipeline(
    # These are the interfaces to implement
    memory_extractor=MemoryExtractor(),      # Extracts memories from transcript
    encoder=ONNXEncoder(),                   # Creates 384D embeddings  
    dimension_analyzer=DimensionAnalyzer(),  # Creates 16D features
    vector_manager=VectorManager(),          # Combines to 400D
    vector_store=QdrantVectorStore(),       # Stores in Qdrant
    memory_repo=MemoryRepository()          # Stores in SQLite
)
```

### 4. Performance Benchmarks
```python
# benchmarks/test_performance.py
@pytest.mark.benchmark
def test_encoding_performance(benchmark):
    encoder = ONNXEncoder()
    text = "This is a sample meeting transcript segment"
    
    result = benchmark(encoder.encode, text)
    
    # Agent knows the targets
    assert benchmark.stats['mean'] < 0.1  # <100ms average
    assert benchmark.stats['max'] < 0.15  # <150ms worst case
```

## What Agents Implement

### 1. Analyze Existing Patterns
Before implementing, agent should:
```bash
# Look for patterns in codebase
find . -name "*.py" -type f | head -20  # See structure
grep -r "class.*Repository" --include="*.py"  # Find patterns
grep -r "async def" --include="*.py"  # Async vs sync?
```

### 2. Follow Discovered Conventions
If agent finds:
- Existing base classes → Inherit from them
- Naming patterns → Follow them
- Error handling patterns → Reuse them
- Logging patterns → Consistent logging

### 3. Implement Incrementally
```python
# Step 1: Simplest working version
class MemoryExtractor:
    def extract(self, transcript: str) -> List[Memory]:
        # Just split sentences initially
        sentences = transcript.split('.')
        return [Memory(content=s.strip()) for s in sentences if s.strip()]

# Step 2: Add pattern matching
    def extract(self, transcript: str) -> List[Memory]:
        memories = []
        for sentence in transcript.split('.'):
            memory_type = self._classify(sentence)
            if memory_type:
                memories.append(Memory(
                    content=sentence.strip(),
                    memory_type=memory_type
                ))
        return memories

# Step 3: Add speaker extraction, timestamps, etc.
```

### 4. Validate Continuously
```bash
# After each implementation step
pytest tests/unit/test_component.py -v

# Check performance
pytest benchmarks/ -v

# Integration test
pytest tests/integration/ -v
```

## Example: Agent Implementing Temporal Extractor

### Specification Provided
```python
class TemporalDimensionExtractor:
    """
    Extract 4D temporal features:
    [0] urgency: 0-1 based on keywords
    [1] deadline: 0-1 based on date proximity  
    [2] sequence: 0-1 position in conversation
    [3] duration: 0-1 time-relevance
    
    Performance: <10ms per extraction
    """
```

### Agent Process
1. **Check for existing extractors**:
   ```bash
   find . -name "*extractor*.py"
   ls src/extraction/dimensions/
   ```

2. **Study patterns if found**:
   ```python
   # If finds base class
   class DimensionExtractor(ABC):
       @abstractmethod
       def extract(self, text: str, context: dict = None) -> np.ndarray:
           pass
   ```

3. **Implement following pattern**:
   ```python
   class TemporalDimensionExtractor(DimensionExtractor):
       def __init__(self):
           self.urgency_keywords = {
               'urgent': 1.0, 'asap': 1.0, 'immediately': 0.9,
               'critical': 0.8, 'important': 0.6
           }
           
       def extract(self, text: str, context: dict = None) -> np.ndarray:
           # Implementation details...
           return np.array([urgency, deadline, sequence, duration])
   ```

4. **Validate implementation**:
   ```python
   def test_temporal_extraction():
       extractor = TemporalDimensionExtractor()
       
       # Test urgent text
       result = extractor.extract("This is urgent! Need ASAP!")
       assert result.shape == (4,)
       assert result[0] > 0.8  # High urgency
       
       # Test performance
       import time
       start = time.time()
       for _ in range(100):
           extractor.extract("Test text")
       assert (time.time() - start) / 100 < 0.01  # <10ms
   ```

## Common Patterns to Provide

### Repository Pattern
```python
class BaseRepository:
    """Agent can inherit if found"""
    def __init__(self, db_connection):
        self.db = db_connection
        
    @contextmanager
    def transaction(self):
        # Transaction handling
        pass
```

### Error Handling
```python
class ComponentError(Exception):
    """Base error for component failures"""
    pass

class ExtractionError(ComponentError):
    """Specific to extraction failures"""
    pass

# Usage pattern
try:
    result = extractor.extract(text)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e}")
    # Graceful degradation
```

### Async Patterns
```python
# If codebase uses async
async def process_transcript(self, transcript: str) -> List[Memory]:
    # Async implementation
    pass

# If codebase is sync
def process_transcript(self, transcript: str) -> List[Memory]:
    # Sync implementation
    pass
```

## Benefits of This Approach

1. **Flexibility**: Agents adapt to existing codebase
2. **Learning**: Agents understand project patterns
3. **Consistency**: Code follows project style
4. **Maintainability**: Easier for humans to review
5. **Testability**: Clear success criteria

## What NOT to Do

1. **Don't provide complete implementations** - Let agents write code
2. **Don't assume technology choices** - Let agents discover what's used
3. **Don't over-specify internals** - Focus on interfaces and behavior
4. **Don't skip discovery phase** - Agents should explore first

## Summary

We provide the "what" and "why", agents determine the "how" based on:
- Specifications and interfaces
- Test cases and benchmarks  
- Performance targets
- Integration points

This creates better code that fits naturally into the existing codebase while meeting all requirements.