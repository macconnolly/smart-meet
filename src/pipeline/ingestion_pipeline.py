"""
Ingestion pipeline for processing meeting transcripts end-to-end.

This module orchestrates the complete pipeline from raw transcript to
stored memories with vectors and connections in the cognitive system.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np

from ..models.entities import (
    Meeting, Memory, MemoryConnection, Vector, ConnectionType
)
from ..extraction.memory_extractor import MemoryExtractor
from ..extraction.dimensions.dimension_analyzer import (
    DimensionAnalyzer, DimensionExtractionContext
)
from ..embedding.onnx_encoder import ONNXEncoder
from ..embedding.vector_manager import VectorManager
from ..storage.sqlite.repositories import (
    MeetingRepository, MemoryRepository, MemoryConnectionRepository
)
from ..storage.qdrant.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingestion pipeline."""
    meeting_id: str
    memories_extracted: int
    memories_stored: int
    connections_created: int
    processing_time_ms: float
    errors: List[str]
    warnings: List[str]


@dataclass
class PipelineConfig:
    """Configuration for ingestion pipeline."""
    batch_size: int = 50
    parallel_encoding: bool = True
    parallel_dimensions: bool = True
    create_sequential_connections: bool = True
    sequential_connection_strength: float = 0.7
    min_memory_length: int = 10  # Minimum characters for valid memory
    max_memory_length: int = 1000  # Maximum characters for single memory


class IngestionPipeline:
    """
    End-to-end pipeline for processing meeting transcripts.
    
    Pipeline stages:
    1. Extract memories from transcript
    2. Generate embeddings for each memory
    3. Extract cognitive dimensions
    4. Compose 400D vectors
    5. Store in SQLite and Qdrant
    6. Create memory connections
    
    Performance target: <2s for typical transcript
    """
    
    def __init__(
        self,
        memory_extractor: MemoryExtractor,
        encoder: ONNXEncoder,
        dimension_analyzer: DimensionAnalyzer,
        vector_manager: VectorManager,
        meeting_repo: MeetingRepository,
        memory_repo: MemoryRepository,
        connection_repo: MemoryConnectionRepository,
        vector_store: QdrantVectorStore,
        config: Optional[PipelineConfig] = None
    ):
        """Initialize the ingestion pipeline with all components."""
        self.memory_extractor = memory_extractor
        self.encoder = encoder
        self.dimension_analyzer = dimension_analyzer
        self.vector_manager = vector_manager
        self.meeting_repo = meeting_repo
        self.memory_repo = memory_repo
        self.connection_repo = connection_repo
        self.vector_store = vector_store
        self.config = config or PipelineConfig()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_memories': 0,
            'total_time_ms': 0.0,
            'stage_times': {}
        }
    
    async def ingest_meeting(
        self,
        meeting: Meeting,
        transcript: str
    ) -> IngestionResult:
        """
        Ingest a complete meeting transcript.
        
        Args:
            meeting: Meeting object
            transcript: Raw transcript text
            
        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        errors = []
        warnings = []
        stage_times = {}
        
        try:
            # Stage 1: Extract memories
            stage_start = time.time()
            memories = await self._extract_memories(
                transcript, meeting, errors, warnings
            )
            stage_times['extraction'] = (time.time() - stage_start) * 1000
            
            if not memories:
                warning = "No memories extracted from transcript"
                warnings.append(warning)
                logger.warning(warning)
                return IngestionResult(
                    meeting_id=meeting.id,
                    memories_extracted=0,
                    memories_stored=0,
                    connections_created=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                    warnings=warnings
                )
            
            # Stage 2: Generate embeddings
            stage_start = time.time()
            embeddings = await self._generate_embeddings(memories, errors)
            stage_times['embedding'] = (time.time() - stage_start) * 1000
            
            # Stage 3: Extract dimensions
            stage_start = time.time()
            dimensions = await self._extract_dimensions(
                memories, meeting, errors
            )
            stage_times['dimensions'] = (time.time() - stage_start) * 1000
            
            # Stage 4: Compose vectors
            stage_start = time.time()
            vectors = self._compose_vectors(embeddings, dimensions, errors)
            stage_times['composition'] = (time.time() - stage_start) * 1000
            
            # Stage 5: Store memories and vectors
            stage_start = time.time()
            stored_count = await self._store_memories_and_vectors(
                memories, vectors, errors
            )
            stage_times['storage'] = (time.time() - stage_start) * 1000
            
            # Stage 6: Create connections
            stage_start = time.time()
            connections_count = await self._create_connections(
                memories, errors
            )
            stage_times['connections'] = (time.time() - stage_start) * 1000
            
            # Update meeting with processing info
            await self._update_meeting_status(meeting, len(memories))
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_stats(len(memories), total_time, stage_times)
            
            # Log stage times
            logger.info(
                f"Pipeline stages for meeting {meeting.id}: "
                f"{', '.join(f'{k}={v:.0f}ms' for k, v in stage_times.items())}"
            )
            
            return IngestionResult(
                meeting_id=meeting.id,
                memories_extracted=len(memories),
                memories_stored=stored_count,
                connections_created=connections_count,
                processing_time_ms=total_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            
            return IngestionResult(
                meeting_id=meeting.id,
                memories_extracted=0,
                memories_stored=0,
                connections_created=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=errors,
                warnings=warnings
            )
    
    async def _extract_memories(
        self,
        transcript: str,
        meeting: Meeting,
        errors: List[str],
        warnings: List[str]
    ) -> List[Memory]:
        """Extract memories from transcript."""
        try:
            # Extract memories
            memories = self.memory_extractor.extract_memories(
                transcript=transcript,
                meeting_id=meeting.id,
                project_id=meeting.project_id,
                meeting_metadata={
                    'participants': meeting.participants,
                    'meeting_type': meeting.meeting_type.value,
                    'duration_ms': meeting.duration_minutes * 60 * 1000 if meeting.duration_minutes else None
                }
            )
            
            # Filter memories
            filtered_memories = []
            for memory in memories:
                # Check length
                if len(memory.content) < self.config.min_memory_length:
                    warnings.append(
                        f"Memory too short ({len(memory.content)} chars): "
                        f"{memory.content[:50]}..."
                    )
                    continue
                
                if len(memory.content) > self.config.max_memory_length:
                    warnings.append(
                        f"Memory too long ({len(memory.content)} chars), truncating"
                    )
                    memory.content = memory.content[:self.config.max_memory_length]
                
                filtered_memories.append(memory)
            
            logger.info(
                f"Extracted {len(filtered_memories)} valid memories "
                f"from {len(memories)} total"
            )
            
            return filtered_memories
            
        except Exception as e:
            error_msg = f"Memory extraction failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return []
    
    async def _generate_embeddings(
        self,
        memories: List[Memory],
        errors: List[str]
    ) -> np.ndarray:
        """Generate embeddings for all memories."""
        try:
            # Extract content for encoding
            contents = [memory.content for memory in memories]
            
            # Generate embeddings in batches
            if self.config.parallel_encoding:
                embeddings = self.encoder.encode_batch(
                    contents,
                    normalize=True
                )
            else:
                embeddings = self.encoder.encode(
                    contents,
                    normalize=True
                )
            
            logger.debug(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Return zero embeddings as fallback
            return np.zeros((len(memories), 384))
    
    async def _extract_dimensions(
        self,
        memories: List[Memory],
        meeting: Meeting,
        errors: List[str]
    ) -> np.ndarray:
        """Extract cognitive dimensions for all memories."""
        try:
            # Prepare contexts
            contexts = []
            meeting_duration_ms = (
                meeting.duration_minutes * 60 * 1000 
                if meeting.duration_minutes else None
            )
            
            for memory in memories:
                context = DimensionExtractionContext(
                    timestamp_ms=memory.timestamp_ms,
                    meeting_duration_ms=meeting_duration_ms,
                    speaker=memory.speaker,
                    speaker_role=memory.speaker_role,
                    content_type=memory.content_type.value,
                    participants=[p.get('name') for p in meeting.participants],
                    meeting_type=meeting.meeting_type.value,
                    project_id=meeting.project_id
                )
                contexts.append(context)
            
            # Extract dimensions
            contents = [memory.content for memory in memories]
            dimensions = await self.dimension_analyzer.batch_analyze(
                contents, contexts
            )
            
            logger.debug(f"Extracted dimensions with shape {dimensions.shape}")
            return dimensions
            
        except Exception as e:
            error_msg = f"Dimension extraction failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Return default dimensions as fallback
            return np.full((len(memories), 16), 0.5)
    
    def _compose_vectors(
        self,
        embeddings: np.ndarray,
        dimensions: np.ndarray,
        errors: List[str]
    ) -> List[Vector]:
        """Compose full 400D vectors."""
        try:
            vectors = self.vector_manager.batch_compose(
                embeddings, dimensions
            )
            
            # Validate vectors
            for i, vector in enumerate(vectors):
                is_valid, error = self.vector_manager.validate_vector(vector)
                if not is_valid:
                    errors.append(f"Invalid vector at index {i}: {error}")
                    # Normalize to fix
                    vectors[i] = self.vector_manager.normalize_vector(vector)
            
            return vectors
            
        except Exception as e:
            error_msg = f"Vector composition failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return []
    
    async def _store_memories_and_vectors(
        self,
        memories: List[Memory],
        vectors: List[Vector],
        errors: List[str]
    ) -> int:
        """Store memories in SQLite and vectors in Qdrant."""
        stored_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(memories), self.config.batch_size):
                batch_memories = memories[i:i + self.config.batch_size]
                batch_vectors = vectors[i:i + self.config.batch_size]
                
                # Store in Qdrant first to get IDs
                qdrant_ids = await self.vector_store.batch_store_memories(
                    batch_memories, batch_vectors
                )
                
                # Update memories with Qdrant IDs
                for memory, qdrant_id in zip(batch_memories, qdrant_ids):
                    memory.qdrant_id = qdrant_id
                    
                    # Store dimensions as JSON
                    idx = memories.index(memory)
                    memory.dimensions_json = self.vector_manager.to_json(
                        vectors[idx]
                    )
                
                # Store in SQLite
                stored = await self.memory_repo.batch_create(batch_memories)
                stored_count += stored
            
            logger.info(f"Stored {stored_count} memories with vectors")
            return stored_count
            
        except Exception as e:
            error_msg = f"Storage failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return stored_count
    
    async def _create_connections(
        self,
        memories: List[Memory],
        errors: List[str]
    ) -> int:
        """Create connections between memories."""
        connections_count = 0
        
        try:
            if self.config.create_sequential_connections and len(memories) > 1:
                # Create sequential connections
                memory_ids = [m.id for m in memories]
                count = await self.connection_repo.create_sequential_connections(
                    memory_ids,
                    self.config.sequential_connection_strength
                )
                connections_count += count
                
                logger.debug(f"Created {count} sequential connections")
            
            # Future: Create other types of connections
            # - Reference connections based on content similarity
            # - Response connections based on speaker patterns
            # - Topic-based connections
            
            return connections_count
            
        except Exception as e:
            error_msg = f"Connection creation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return connections_count
    
    async def _update_meeting_status(
        self,
        meeting: Meeting,
        memory_count: int
    ) -> None:
        """Update meeting with processing status."""
        try:
            await self.meeting_repo.mark_as_processed(
                meeting.id, memory_count
            )
        except Exception as e:
            logger.error(f"Failed to update meeting status: {e}")
    
    def _update_stats(
        self,
        memory_count: int,
        total_time: float,
        stage_times: Dict[str, float]
    ) -> None:
        """Update pipeline statistics."""
        self.stats['total_processed'] += 1
        self.stats['total_memories'] += memory_count
        self.stats['total_time_ms'] += total_time
        
        # Update stage times
        for stage, time_ms in stage_times.items():
            if stage not in self.stats['stage_times']:
                self.stats['stage_times'][stage] = []
            self.stats['stage_times'][stage].append(time_ms)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        stats = {
            'meetings_processed': self.stats['total_processed'],
            'total_memories': self.stats['total_memories'],
            'avg_memories_per_meeting': (
                self.stats['total_memories'] / self.stats['total_processed']
                if self.stats['total_processed'] > 0 else 0
            ),
            'avg_processing_time_ms': (
                self.stats['total_time_ms'] / self.stats['total_processed']
                if self.stats['total_processed'] > 0 else 0
            )
        }
        
        # Add stage statistics
        stage_stats = {}
        for stage, times in self.stats['stage_times'].items():
            if times:
                stage_stats[f"{stage}_avg_ms"] = np.mean(times)
                stage_stats[f"{stage}_max_ms"] = np.max(times)
        
        stats['stages'] = stage_stats
        
        return stats
    
    async def process_batch(
        self,
        meetings_and_transcripts: List[Tuple[Meeting, str]]
    ) -> List[IngestionResult]:
        """
        Process multiple meetings in batch.
        
        Args:
            meetings_and_transcripts: List of (Meeting, transcript) tuples
            
        Returns:
            List of IngestionResults
        """
        results = []
        
        for meeting, transcript in meetings_and_transcripts:
            result = await self.ingest_meeting(meeting, transcript)
            results.append(result)
        
        return results


# Factory function for creating pipeline
async def create_ingestion_pipeline(
    db_connection,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    config: Optional[PipelineConfig] = None
) -> IngestionPipeline:
    """
    Create a configured ingestion pipeline.
    
    Args:
        db_connection: Database connection
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        config: Pipeline configuration
        
    Returns:
        Configured IngestionPipeline
    """
    from ..embedding.onnx_encoder import get_encoder
    from ..embedding.vector_manager import get_vector_manager
    from ..extraction.dimensions.dimension_analyzer import get_dimension_analyzer
    from ..storage.qdrant.vector_store import get_vector_store
    from ..storage.sqlite.repositories import (
        get_meeting_repository,
        get_memory_repository,
        get_memory_connection_repository
    )
    
    # Create components
    memory_extractor = MemoryExtractor()
    encoder = get_encoder()
    dimension_analyzer = get_dimension_analyzer()
    vector_manager = get_vector_manager()
    
    # Create repositories
    meeting_repo = get_meeting_repository(db_connection)
    memory_repo = get_memory_repository(db_connection)
    connection_repo = get_memory_connection_repository(db_connection)
    
    # Create vector store
    vector_store = get_vector_store(qdrant_host, qdrant_port)
    
    # Create pipeline
    pipeline = IngestionPipeline(
        memory_extractor=memory_extractor,
        encoder=encoder,
        dimension_analyzer=dimension_analyzer,
        vector_manager=vector_manager,
        meeting_repo=meeting_repo,
        memory_repo=memory_repo,
        connection_repo=connection_repo,
        vector_store=vector_store,
        config=config
    )
    
    return pipeline


