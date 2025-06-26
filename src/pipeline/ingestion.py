"""
Ingestion pipeline orchestrating the full memory extraction process.

Reference: IMPLEMENTATION_GUIDE.md - Day 5: Extraction Pipeline
Coordinates extraction, embedding, dimension analysis, and storage.
"""

from typing import List, Dict, Optional
import asyncio
import logging
from pathlib import Path

from src.models.entities import Memory, Meeting, MemoryConnection, ConnectionType
from src.extraction.memory_extractor import MemoryExtractor
from src.extraction.dimensions.analyzer import DimensionAnalyzer
from src.embedding.onnx_encoder import get_encoder
from src.embedding.vector_manager import VectorManager
from src.storage.sqlite.repositories import MemoryRepository, MeetingRepository, ConnectionRepository
from src.storage.qdrant.vector_store import get_vector_store
from src.storage.sqlite.connection import get_db

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the complete ingestion process.
    
    TODO Day 5:
    - [ ] Implement full pipeline flow
    - [ ] Add error handling and recovery
    - [ ] Implement batch processing
    - [ ] Add progress tracking
    - [ ] Create sequential connections
    - [ ] Target: <2s for 1-hour transcript
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.memory_extractor = MemoryExtractor()
        self.dimension_analyzer = DimensionAnalyzer()
        self.encoder = get_encoder()
        self.vector_store = get_vector_store()
        
        # Repositories
        db = get_db()
        self.memory_repo = MemoryRepository(db)
        self.meeting_repo = MeetingRepository(db)
        self.connection_repo = ConnectionRepository(db)
        
        logger.info("IngestionPipeline initialized")
    
    async def ingest_meeting(self, meeting: Meeting, transcript_path: str) -> Dict:
        """
        Ingest a complete meeting transcript.
        
        Args:
            meeting: Meeting object
            transcript_path: Path to transcript file
            
        Returns:
            Dict with ingestion statistics
            
        TODO Day 5:
        - [ ] Load transcript
        - [ ] Extract memories
        - [ ] Generate embeddings
        - [ ] Extract dimensions
        - [ ] Store in databases
        - [ ] Create connections
        """
        stats = {
            "meeting_id": meeting.id,
            "memories_extracted": 0,
            "vectors_stored": 0,
            "connections_created": 0,
            "errors": []
        }
        
        try:
            # TODO Day 5: Load transcript
            transcript = await self._load_transcript(transcript_path)
            
            # TODO Day 5: Extract memories
            memories = self.memory_extractor.extract_memories(transcript, meeting.id)
            stats["memories_extracted"] = len(memories)
            
            # TODO Day 5: Process memories in batches
            batch_size = 50
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                await self._process_memory_batch(batch, stats)
            
            # TODO Day 5: Create sequential connections
            connections_created = await self._create_connections(memories)
            stats["connections_created"] = connections_created
            
            # TODO Day 5: Update meeting status
            await self.meeting_repo.mark_processed(meeting.id, len(memories))
            
            logger.info(f"Ingestion complete: {stats}")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            stats["errors"].append(str(e))
        
        return stats
    
    async def _load_transcript(self, transcript_path: str) -> str:
        """
        Load transcript from file.
        
        TODO Day 5:
        - [ ] Read file content
        - [ ] Handle different formats
        - [ ] Validate content
        """
        path = Path(transcript_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        # TODO Day 5: Handle async file reading
        return path.read_text(encoding='utf-8')
    
    async def _process_memory_batch(self, memories: List[Memory], stats: Dict) -> None:
        """
        Process a batch of memories.
        
        TODO Day 5:
        - [ ] Generate embeddings
        - [ ] Extract dimensions
        - [ ] Compose vectors
        - [ ] Store in SQLite
        - [ ] Store in Qdrant
        """
        # TODO Day 5: Extract texts for batch encoding
        texts = [m.content for m in memories]
        
        # TODO Day 5: Generate embeddings
        embeddings = await self.encoder.encode_batch(texts)
        
        # TODO Day 5: Extract dimensions for each
        all_dimensions = []
        for memory in memories:
            context = {
                "speaker": memory.speaker,
                "timestamp_ms": memory.timestamp_ms,
                "memory_type": memory.memory_type.value
            }
            dimensions = self.dimension_analyzer.analyze(memory.content, context)
            all_dimensions.append(dimensions)
        
        # TODO Day 5: Process each memory
        for i, memory in enumerate(memories):
            try:
                # Compose vector
                full_vector = VectorManager.compose_vector(
                    embeddings[i],
                    {"all": all_dimensions[i]}  # TODO: Proper dimension dict
                )
                
                # Store in SQLite
                memory_id = await self.memory_repo.create(memory)
                
                # Store in Qdrant
                payload = {
                    "memory_id": memory_id,
                    "memory_type": memory.memory_type.value,
                    "speaker": memory.speaker,
                    "timestamp_ms": memory.timestamp_ms
                }
                
                collection = f"cognitive_{['concepts', 'contexts', 'episodes'][memory.level]}"
                success = await self.vector_store.store_vector(
                    collection,
                    memory_id,
                    full_vector,
                    payload
                )
                
                if success:
                    stats["vectors_stored"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process memory: {e}")
                stats["errors"].append(f"Memory processing: {str(e)}")
    
    async def _create_connections(self, memories: List[Memory]) -> int:
        """
        Create sequential connections between memories.
        
        TODO Day 5:
        - [ ] Connect sequential memories
        - [ ] Set connection strength by time gap
        - [ ] Store in database
        """
        connections_created = 0
        
        # TODO Day 5: Create sequential connections
        for i in range(len(memories) - 1):
            current = memories[i]
            next_mem = memories[i + 1]
            
            # Calculate connection strength based on time gap
            time_gap = abs(next_mem.timestamp_ms - current.timestamp_ms)
            strength = max(0.3, 1.0 - (time_gap / 60000))  # Decay over minutes
            
            connection = MemoryConnection(
                source_id=current.id,
                target_id=next_mem.id,
                connection_strength=strength,
                connection_type=ConnectionType.SEQUENTIAL
            )
            
            if await self.connection_repo.create(connection):
                connections_created += 1
        
        return connections_created
    
    async def ingest_batch(self, meetings: List[Meeting]) -> List[Dict]:
        """
        Ingest multiple meetings.
        
        TODO Day 5:
        - [ ] Process meetings in parallel
        - [ ] Aggregate statistics
        """
        tasks = []
        for meeting in meetings:
            if meeting.transcript_path:
                task = self.ingest_meeting(meeting, meeting.transcript_path)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results