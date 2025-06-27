"""
Dual memory system implementation with episodic and semantic memory.

This module implements the dual memory system that mirrors human memory
characteristics with fast-decaying episodic memory and slow-decaying semantic
memory, including automatic consolidation mechanisms.

Reference: IMPLEMENTATION_GUIDE.md - Phase 3: Advanced Memory Features
"""

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger

from ...models.entities import Memory, ContentType, Priority
from ...storage.sqlite.repositories import MemoryRepository
from ...storage.sqlite.connection import DatabaseConnection


class MemoryType(Enum):
    """Memory types in the dual memory system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryAccessPattern:
    """Tracks access patterns for memory consolidation."""
    
    memory_id: str
    access_times: List[float] = field(default_factory=list)
    last_consolidation_check: float = 0.0
    consolidation_score: float = 0.0

    def add_access(self, timestamp: float) -> None:
        """Add an access timestamp."""
        self.access_times.append(timestamp)
        # Keep only recent accesses (last 30 days)
        cutoff = timestamp - (30 * 24 * 3600)  # 30 days in seconds
        self.access_times = [t for t in self.access_times if t >= cutoff]

    def calculate_access_frequency(self, window_hours: float = 168.0) -> float:
        """Calculate access frequency within time window (default 1 week)."""
        now = time.time()
        cutoff = now - (window_hours * 3600)
        recent_accesses = [t for t in self.access_times if t >= cutoff]
        return len(recent_accesses) / window_hours if window_hours > 0 else 0.0

    def calculate_recency_score(self) -> float:
        """Calculate recency score (0-1, higher = more recent)."""
        if not self.access_times:
            return 0.0

        now = time.time()
        last_access = max(self.access_times)
        hours_since = (now - last_access) / 3600

        # Exponential decay with half-life of 7 days
        return math.exp(-hours_since / (7 * 24))

    def calculate_consolidation_score(self) -> float:
        """Calculate consolidation score based on access patterns."""
        frequency = self.calculate_access_frequency()
        recency = self.calculate_recency_score()

        # Access distribution (more distributed = higher score)
        if len(self.access_times) < 2:
            distribution = 0.0
        else:
            intervals = []
            sorted_times = sorted(self.access_times)
            for i in range(1, len(sorted_times)):
                intervals.append(sorted_times[i] - sorted_times[i - 1])

            if intervals:
                # Calculate coefficient of variation (std/mean)
                mean_interval = sum(intervals) / len(intervals)
                if mean_interval > 0:
                    std_interval = math.sqrt(
                        sum((x - mean_interval) ** 2 for x in intervals)
                        / len(intervals)
                    )
                    distribution = 1.0 - min(1.0, std_interval / mean_interval)
                else:
                    distribution = 0.0
            else:
                distribution = 0.0

        # Combine factors (frequency 40%, recency 30%, distribution 30%)
        self.consolidation_score = (
            0.4 * min(1.0, frequency) + 0.3 * recency + 0.3 * distribution
        )

        return self.consolidation_score


class ContentTypeDecayProfile:
    """Decay profiles for different content types."""
    
    # Decay multipliers for different content types (lower = slower decay)
    DECAY_PROFILES = {
        ContentType.DECISION: 0.5,      # Decisions decay slowly
        ContentType.ACTION: 0.8,        # Actions decay moderately
        ContentType.COMMITMENT: 0.6,    # Commitments decay slowly
        ContentType.INSIGHT: 0.7,       # Insights decay moderately slowly
        ContentType.DELIVERABLE: 0.5,   # Deliverables decay slowly
        ContentType.MILESTONE: 0.4,     # Milestones decay very slowly
        ContentType.RISK: 0.6,          # Risks decay slowly
        ContentType.ISSUE: 0.9,         # Issues decay faster
        ContentType.HYPOTHESIS: 0.7,    # Hypotheses decay moderately slowly
        ContentType.FINDING: 0.6,       # Findings decay slowly
        ContentType.RECOMMENDATION: 0.5, # Recommendations decay slowly
        ContentType.QUESTION: 1.0,      # Questions decay normally
        ContentType.CONTEXT: 1.0,       # Context decays normally
    }
    
    @classmethod
    def get_decay_multiplier(cls, content_type: ContentType) -> float:
        """Get decay multiplier for content type."""
        return cls.DECAY_PROFILES.get(content_type, 1.0)


class EpisodicMemoryStore:
    """
    Episodic memory store with fast decay and specific experiences.

    Episodic memories represent specific experiences and events with
    fast decay rates, typically persisting for days to weeks.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        base_decay_rate: float = 0.1,
        max_retention_days: int = 30
    ):
        """Initialize episodic memory store."""
        self.memory_repo = memory_repo
        self.base_decay_rate = base_decay_rate
        self.max_retention_days = max_retention_days

    async def store_episodic_memory(self, memory: Memory) -> bool:
        """Store an episodic memory with fast decay parameters."""
        try:
            # Set episodic-specific parameters
            memory.memory_type = MemoryType.EPISODIC.value
            memory.decay_rate = self.base_decay_rate
            memory.importance_score = 0.0  # Initial importance
            
            # Store in database
            success = await self.memory_repo.create(memory)
            
            if success:
                logger.debug(
                    "Episodic memory stored",
                    memory_id=memory.id,
                    content_preview=memory.content[:50] + "..."
                    if len(memory.content) > 50
                    else memory.content,
                )
            
            return success

        except Exception as e:
            logger.error(
                "Failed to store episodic memory", 
                memory_id=memory.id, 
                error=str(e)
            )
            return False

    async def get_episodic_memories(
        self,
        limit: Optional[int] = None,
        min_strength: float = 0.0,
        project_id: Optional[str] = None
    ) -> List[Memory]:
        """Get episodic memories with decay applied."""
        try:
            # Get memories filtered by type
            memories = await self.memory_repo.get_by_memory_type(
                MemoryType.EPISODIC.value,
                project_id=project_id,
                limit=limit
            )
            
            # Apply decay and filter by strength
            active_memories = []
            for memory in memories:
                current_strength = self._calculate_decayed_strength(memory)
                if current_strength >= min_strength:
                    memory.importance_score = current_strength
                    active_memories.append(memory)
            
            # Sort by strength
            active_memories.sort(key=lambda m: m.importance_score, reverse=True)
            
            return active_memories[:limit] if limit else active_memories

        except Exception as e:
            logger.error("Failed to get episodic memories", error=str(e))
            return []

    def _calculate_decayed_strength(self, memory: Memory) -> float:
        """Calculate current strength after decay."""
        now = datetime.now()
        hours_elapsed = (now - memory.timestamp).total_seconds() / 3600
        
        # Get content-type specific decay rate
        decay_multiplier = ContentTypeDecayProfile.get_decay_multiplier(
            memory.content_type
        )
        effective_decay_rate = self.base_decay_rate * decay_multiplier
        
        # Apply priority-based decay modification
        if memory.priority == Priority.CRITICAL:
            effective_decay_rate *= 0.5  # Critical items decay 50% slower
        elif memory.priority == Priority.HIGH:
            effective_decay_rate *= 0.7  # High priority items decay 30% slower
        
        # Exponential decay: strength = initial * exp(-decay_rate * time)
        decayed_strength = 1.0 * math.exp(-effective_decay_rate * hours_elapsed / 24)
        
        return max(0.0, min(1.0, decayed_strength))

    async def cleanup_expired_memories(self) -> int:
        """Remove episodic memories that have decayed below threshold."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_retention_days)
            
            # Get old memories
            old_memories = await self.memory_repo.get_memories_before_date(
                cutoff_date,
                memory_type=MemoryType.EPISODIC.value
            )
            
            deleted_count = 0
            for memory in old_memories:
                if self._calculate_decayed_strength(memory) < 0.01:
                    if await self.memory_repo.delete(memory.id):
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired episodic memories")
            
            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup expired memories", error=str(e))
            return 0


class SemanticMemoryStore:
    """
    Semantic memory store with slow decay and generalized knowledge.

    Semantic memories represent generalized patterns and knowledge with
    slow decay rates, typically persisting for months to years.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        base_decay_rate: float = 0.01,
        min_consolidation_score: float = 0.6
    ):
        """Initialize semantic memory store."""
        self.memory_repo = memory_repo
        self.base_decay_rate = base_decay_rate
        self.min_consolidation_score = min_consolidation_score

    async def store_semantic_memory(self, memory: Memory) -> bool:
        """Store a semantic memory with slow decay parameters."""
        try:
            # Set semantic-specific parameters
            memory.memory_type = MemoryType.SEMANTIC.value
            memory.decay_rate = self.base_decay_rate
            memory.importance_score = memory.importance_score or 0.8  # Higher default
            
            # Store in database
            success = await self.memory_repo.create(memory)
            
            if success:
                logger.debug(
                    "Semantic memory stored",
                    memory_id=memory.id,
                    content_preview=memory.content[:50] + "..."
                    if len(memory.content) > 50
                    else memory.content,
                )
            
            return success

        except Exception as e:
            logger.error(
                "Failed to store semantic memory", 
                memory_id=memory.id, 
                error=str(e)
            )
            return False

    async def get_semantic_memories(
        self,
        limit: Optional[int] = None,
        min_strength: float = 0.0,
        project_id: Optional[str] = None
    ) -> List[Memory]:
        """Get semantic memories with slow decay applied."""
        try:
            # Get memories filtered by type
            memories = await self.memory_repo.get_by_memory_type(
                MemoryType.SEMANTIC.value,
                project_id=project_id,
                limit=limit
            )
            
            # Apply slow decay and filter by strength
            active_memories = []
            for memory in memories:
                current_strength = self._calculate_decayed_strength(memory)
                if current_strength >= min_strength:
                    memory.importance_score = current_strength
                    active_memories.append(memory)
            
            # Sort by importance and access count
            active_memories.sort(
                key=lambda m: (m.importance_score, m.access_count), 
                reverse=True
            )
            
            return active_memories[:limit] if limit else active_memories

        except Exception as e:
            logger.error("Failed to get semantic memories", error=str(e))
            return []

    def _calculate_decayed_strength(self, memory: Memory) -> float:
        """Calculate current strength after slow decay."""
        now = datetime.now()
        days_elapsed = (now - memory.timestamp).total_seconds() / (24 * 3600)
        
        # Get content-type specific decay rate
        decay_multiplier = ContentTypeDecayProfile.get_decay_multiplier(
            memory.content_type
        )
        effective_decay_rate = self.base_decay_rate * decay_multiplier
        
        # Very slow exponential decay
        decayed_strength = memory.importance_score * math.exp(
            -effective_decay_rate * days_elapsed / 30
        )
        
        return max(0.0, min(1.0, decayed_strength))


class MemoryConsolidation:
    """
    Handles consolidation of episodic memories to semantic memories.

    Implements automatic consolidation based on access patterns,
    importance scoring, and time-based criteria.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        episodic_store: EpisodicMemoryStore,
        semantic_store: SemanticMemoryStore
    ):
        """Initialize memory consolidation system."""
        self.memory_repo = memory_repo
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.access_patterns: Dict[str, MemoryAccessPattern] = {}
        self.consolidation_threshold = 0.6
        self.min_accesses_for_consolidation = 3
        self.consolidation_cooldown_hours = 24

    def track_memory_access(
        self, memory_id: str, timestamp: Optional[float] = None
    ) -> None:
        """Track memory access for consolidation scoring."""
        if timestamp is None:
            timestamp = time.time()

        if memory_id not in self.access_patterns:
            self.access_patterns[memory_id] = MemoryAccessPattern(memory_id)

        self.access_patterns[memory_id].add_access(timestamp)

    async def identify_consolidation_candidates(self) -> List[Tuple[str, float]]:
        """Identify episodic memories ready for consolidation."""
        candidates = []

        try:
            # Get episodic memories with sufficient access
            episodic_memories = await self.memory_repo.get_by_memory_type(
                MemoryType.EPISODIC.value,
                min_access_count=self.min_accesses_for_consolidation
            )
            
            for memory in episodic_memories:
                # Check cooldown period
                if memory.id in self.access_patterns:
                    pattern = self.access_patterns[memory.id]
                    if (time.time() - pattern.last_consolidation_check < 
                        self.consolidation_cooldown_hours * 3600):
                        continue
                    
                    # Calculate consolidation score
                    score = pattern.calculate_consolidation_score()
                else:
                    # Fallback scoring based on database data
                    age_days = (datetime.now() - memory.timestamp).days
                    access_rate = memory.access_count / max(1, age_days)
                    score = min(1.0, access_rate * 0.5)
                
                if score >= self.consolidation_threshold:
                    candidates.append((memory.id, score))
                
                # Update last check time
                if memory.id in self.access_patterns:
                    self.access_patterns[memory.id].last_consolidation_check = time.time()

            # Sort by consolidation score
            candidates.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Found {len(candidates)} consolidation candidates")
            return candidates

        except Exception as e:
            logger.error("Failed to identify consolidation candidates", error=str(e))
            return []

    async def consolidate_memory(self, memory_id: str) -> bool:
        """Consolidate an episodic memory to semantic memory."""
        try:
            # Get the episodic memory
            episodic_memory = await self.memory_repo.get_by_id(memory_id)
            if not episodic_memory or episodic_memory.memory_type != MemoryType.EPISODIC.value:
                logger.warning(
                    "Memory not found or not episodic for consolidation", 
                    memory_id=memory_id
                )
                return False

            # Create semantic version
            semantic_memory = Memory(
                id=f"{memory_id}_semantic",
                meeting_id=episodic_memory.meeting_id,
                project_id=episodic_memory.project_id,
                content=episodic_memory.content,
                speaker=episodic_memory.speaker,
                speaker_role=episodic_memory.speaker_role,
                timestamp=datetime.now(),  # New timestamp for semantic memory
                memory_type=MemoryType.SEMANTIC.value,
                content_type=episodic_memory.content_type,
                priority=episodic_memory.priority,
                level=max(0, episodic_memory.level - 1),  # Move up hierarchy
                qdrant_id=f"{episodic_memory.qdrant_id}_semantic",
                dimensions_json=episodic_memory.dimensions_json,
                importance_score=self.access_patterns.get(
                    memory_id, 
                    MemoryAccessPattern(memory_id)
                ).calculate_consolidation_score(),
                access_count=episodic_memory.access_count
            )
            
            # Store semantic memory
            success = await self.semantic_store.store_semantic_memory(semantic_memory)
            
            if success:
                # Mark original as consolidated
                episodic_memory.metadata = episodic_memory.metadata or {}
                episodic_memory.metadata["consolidated_to"] = semantic_memory.id
                episodic_memory.metadata["consolidation_date"] = datetime.now().isoformat()
                await self.memory_repo.update(episodic_memory)
                
                logger.info(
                    "Memory consolidated successfully",
                    episodic_id=memory_id,
                    semantic_id=semantic_memory.id,
                    importance_score=semantic_memory.importance_score,
                )
            
            return success

        except Exception as e:
            logger.error(
                "Failed to consolidate memory", 
                memory_id=memory_id, 
                error=str(e)
            )
            return False

    async def run_consolidation_cycle(self) -> Dict[str, int]:
        """Run a complete consolidation cycle."""
        stats = {
            "candidates_identified": 0, 
            "memories_consolidated": 0, 
            "errors": 0
        }

        try:
            # Identify candidates
            candidates = await self.identify_consolidation_candidates()
            stats["candidates_identified"] = len(candidates)

            # Consolidate memories
            for memory_id, score in candidates:
                try:
                    if await self.consolidate_memory(memory_id):
                        stats["memories_consolidated"] += 1
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(
                        "Error consolidating memory", 
                        memory_id=memory_id, 
                        error=str(e)
                    )
                    stats["errors"] += 1

            logger.info("Consolidation cycle completed", stats=stats)

        except Exception as e:
            logger.error("Consolidation cycle failed", error=str(e))
            stats["errors"] += 1

        return stats


class DualMemorySystem:
    """
    Complete dual memory system combining episodic and semantic stores.

    Provides unified interface for storing, retrieving, and managing
    both episodic and semantic memories with automatic consolidation.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        episodic_decay_rate: float = 0.1,
        semantic_decay_rate: float = 0.01,
        max_episodic_retention_days: int = 30
    ):
        """Initialize dual memory system."""
        self.memory_repo = memory_repo
        
        # Initialize memory stores
        self.episodic_store = EpisodicMemoryStore(
            memory_repo,
            base_decay_rate=episodic_decay_rate,
            max_retention_days=max_episodic_retention_days
        )
        self.semantic_store = SemanticMemoryStore(
            memory_repo,
            base_decay_rate=semantic_decay_rate
        )
        
        # Initialize consolidation system
        self.consolidation = MemoryConsolidation(
            memory_repo,
            self.episodic_store,
            self.semantic_store
        )

    async def store_experience(self, memory: Memory) -> bool:
        """Store a new experience as episodic memory."""
        return await self.episodic_store.store_episodic_memory(memory)

    async def store_knowledge(self, memory: Memory) -> bool:
        """Store generalized knowledge as semantic memory."""
        return await self.semantic_store.store_semantic_memory(memory)

    async def retrieve_memories(
        self,
        memory_types: Optional[List[MemoryType]] = None,
        limit: Optional[int] = None,
        min_strength: float = 0.0,
        project_id: Optional[str] = None
    ) -> Dict[MemoryType, List[Memory]]:
        """Retrieve memories from both stores."""
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]

        results = {}

        if MemoryType.EPISODIC in memory_types:
            results[MemoryType.EPISODIC] = await self.episodic_store.get_episodic_memories(
                limit=limit, 
                min_strength=min_strength,
                project_id=project_id
            )

        if MemoryType.SEMANTIC in memory_types:
            results[MemoryType.SEMANTIC] = await self.semantic_store.get_semantic_memories(
                limit=limit, 
                min_strength=min_strength,
                project_id=project_id
            )

        return results

    async def access_memory(self, memory_id: str) -> Optional[Memory]:
        """Access a memory and track the access for consolidation."""
        self.consolidation.track_memory_access(memory_id)
        
        memory = await self.memory_repo.get_by_id(memory_id)
        if memory:
            # Update access count
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            await self.memory_repo.update(memory)
        
        return memory

    async def consolidate_memories(self) -> Dict[str, int]:
        """Trigger memory consolidation cycle."""
        return await self.consolidation.run_consolidation_cycle()

    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories from both stores."""
        episodic_cleaned = await self.episodic_store.cleanup_expired_memories()
        
        return {
            "episodic_cleaned": episodic_cleaned,
            "semantic_cleaned": 0,  # Semantic memories don't expire automatically
        }

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            stats = {}
            
            # Get counts by type
            for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
                memories = await self.memory_repo.get_by_memory_type(memory_type.value)
                stats[memory_type.value] = {
                    "total_memories": len(memories),
                    "average_access_count": sum(m.access_count for m in memories) / len(memories) if memories else 0,
                    "max_access_count": max((m.access_count for m in memories), default=0)
                }
            
            # Consolidation statistics
            stats["consolidation"] = {
                "tracked_patterns": len(self.consolidation.access_patterns),
                "candidates_above_threshold": len([
                    p for p in self.consolidation.access_patterns.values()
                    if p.calculate_consolidation_score() >= self.consolidation.consolidation_threshold
                ])
            }
            
            return stats

        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"error": str(e)}
