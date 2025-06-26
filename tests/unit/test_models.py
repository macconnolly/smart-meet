"""
Unit tests for data models.

Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
"""

import pytest
from datetime import datetime
import numpy as np

from src.models.entities import (
    Memory, Meeting, MemoryConnection, Vector,
    MemoryType, ContentType, ConnectionType
)


class TestMemory:
    """Test Memory model."""
    
    def test_memory_creation(self):
        """Test creating a memory with defaults."""
        memory = Memory(
            meeting_id="test-meeting-123",
            content="We decided to implement caching"
        )
        
        assert memory.id is not None
        assert memory.meeting_id == "test-meeting-123"
        assert memory.content == "We decided to implement caching"
        assert memory.memory_type == MemoryType.CONTEXT
        assert memory.level == 2
        assert memory.importance_score == 0.5
        
    def test_memory_type_enum(self):
        """Test memory type enumeration."""
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.ACTION.value == "action"
        assert MemoryType.IDEA.value == "idea"
        
    # TODO Day 1: Add tests for to_dict and from_dict methods


class TestMeeting:
    """Test Meeting model."""
    
    def test_meeting_creation(self):
        """Test creating a meeting."""
        start_time = datetime.now()
        meeting = Meeting(
            title="Sprint Planning",
            start_time=start_time,
            participants=["Alice", "Bob", "Charlie"]
        )
        
        assert meeting.id is not None
        assert meeting.title == "Sprint Planning"
        assert len(meeting.participants) == 3
        assert meeting.memory_count == 0
        
    def test_duration_calculation(self):
        """Test duration property."""
        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 11, 30)
        
        meeting = Meeting(
            title="Test Meeting",
            start_time=start,
            end_time=end
        )
        
        assert meeting.duration_minutes == 90


class TestVector:
    """Test Vector model."""
    
    def test_vector_creation(self):
        """Test creating a vector."""
        semantic = np.random.rand(384)
        dimensions = np.random.rand(16)
        
        vector = Vector(semantic=semantic, dimensions=dimensions)
        
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        
    # TODO Day 3: Add tests for full_vector and normalize methods


# TODO Day 1: Add more comprehensive tests