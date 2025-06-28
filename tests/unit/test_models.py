"""
Unit tests for the core data models in src.models.entities.

Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
"""

import unittest
import uuid
from datetime import datetime
import numpy as np

# Assuming these are the primary entities based on the implementation guide.
# If the actual file has different structures, these tests will fail and
# will need to be adjusted.
from src.models.entities import Memory, Vector, MemoryType, ContentType, Priority

class TestCoreModels(unittest.TestCase):

    def test_memory_creation_defaults(self):
        """
        Test that a Memory object can be created with default values.
        """
        now = datetime.now()
        memory = Memory(
            meeting_id="test_meeting",
            project_id="test_project",
            content="This is a test memory.",
            created_at=now
        )
        self.assertIsNotNone(memory.id)
        self.assertIsInstance(uuid.UUID(memory.id), uuid.UUID)
        self.assertEqual(memory.meeting_id, "test_meeting")
        self.assertEqual(memory.project_id, "test_project")
        self.assertEqual(memory.content, "This is a test memory.")
        self.assertEqual(memory.memory_type, MemoryType.EPISODIC) # Assuming EPISODIC is a default
        self.assertEqual(memory.content_type, ContentType.CONTEXT) # Assuming CONTEXT is a default
        self.assertEqual(memory.level, 2) # L2 is the default for new memories
        self.assertIsNone(memory.priority)
        self.assertEqual(memory.created_at, now)

    def test_memory_creation_full(self):
        """
        Test that a Memory object can be created with all fields specified.
        """
        now = datetime.now()
        q_id = str(uuid.uuid4())
        memory = Memory(
            id="test_id",
            meeting_id="meeting_123",
            project_id="project_abc",
            content="An important decision was made.",
            speaker="Alice",
            speaker_role="Manager",
            timestamp_ms=123456789,
            memory_type=MemoryType.SEMANTIC,
            content_type=ContentType.DECISION,
            level=1,
            importance_score=0.9,
            priority=Priority.HIGH,
            qdrant_id=q_id,
            created_at=now,
            updated_at=now
        )
        self.assertEqual(memory.id, "test_id")
        self.assertEqual(memory.speaker, "Alice")
        self.assertEqual(memory.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(memory.content_type, ContentType.DECISION)
        self.assertEqual(memory.level, 1)
        self.assertEqual(memory.importance_score, 0.9)
        self.assertEqual(memory.priority, Priority.HIGH)
        self.assertEqual(memory.qdrant_id, q_id)

    def test_vector_creation(self):
        """
        Test the creation and properties of a Vector object.
        """
        semantic = np.random.rand(384).astype(np.float32)
        cognitive = np.random.rand(16).astype(np.float32)
        
        vector = Vector(semantic=semantic, dimensions=cognitive)
        
        self.assertEqual(vector.semantic.shape, (384,))
        self.assertEqual(vector.dimensions.shape, (16,))
        
        # Test the full_vector property
        full_vector = vector.full_vector
        self.assertEqual(full_vector.shape, (400,))
        np.testing.assert_array_equal(full_vector[:384], semantic)
        np.testing.assert_array_equal(full_vector[384:], cognitive)

    def test_vector_from_list(self):
        """
        Test creating a Vector from a flat list or array.
        """
        full_vector_list = np.random.rand(400).astype(np.float32).tolist()
        vector = Vector.from_list(full_vector_list)

        self.assertEqual(vector.semantic.shape, (384,))
        self.assertEqual(vector.dimensions.shape, (16,))
        np.testing.assert_array_equal(vector.full_vector, full_vector_list)

    def test_enums(self):
        """
        Test that the enums have the expected values.
        """
        self.assertIsNotNone(MemoryType.EPISODIC)
        self.assertIsNotNone(MemoryType.SEMANTIC)
        self.assertIsNotNone(ContentType.DECISION)
        self.assertIsNotNone(ContentType.ACTION)
        self.assertIsNotNone(ContentType.CONTEXT)
        self.assertIsNotNone(Priority.HIGH)
        self.assertIsNotNone(Priority.MEDIUM)
        self.assertIsNotNone(Priority.LOW)

if __name__ == '__main__':
    unittest.main()