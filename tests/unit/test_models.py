"""
Unit tests for data models.

Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
Tests all dataclasses including consulting-specific enhancements.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import json

from src.models.entities import (
    # Core models
    Memory,
    Meeting,
    MemoryConnection,
    Vector,
    SearchResult,
    Project,
    Stakeholder,
    Deliverable,
    MeetingSeries,
    # Enums
    MemoryType,
    ContentType,
    ConnectionType,
    Priority,
    Status,
    ProjectType,
    ProjectStatus,
    MeetingType,
    MeetingCategory,
    DeliverableType,
    DeliverableStatus,
    StakeholderType,
    InfluenceLevel,
    EngagementLevel,
    MeetingFrequency,
)


class TestEnums:
    """Test all enum definitions."""

    def test_memory_type_enum(self):
        """Test memory type enumeration."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_content_type_enum(self):
        """Test enhanced content type enumeration."""
        assert ContentType.DECISION.value == "decision"
        assert ContentType.ACTION.value == "action"
        assert ContentType.DELIVERABLE.value == "deliverable"
        assert ContentType.RISK.value == "risk"
        assert ContentType.HYPOTHESIS.value == "hypothesis"
        assert len(ContentType) == 15  # All content types

    def test_priority_enum(self):
        """Test priority levels."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"


class TestProject:
    """Test Project model."""

    def test_project_creation_defaults(self):
        """Test creating a project with minimal fields."""
        project = Project(name="Test Project", client_name="Test Client")

        assert project.id is not None
        assert project.name == "Test Project"
        assert project.client_name == "Test Client"
        assert project.project_type == ProjectType.OTHER
        assert project.status == ProjectStatus.ACTIVE
        assert project.consumed_hours == 0
        assert isinstance(project.metadata, dict)

    def test_project_full_creation(self):
        """Test creating a project with all fields."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=90)

        project = Project(
            id="proj_test",
            name="Digital Transformation",
            client_name="Acme Corp",
            project_type=ProjectType.TRANSFORMATION,
            status=ProjectStatus.ACTIVE,
            start_date=start_date,
            end_date=end_date,
            project_manager="John Smith",
            engagement_code="ACME-2024-001",
            budget_hours=1000,
            consumed_hours=250,
            metadata={"region": "EMEA", "industry": "retail"},
        )

        assert project.engagement_code == "ACME-2024-001"
        assert project.budget_hours == 1000
        assert project.consumed_hours == 250
        assert project.metadata["industry"] == "retail"

    def test_project_serialization(self):
        """Test project to_dict and from_dict methods."""
        project = Project(
            name="Test Project",
            client_name="Test Client",
            project_type=ProjectType.STRATEGY,
            metadata={"key": "value"},
        )

        # Test to_dict
        project_dict = project.to_dict()
        assert project_dict["name"] == "Test Project"
        assert project_dict["project_type"] == "strategy"
        assert "metadata_json" in project_dict

        # Test from_dict
        project2 = Project.from_dict(project_dict)
        assert project2.name == project.name
        assert project2.project_type == project.project_type
        assert project2.metadata == project.metadata


class TestStakeholder:
    """Test Stakeholder model."""

    def test_stakeholder_creation(self):
        """Test creating a stakeholder."""
        stakeholder = Stakeholder(
            project_id="proj_001",
            name="Jane Doe",
            organization="Client Corp",
            role="VP Engineering",
        )

        assert stakeholder.id is not None
        assert stakeholder.name == "Jane Doe"
        assert stakeholder.stakeholder_type == StakeholderType.OTHER
        assert stakeholder.influence_level == InfluenceLevel.MEDIUM
        assert stakeholder.engagement_level == EngagementLevel.NEUTRAL

    def test_stakeholder_serialization(self):
        """Test stakeholder serialization."""
        stakeholder = Stakeholder(
            project_id="proj_001",
            name="John CEO",
            organization="Acme",
            role="CEO",
            stakeholder_type=StakeholderType.CLIENT_SPONSOR,
            influence_level=InfluenceLevel.HIGH,
            engagement_level=EngagementLevel.CHAMPION,
            email="john@acme.com",
            notes="Key decision maker",
        )

        # Test to_dict
        data = stakeholder.to_dict()
        assert data["stakeholder_type"] == "client_sponsor"
        assert data["influence_level"] == "high"

        # Test from_dict
        stakeholder2 = Stakeholder.from_dict(data)
        assert stakeholder2.email == "john@acme.com"
        assert stakeholder2.notes == "Key decision maker"


class TestDeliverable:
    """Test Deliverable model."""

    def test_deliverable_creation(self):
        """Test creating a deliverable."""
        due_date = datetime.now() + timedelta(days=30)

        deliverable = Deliverable(
            project_id="proj_001",
            name="Strategy Presentation",
            owner="Sarah Consultant",
            due_date=due_date,
        )

        assert deliverable.id is not None
        assert deliverable.deliverable_type == DeliverableType.OTHER
        assert deliverable.status == DeliverableStatus.PLANNED
        assert deliverable.version == 1.0

    def test_deliverable_with_criteria(self):
        """Test deliverable with acceptance criteria."""
        deliverable = Deliverable(
            project_id="proj_001",
            name="Market Analysis",
            deliverable_type=DeliverableType.ANALYSIS,
            owner="Team",
            acceptance_criteria={
                "depth": "Analyze top 10 competitors",
                "metrics": "Include market share data",
                "format": "PowerPoint presentation",
            },
            dependencies=["market_data", "interviews"],
        )

        assert len(deliverable.acceptance_criteria) == 3
        assert len(deliverable.dependencies) == 2
        assert deliverable.acceptance_criteria["depth"] == "Analyze top 10 competitors"


class TestMemory:
    """Test enhanced Memory model."""

    def test_memory_creation_minimal(self):
        """Test creating a memory with minimal fields."""
        memory = Memory(
            meeting_id="meet_001", project_id="proj_001", content="Important decision made"
        )

        assert memory.id is not None
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.content_type == ContentType.CONTEXT
        assert memory.level == 2
        assert memory.decay_rate == 0.1

    def test_memory_consulting_fields(self):
        """Test memory with consulting-specific fields."""
        due_date = datetime.now() + timedelta(days=7)

        memory = Memory(
            meeting_id="meet_001",
            project_id="proj_001",
            content="Complete market analysis by Friday",
            speaker="Project Manager",
            speaker_role="consultant",
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.ACTION,
            priority=Priority.HIGH,
            status=Status.IN_PROGRESS,
            due_date=due_date,
            owner="Analyst Team",
            deliverable_id="deliv_001",
        )

        assert memory.priority == Priority.HIGH
        assert memory.status == Status.IN_PROGRESS
        assert memory.owner == "Analyst Team"
        assert memory.deliverable_id == "deliv_001"

    def test_memory_serialization(self):
        """Test memory serialization with all fields."""
        memory = Memory(
            meeting_id="meet_001",
            project_id="proj_001",
            content="Test memory",
            timestamp_ms=1500000,
            qdrant_id="qdrant_test",
            raw_dimensions_json=json.dumps({"temporal": [0.5] * 4}),
        )

        # Test to_dict
        data = memory.to_dict()
        assert data["timestamp"] == 1500.0  # Converted to seconds
        assert data["memory_type"] == "episodic"
        assert data["raw_dimensions_json"] is not None

        # Test from_dict
        memory2 = Memory.from_dict(data)
        assert memory2.timestamp_ms == 1500000  # Converted back to ms
        assert memory2.content == "Test memory"

    def test_memory_with_cognitive_dimensions(self):
        """Test memory with CognitiveDimensions object."""
        from src.extraction.dimensions.dimension_analyzer import CognitiveDimensions, TemporalFeatures, EmotionalFeatures, SocialFeatures, CausalFeatures, EvolutionaryFeatures

        # Create a sample CognitiveDimensions object with non-default values for enhanced dimensions
        cognitive_dims = CognitiveDimensions(
            temporal=TemporalFeatures(urgency=0.8, deadline_proximity=0.7, sequence_position=0.5, duration_relevance=0.6),
            emotional=EmotionalFeatures(polarity=0.9, intensity=0.8, confidence=0.7),
            social=SocialFeatures(authority=0.9, influence=0.8, team_dynamics=0.7), # Enhanced values
            causal=CausalFeatures(dependencies=0.8, impact=0.9, risk_factors=0.7), # Enhanced values
            evolutionary=EvolutionaryFeatures(change_rate=0.8, innovation_level=0.9, adaptation_need=0.7) # Enhanced values
        )

        memory = Memory(
            meeting_id="meet_001",
            project_id="proj_001",
            content="Test memory with full dimensions",
            cognitive_dimensions=cognitive_dims,
        )

        # Test to_dict
        data = memory.to_dict()
        assert "raw_dimensions_json" in data
        assert data["raw_dimensions_json"] is not None

        # Verify the content of raw_dimensions_json
        parsed_dims = json.loads(data["raw_dimensions_json"])
        assert parsed_dims["temporal"] == cognitive_dims.temporal.to_array().tolist()
        assert parsed_dims["social"] == cognitive_dims.social.to_array().tolist()
        assert parsed_dims["causal"] == cognitive_dims.causal.to_array().tolist()
        assert parsed_dims["evolutionary"] == cognitive_dims.evolutionary.to_array().tolist()

        # Test from_dict
        memory2 = Memory.from_dict(data)
        assert memory2.cognitive_dimensions is not None
        assert memory2.cognitive_dimensions.temporal.urgency == cognitive_dims.temporal.urgency
        assert memory2.cognitive_dimensions.social.authority == cognitive_dims.social.authority
        assert memory2.cognitive_dimensions.causal.impact == cognitive_dims.causal.impact
        assert memory2.cognitive_dimensions.evolutionary.innovation_level == cognitive_dims.evolutionary.innovation_level

        # Test with no cognitive dimensions
        memory_no_dims = Memory(
            meeting_id="meet_002",
            project_id="proj_002",
            content="Memory without dimensions",
        )
        data_no_dims = memory_no_dims.to_dict()
        assert data_no_dims["raw_dimensions_json"] is None
        memory_no_dims2 = Memory.from_dict(data_no_dims)
        assert memory_no_dims2.cognitive_dimensions is None


class TestMemoryConnection:
    """Test MemoryConnection model."""

    def test_connection_creation(self):
        """Test creating a memory connection."""
        conn = MemoryConnection(
            source_id="mem_001",
            target_id="mem_002",
            connection_strength=0.75,
            connection_type=ConnectionType.SUPPORTS,
        )

        assert conn.connection_strength == 0.75
        assert conn.connection_type == ConnectionType.SUPPORTS
        assert conn.activation_count == 0

    def test_connection_strength_validation(self):
        """Test connection strength validation."""
        with pytest.raises(ValueError):
            MemoryConnection(
                source_id="mem_001", target_id="mem_002", connection_strength=1.5  # Invalid: > 1
            )

        with pytest.raises(ValueError):
            MemoryConnection(
                source_id="mem_001", target_id="mem_002", connection_strength=-0.1  # Invalid: < 0
            )

    def test_update_activation(self):
        """Test activation update method."""
        conn = MemoryConnection(source_id="mem_001", target_id="mem_002")

        assert conn.activation_count == 0
        assert conn.last_activated is None

        conn.update_activation()

        assert conn.activation_count == 1
        assert conn.last_activated is not None


class TestMeeting:
    """Test enhanced Meeting model."""

    def test_meeting_creation_minimal(self):
        """Test creating a meeting with minimal fields."""
        meeting = Meeting(project_id="proj_001", title="Team Sync")

        assert meeting.id is not None
        assert meeting.meeting_type == MeetingType.WORKING_SESSION
        assert meeting.meeting_category == MeetingCategory.INTERNAL
        assert meeting.is_recurring is False

    def test_meeting_consulting_fields(self):
        """Test meeting with consulting fields."""
        meeting = Meeting(
            project_id="proj_001",
            title="Client SteerCo",
            meeting_type=MeetingType.CLIENT_STEERING,
            meeting_category=MeetingCategory.EXTERNAL,
            participants=[
                {"name": "CEO", "role": "client", "org": "Acme"},
                {"name": "Partner", "role": "consultant", "org": "McKinsey"},
            ],
            key_decisions=["Approve Phase 2", "Increase budget"],
            action_items=[{"task": "Update roadmap", "owner": "PM", "due": "2024-02-01"}],
        )

        assert meeting.meeting_type == MeetingType.CLIENT_STEERING
        assert len(meeting.participants) == 2
        assert len(meeting.key_decisions) == 2
        assert meeting.action_items[0]["task"] == "Update roadmap"

    def test_meeting_duration(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 11, 30)

        meeting = Meeting(project_id="proj_001", title="Workshop", start_time=start, end_time=end)

        assert meeting.duration_minutes == 90

    def test_meeting_serialization(self):
        """Test meeting serialization."""
        meeting = Meeting(
            project_id="proj_001",
            title="Planning Session",
            participants=[{"name": "Alice"}, {"name": "Bob"}],
            agenda={"items": ["Review", "Plan", "Decide"]},
        )

        data = meeting.to_dict()
        assert "participants_json" in data
        assert "agenda_json" in data

        meeting2 = Meeting.from_dict(data)
        assert len(meeting2.participants) == 2
        assert len(meeting2.agenda["items"]) == 3


class TestMeetingSeries:
    """Test MeetingSeries model."""

    def test_meeting_series_creation(self):
        """Test creating a meeting series."""
        series = MeetingSeries(
            project_id="proj_001",
            series_name="Weekly SteerCo",
            frequency=MeetingFrequency.WEEKLY,
            day_of_week=1,  # Tuesday
            typical_duration_minutes=60,
        )

        assert series.frequency == MeetingFrequency.WEEKLY
        assert series.day_of_week == 1

    def test_day_of_week_validation(self):
        """Test day of week validation."""
        with pytest.raises(ValueError):
            MeetingSeries(
                project_id="proj_001", series_name="Invalid", day_of_week=7  # Invalid: must be 0-6
            )


class TestVector:
    """Test Vector model."""

    def test_vector_creation(self):
        """Test creating a vector."""
        semantic = np.random.rand(384)
        dimensions = np.random.rand(16)

        vector = Vector(semantic=semantic, dimensions=dimensions)

        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)

    def test_vector_dimension_validation(self):
        """Test vector dimension validation."""
        with pytest.raises(ValueError):
            Vector(semantic=np.random.rand(300), dimensions=np.random.rand(16))  # Wrong size

        with pytest.raises(ValueError):
            Vector(semantic=np.random.rand(384), dimensions=np.random.rand(20))  # Wrong size

    def test_full_vector(self):
        """Test full vector composition."""
        semantic = np.ones(384) * 0.5
        dimensions = np.ones(16) * 0.8

        vector = Vector(semantic=semantic, dimensions=dimensions)
        full = vector.full_vector

        assert full.shape == (400,)
        assert np.all(full[:384] == 0.5)
        assert np.all(full[384:] == 0.8)

    def test_vector_normalization(self):
        """Test vector normalization."""
        semantic = np.ones(384) * 2.0  # Not normalized
        dimensions = np.array([0.1] * 16)  # Already in [0,1]

        vector = Vector(semantic=semantic, dimensions=dimensions)
        normalized = vector.normalize()

        # Check semantic is normalized
        semantic_norm = np.linalg.norm(normalized.semantic)
        assert abs(semantic_norm - 1.0) < 1e-6

        # Check dimensions unchanged
        assert np.array_equal(normalized.dimensions, dimensions)

    def test_vector_serialization(self):
        """Test vector to/from list conversion."""
        semantic = np.random.rand(384)
        dimensions = np.random.rand(16)

        vector = Vector(semantic=semantic, dimensions=dimensions)

        # To list
        vector_list = vector.to_list()
        assert len(vector_list) == 400
        assert isinstance(vector_list, list)

        # From list
        vector2 = Vector.from_list(vector_list)
        assert np.array_equal(vector2.semantic, semantic)
        assert np.array_equal(vector2.dimensions, dimensions)


class TestSearchResult:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        memory = Memory(meeting_id="meet_001", project_id="proj_001", content="Test result")

        result = SearchResult(
            memory=memory,
            score=0.95,
            distance=0.05,
            activation_path=["mem_001", "mem_002", "mem_003"],
            explanation="Found via activation spreading",
        )

        assert result.score == 0.95
        assert result.distance == 0.05
        assert len(result.activation_path) == 3

    def test_search_result_serialization(self):
        """Test search result to_dict."""
        memory = Memory(meeting_id="meet_001", project_id="proj_001", content="Test")

        result = SearchResult(memory=memory, score=0.9, distance=0.1)

        data = result.to_dict()
        assert "memory" in data
        assert data["score"] == 0.9
        assert data["distance"] == 0.1


class TestIntegration:
    """Test model interactions and relationships."""

    def test_project_meeting_memory_relationship(self):
        """Test the relationship between projects, meetings, and memories."""
        # Create project
        project = Project(
            id="proj_integration", name="Integration Test Project", client_name="Test Client"
        )

        # Create meeting for project
        meeting = Meeting(id="meet_integration", project_id=project.id, title="Project Kickoff")

        # Create memory for meeting
        memory = Memory(
            meeting_id=meeting.id,
            project_id=project.id,
            content="Key decision from kickoff",
            content_type=ContentType.DECISION,
        )

        # Verify relationships
        assert memory.project_id == project.id
        assert memory.meeting_id == meeting.id
        assert meeting.project_id == project.id

    def test_memory_deliverable_relationship(self):
        """Test memory linked to deliverable."""
        deliverable = Deliverable(
            id="deliv_test", project_id="proj_001", name="Test Deliverable", owner="Owner"
        )

        memory = Memory(
            meeting_id="meet_001",
            project_id="proj_001",
            content="Complete the deliverable",
            content_type=ContentType.ACTION,
            deliverable_id=deliverable.id,
        )

        assert memory.deliverable_id == deliverable.id
