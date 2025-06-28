import pytest
import asyncio
from datetime import datetime, timedelta

from src.extraction.dimensions.dimension_analyzer import DimensionAnalyzer, DimensionExtractionContext, CognitiveDimensions
from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor
from src.extraction.dimensions.emotional_extractor import EmotionalDimensionExtractor
from src.extraction.dimensions.social_extractor import SocialDimensionExtractor
from src.extraction.dimensions.causal_extractor import CausalDimensionExtractor
from src.extraction.dimensions.evolutionary_extractor import EvolutionaryDimensionExtractor

@pytest.fixture
def dimension_analyzer():
    return DimensionAnalyzer()

@pytest.fixture
def default_context():
    return DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="Test Speaker",
        speaker_role="consultant",
        content_type="context",
        project_id="proj-123",
        meeting_type="internal_team",
        current_memory_index=0,
        total_memories=1
    )

@pytest.mark.asyncio
async def test_dimension_analyzer_basic(dimension_analyzer, default_context):
    content = "This is a test sentence about a project deadline."
    dimensions = await dimension_analyzer.analyze(content, default_context)

    assert isinstance(dimensions, CognitiveDimensions)
    assert 0.0 <= dimensions.temporal.urgency <= 1.0
    assert 0.0 <= dimensions.emotional.polarity <= 1.0
    assert 0.0 <= dimensions.social.authority <= 1.0
    assert 0.0 <= dimensions.causal.impact <= 1.0
    assert 0.0 <= dimensions.evolutionary.change_rate <= 1.0

@pytest.mark.asyncio
async def test_temporal_extractor():
    extractor = TemporalDimensionExtractor()
    context = DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="Test", speaker_role="consultant", content_type="action", project_id="p1",
        meeting_type="internal_team", current_memory_index=0, total_memories=1
    )

    # Test urgency
    content = "We need to finalize this by tomorrow, it's critical."
    dims = await extractor.extract(content, context)
    assert dims.urgency > 0.7

    # Test deadline_proximity
    future_date = datetime.now() + timedelta(days=2)
    context.timestamp_ms = int(future_date.timestamp() * 1000)
    content = "The deadline is in two days."
    dims = await extractor.extract(content, context)
    assert dims.deadline_proximity > 0.5

@pytest.mark.asyncio
async def test_emotional_extractor():
    extractor = EmotionalDimensionExtractor()
    context = DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="Test", speaker_role="consultant", content_type="insight", project_id="p1",
        meeting_type="internal_team", current_memory_index=0, total_memories=1
    )

    # Test polarity
    content = "This is an excellent idea, I'm very happy with it."
    dims = await extractor.extract(content, context)
    assert dims.polarity > 0.7

    content = "I am extremely disappointed with the results."
    dims = await extractor.extract(content, context)
    assert dims.polarity < 0.3

@pytest.mark.asyncio
async def test_social_extractor():
    extractor = SocialDimensionExtractor()
    context = DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="CEO", speaker_role="client_sponsor", content_type="decision", project_id="p1",
        meeting_type="client_steering", current_memory_index=0, total_memories=1
    )

    # Test authority
    content = "As the CEO, I approve this strategy."
    dims = await extractor.extract(content, context)
    assert dims.authority > 0.8

    # Test influence (simplified)
    content = "I suggest we consider this approach."
    context.speaker_role = "consultant"
    dims = await extractor.extract(content, context)
    assert dims.influence > 0.5

@pytest.mark.asyncio
async def test_causal_extractor():
    extractor = CausalDimensionExtractor()
    context = DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="Test", speaker_role="consultant", content_type="risk", project_id="p1",
        meeting_type="internal_team", current_memory_index=0, total_memories=1
    )

    # Test impact
    content = "This decision will significantly impact our Q3 revenue."
    dims = await extractor.extract(content, context)
    assert dims.impact > 0.7

    # Test risk_factors
    content = "There are several risks associated with this approach, including market volatility."
    dims = await extractor.extract(content, context)
    assert dims.risk_factors > 0.7

@pytest.mark.asyncio
async def test_evolutionary_extractor():
    extractor = EvolutionaryDimensionExtractor()
    context = DimensionExtractionContext(
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        speaker="Test", speaker_role="consultant", content_type="insight", project_id="p1",
        meeting_type="internal_team", current_memory_index=0, total_memories=1
    )

    # Test change_rate
    content = "The market has been rapidly evolving over the last few months."
    dims = await extractor.extract(content, context)
    assert dims.change_rate > 0.7

    # Test innovation_level
    content = "This is a groundbreaking new solution that will revolutionize the industry."
    dims = await extractor.extract(content, context)
    assert dims.innovation_level > 0.8
