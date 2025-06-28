
import pytest
from datetime import datetime
import json

from src.extraction.memory_extractor import MemoryExtractor, ExtractedMemory
from src.models.entities import Memory, MemoryType, ContentType, Priority


@pytest.fixture
def extractor():
    return MemoryExtractor()


class TestMemoryExtractor:
    def test_extract_speaker(self, extractor):
        line1 = "John Smith: This is a statement."
        line2 = "[00:01:05] Sarah Johnson: And this is another one."
        line3 = "Just a regular sentence."

        assert extractor._extract_speaker(line1) == "John Smith"
        assert extractor._extract_speaker(line2) == "Sarah Johnson"
        assert extractor._extract_speaker(line3) is None

    def test_remove_speaker_prefix(self, extractor):
        line1 = "John Smith: This is a statement."
        line2 = "[00:01:05] Sarah Johnson: And this is another one."
        line3 = "Just a regular sentence."

        assert extractor._remove_speaker_prefix(line1) == "This is a statement."
        assert extractor._remove_speaker_prefix(line2) == "And this is another one."
        assert extractor._remove_speaker_prefix(line3) == "Just a regular sentence."

    def test_extract_timestamp(self, extractor):
        line1 = "[00:00:15] Speaker: Content."
        line2 = "(01:23:45) Speaker: Content."
        line3 = "02:00:00 Speaker: Content."
        line4 = "No timestamp here."

        assert extractor._extract_timestamp(line1) == 15000
        assert extractor._extract_timestamp(line2) == 4985000
        assert extractor._extract_timestamp(line3) == 7200000
        assert extractor._extract_timestamp(line4) is None

    def test_remove_timestamp(self, extractor):
        line1 = "[00:00:15] Speaker: Content."
        line2 = "(01:23:45) Speaker: Content."
        line3 = "02:00:00 Speaker: Content."
        line4 = "No timestamp here."

        assert extractor._remove_timestamp(line1) == "Speaker: Content."
        assert extractor._remove_timestamp(line2) == "Speaker: Content."
        assert extractor._remove_timestamp(line3) == "Speaker: Content."
        assert extractor._remove_timestamp(line4) == "No timestamp here."

    def test_segment_transcript(self, extractor):
        transcript = """
John: Hello.
This is a paragraph.

Sarah: Hi there.
Another line.
[00:00:10] Mike: And a timestamped line.
"""
        segments = extractor._segment_transcript(transcript)
        expected_segments = [
            "John: Hello.",
            "This is a paragraph.",
            "Sarah: Hi there.",
            "Another line.",
            "[00:00:10] Mike: And a timestamped line.",
        ]
        assert segments == expected_segments

    def test_classify_content_type(self, extractor):
        assert extractor._classify_content_type("We decided to proceed.")[0] == ContentType.DECISION
        assert extractor._classify_content_type("John, please send the report by Friday.")[0] == ContentType.ACTION
        assert extractor._classify_content_type("I commit to this task.")[0] == ContentType.COMMITMENT
        assert extractor._classify_content_type("What is the plan?")[0] == ContentType.QUESTION
        assert extractor._classify_content_type("I realized that this is key.")[0] == ContentType.INSIGHT
        assert extractor._classify_content_type("There is a risk of delay.")[0] == ContentType.RISK
        assert extractor._classify_content_type("We have an issue with the server.")[0] == ContentType.ISSUE
        assert extractor._classify_content_type("I assume this will work.")[0] == ContentType.ASSUMPTION
        assert extractor._classify_content_type("My hypothesis is that it will improve.")[0] == ContentType.HYPOTHESIS
        assert extractor._classify_content_type("Data shows a clear trend.")[0] == ContentType.FINDING
        assert extractor._classify_content_type("I recommend we move forward.")[0] == ContentType.RECOMMENDATION
        assert extractor._classify_content_type("This depends on the client's approval.")[0] == ContentType.DEPENDENCY
        assert extractor._classify_content_type("Just a general statement.")[0] == ContentType.CONTEXT

    def test_extract_memories_speaker_identification(self, extractor):
        transcript = """
Alice: This is from Alice.
Bob: This is from Bob.
"""
        memories = extractor.extract_memories(transcript, "m1", "p1")
        assert len(memories) == 2
        assert memories[0].speaker == "Alice"
        assert memories[1].speaker == "Bob"

    def test_extract_memories_timestamp_extraction(self, extractor):
        transcript = """
[00:00:05] Alice: First statement.
[00:00:10] Bob: Second statement.
"""
        memories = extractor.extract_memories(transcript, "m1", "p1")
        assert len(memories) == 2
        assert memories[0].timestamp_ms == 5000
        assert memories[1].timestamp_ms == 10000

    def test_extract_memories_content_type_classification(self, extractor):
        transcript = """
Alice: We decided to go with option A.
Bob: I will send the summary by end of day.
"""
        memories = extractor.extract_memories(transcript, "m1", "p1")
        assert len(memories) == 2
        assert memories[0].content_type == ContentType.DECISION
        assert memories[1].content_type == ContentType.ACTION

    def test_extract_memories_full_pipeline(self, extractor):
        transcript = """
John Smith: Good morning everyone. Let's start with the project status update.

Sarah Johnson: Thanks John. I wanted to highlight that we've completed the API integration ahead of schedule.

Mike Chen: [00:05:30] I have a concern about the database performance. We might need to optimize our queries.

Sarah Johnson: That's a valid point. I suggest we schedule a technical review session this week.

John Smith: Agreed. Let's make that an action item. Mike, can you lead the performance review by Friday?

Mike Chen: I'll take care of it. I'll also document our findings and recommendations.

John Smith: Perfect. One more thing - we need to decide on the deployment strategy for next month.

Sarah Johnson: I recommend a phased rollout to minimize risk.

John Smith: That makes sense. Let's go with the phased approach. This is our official decision.
"""
        memories = extractor.extract_memories(transcript, "meeting-001", "project-001")
        assert len(memories) > 0
        # Further assertions can be added here to check specific memory content, types, etc.
        assert any(m.content_type == ContentType.DECISION for m in memories)
        assert any(m.content_type == ContentType.ACTION for m in memories)
        assert any(m.speaker == "John Smith" for m in memories)
        assert any(m.timestamp_ms > 0 for m in memories)

    def test_extract_speakers(self, extractor):
        transcript = """
Alice: Hello.
Bob: Hi.
Alice: How are you?
Charlie: Good.
"""
        speakers = extractor.extract_speakers(transcript)
        assert sorted(speakers) == ["Alice", "Bob", "Charlie"]

    def test_get_extraction_stats(self, extractor):
        transcript = """
Alice: We decided.
Bob: Action item.
Alice: Just talking.
"""
        extractor.extract_memories(transcript, "m1", "p1")
        stats = extractor.get_extraction_stats()
        assert stats["total"] == 3
        assert stats[ContentType.DECISION.value] == 1
        assert stats[ContentType.ACTION.value] == 1
        assert stats[ContentType.CONTEXT.value] == 1
