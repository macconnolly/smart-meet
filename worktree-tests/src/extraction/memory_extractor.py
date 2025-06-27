"""
Memory extractor for processing meeting transcripts.

This module extracts individual memories from meeting transcripts,
identifying speakers, classifying content types, and preparing
memories for the cognitive pipeline.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import uuid
from collections import defaultdict

from ..models.entities import Memory, MemoryType, ContentType, Priority

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMemory:
    """Raw extracted memory before full processing."""
    content: str
    speaker: Optional[str]
    timestamp_ms: int
    detected_type: ContentType
    confidence: float
    metadata: Dict[str, Any]


class MemoryExtractor:
    """
    Extracts memories from meeting transcripts.
    
    Features:
    - Speaker identification
    - Content type classification
    - Timestamp extraction
    - Memory segmentation
    - Metadata extraction
    """
    
    # Content type patterns
    CONTENT_PATTERNS = {
        ContentType.DECISION: [
            r"(?:we|I|let's)\s+(?:will|should|must|need to|have to|decided to|agree to)",
            r"(?:decision|decided|agreed|concluded)\s+(?:to|that|on)",
            r"(?:final|official)\s+(?:decision|answer|position)",
            r"(?:approved|rejected|selected|chosen)",
        ],
        ContentType.ACTION: [
            r"(?:I|you|we|they)\s+(?:will|should|must|need to)\s+(?:do|complete|finish|create|build|send|review)",
            r"action\s+item",
            r"(?:by|before|until)\s+(?:tomorrow|next|this|end of)",
            r"(?:assigned to|owner:?|responsible:?)",
        ],
        ContentType.COMMITMENT: [
            r"(?:I|we)\s+(?:commit|promise|guarantee|ensure|will definitely)",
            r"(?:commitment|obligation|promise)",
            r"(?:accountable|responsible)\s+for",
        ],
        ContentType.QUESTION: [
            r"^(?:what|when|where|who|why|how|can|could|should|would|is|are|do|does)",
            r"\?$",
            r"(?:question|wondering|curious|unclear)",
        ],
        ContentType.INSIGHT: [
            r"(?:realized|discovered|found|learned|understood)\s+that",
            r"(?:insight|observation|finding|discovery)",
            r"(?:interesting|important|key|critical)\s+(?:point|aspect|factor)",
            r"(?:means|implies|suggests|indicates)\s+that",
        ],
        ContentType.RISK: [
            r"(?:risk|threat|concern|danger|hazard)",
            r"(?:might|could|may)\s+(?:fail|break|delay|impact)",
            r"(?:worried|concerned)\s+(?:about|that)",
            r"(?:vulnerability|exposure|liability)",
        ],
        ContentType.ISSUE: [
            r"(?:problem|issue|challenge|obstacle|blocker)",
            r"(?:broken|failed|not working|delayed)",
            r"(?:stuck|blocked|waiting on)",
            r"(?:error|bug|defect|failure)",
        ],
        ContentType.ASSUMPTION: [
            r"(?:assume|assuming|presume|suppose)",
            r"(?:probably|likely|should be|must be)",
            r"(?:based on|given that|considering)",
        ],
        ContentType.HYPOTHESIS: [
            r"(?:hypothesis|theory|believe that|think that)",
            r"(?:if.*then|when.*should)",
            r"(?:predict|expect|anticipate)\s+that",
        ],
        ContentType.FINDING: [
            r"(?:data shows|analysis reveals|report indicates)",
            r"(?:found that|shows that|demonstrates that)",
            r"(?:evidence|proof|validation)\s+(?:of|that)",
        ],
        ContentType.RECOMMENDATION: [
            r"(?:recommend|suggest|propose|advise)",
            r"(?:should|ought to|better to|would be good to)",
            r"(?:recommendation|suggestion|proposal)",
        ],
        ContentType.DEPENDENCY: [
            r"(?:depends on|dependent on|requires|needs)",
            r"(?:blocked by|waiting for|contingent on)",
            r"(?:prerequisite|precondition|requirement)",
        ],
    }
    
    # Speaker pattern
    SPEAKER_PATTERN = re.compile(r"^([A-Z][A-Za-z\s\-']+):\s*(.+)$")
    
    # Timestamp patterns
    TIMESTAMP_PATTERNS = [
        re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})\]"),  # [HH:MM:SS]
        re.compile(r"\((\d{1,2}):(\d{2}):(\d{2})\)"),  # (HH:MM:SS)
        re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})\s+"),  # HH:MM:SS at start
    ]
    
    def __init__(self):
        """Initialize the memory extractor."""
        # Compile content patterns
        self.compiled_patterns = {}
        for content_type, patterns in self.CONTENT_PATTERNS.items():
            self.compiled_patterns[content_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Statistics
        self.extraction_stats = defaultdict(int)
    
    def extract_memories(
        self,
        transcript: str,
        meeting_id: str,
        project_id: str,
        meeting_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """
        Extract memories from a meeting transcript.
        
        Args:
            transcript: Raw transcript text
            meeting_id: ID of the meeting
            project_id: ID of the project
            meeting_metadata: Optional meeting metadata
            
        Returns:
            List of extracted Memory objects
        """
        # Reset stats
        self.extraction_stats.clear()
        
        # Split into segments
        segments = self._segment_transcript(transcript)
        
        # Extract raw memories
        raw_memories = []
        current_speaker = None
        current_timestamp_ms = 0
        
        for segment in segments:
            # Try to extract speaker
            speaker = self._extract_speaker(segment)
            if speaker:
                current_speaker = speaker
                segment = self._remove_speaker_prefix(segment)
            
            # Try to extract timestamp
            timestamp_ms = self._extract_timestamp(segment)
            if timestamp_ms is not None:
                current_timestamp_ms = timestamp_ms
                segment = self._remove_timestamp(segment)
            
            # Skip empty segments
            if not segment.strip():
                continue
            
            # Extract memory
            raw_memory = self._extract_single_memory(
                content=segment.strip(),
                speaker=current_speaker,
                timestamp_ms=current_timestamp_ms
            )
            raw_memories.append(raw_memory)
        
        # Convert to Memory objects
        memories = []
        for i, raw in enumerate(raw_memories):
            memory = self._create_memory(
                raw_memory=raw,
                meeting_id=meeting_id,
                project_id=project_id,
                sequence_number=i,
                meeting_metadata=meeting_metadata
            )
            memories.append(memory)
            
            # Update stats
            self.extraction_stats['total'] += 1
            self.extraction_stats[raw.detected_type.value] += 1
        
        # Log extraction statistics
        logger.info(
            f"Extracted {len(memories)} memories from meeting {meeting_id}: "
            f"{dict(self.extraction_stats)}"
        )
        
        return memories
    
    def _segment_transcript(self, transcript: str) -> List[str]:
        """
        Segment transcript into processable chunks.
        
        Args:
            transcript: Raw transcript
            
        Returns:
            List of segments
        """
        # Split by common delimiters
        # First, split by double newlines (paragraph breaks)
        paragraphs = transcript.split('\n\n')
        
        segments = []
        for paragraph in paragraphs:
            # Then split by sentence-ending punctuation followed by newline
            lines = paragraph.split('\n')
            
            current_segment = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts with a speaker name
                if self.SPEAKER_PATTERN.match(line):
                    # Save current segment if any
                    if current_segment:
                        segments.append(' '.join(current_segment))
                        current_segment = []
                    segments.append(line)
                else:
                    # Check if line ends with sentence punctuation
                    if line.endswith(('.', '!', '?')):
                        current_segment.append(line)
                        segments.append(' '.join(current_segment))
                        current_segment = []
                    else:
                        current_segment.append(line)
            
            # Don't forget the last segment
            if current_segment:
                segments.append(' '.join(current_segment))
        
        return segments
    
    def _extract_speaker(self, text: str) -> Optional[str]:
        """Extract speaker name from text."""
        match = self.SPEAKER_PATTERN.match(text)
        if match:
            return match.group(1).strip()
        return None
    
    def _remove_speaker_prefix(self, text: str) -> str:
        """Remove speaker prefix from text."""
        match = self.SPEAKER_PATTERN.match(text)
        if match:
            return match.group(2).strip()
        return text
    
    def _extract_timestamp(self, text: str) -> Optional[int]:
        """Extract timestamp from text and convert to milliseconds."""
        for pattern in self.TIMESTAMP_PATTERNS:
            match = pattern.search(text)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3))
                
                # Convert to milliseconds
                total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
                return total_ms
        
        return None
    
    def _remove_timestamp(self, text: str) -> str:
        """Remove timestamp from text."""
        for pattern in self.TIMESTAMP_PATTERNS:
            text = pattern.sub('', text)
        return text.strip()
    
    def _extract_single_memory(
        self,
        content: str,
        speaker: Optional[str],
        timestamp_ms: int
    ) -> ExtractedMemory:
        """
        Extract a single memory from content.
        
        Args:
            content: Memory content
            speaker: Speaker name
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            ExtractedMemory object
        """
        # Classify content type
        detected_type, confidence = self._classify_content_type(content)
        
        # Extract any metadata
        metadata = self._extract_metadata(content)
        
        return ExtractedMemory(
            content=content,
            speaker=speaker,
            timestamp_ms=timestamp_ms,
            detected_type=detected_type,
            confidence=confidence,
            metadata=metadata
        )
    
    def _classify_content_type(self, content: str) -> Tuple[ContentType, float]:
        """
        Classify the content type of a memory.
        
        Args:
            content: Memory content
            
        Returns:
            Tuple of (ContentType, confidence)
        """
        scores = {}
        
        # Check each content type
        for content_type, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(content):
                    matches += 1
                    score += 1.0
            
            # Normalize score
            if patterns:
                score = score / len(patterns)
            
            scores[content_type] = score
        
        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            # If confidence is too low, default to CONTEXT
            if confidence < 0.2:
                return ContentType.CONTEXT, 0.5
            
            return best_type, min(confidence * 2, 1.0)  # Scale confidence
        
        # Default to CONTEXT
        return ContentType.CONTEXT, 0.5
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from content.
        
        Args:
            content: Memory content
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract mentioned people (simple name detection)
        # Look for capitalized names
        name_pattern = re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b')
        names = name_pattern.findall(content)
        if names:
            metadata['mentioned_people'] = list(set(names))
        
        # Extract dates
        date_patterns = [
            re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', re.IGNORECASE),
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(pattern.findall(content))
        
        if dates:
            metadata['mentioned_dates'] = dates
        
        # Extract numbers/metrics
        number_pattern = re.compile(r'\b(\d+(?:\.\d+)?%?)\b')
        numbers = number_pattern.findall(content)
        if numbers:
            metadata['metrics'] = numbers
        
        # Extract quoted text
        quote_pattern = re.compile(r'"([^"]+)"')
        quotes = quote_pattern.findall(content)
        if quotes:
            metadata['quotes'] = quotes
        
        return metadata
    
    def _create_memory(
        self,
        raw_memory: ExtractedMemory,
        meeting_id: str,
        project_id: str,
        sequence_number: int,
        meeting_metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Create a Memory object from extracted data.
        
        Args:
            raw_memory: Extracted memory data
            meeting_id: Meeting ID
            project_id: Project ID
            sequence_number: Position in meeting
            meeting_metadata: Optional meeting metadata
            
        Returns:
            Memory object
        """
        # Determine priority based on content type
        priority = None
        if raw_memory.detected_type in [ContentType.DECISION, ContentType.RISK]:
            priority = Priority.HIGH
        elif raw_memory.detected_type in [ContentType.ACTION, ContentType.COMMITMENT]:
            priority = Priority.MEDIUM
        
        # Create memory
        memory = Memory(
            id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            project_id=project_id,
            content=raw_memory.content,
            speaker=raw_memory.speaker,
            timestamp_ms=raw_memory.timestamp_ms,
            memory_type=MemoryType.EPISODIC,  # All extracted memories are episodic
            content_type=raw_memory.detected_type,
            priority=priority,
            level=2,  # L2 (episodic) by default
            created_at=datetime.now()
        )
        
        # Add speaker role if available in meeting metadata
        if meeting_metadata and 'participants' in meeting_metadata:
            for participant in meeting_metadata['participants']:
                if participant.get('name') == raw_memory.speaker:
                    memory.speaker_role = participant.get('role', 'participant')
                    break
        
        return memory
    
    def extract_speakers(self, transcript: str) -> List[str]:
        """
        Extract unique speakers from transcript.
        
        Args:
            transcript: Raw transcript
            
        Returns:
            List of unique speaker names
        """
        speakers = set()
        
        for line in transcript.split('\n'):
            speaker = self._extract_speaker(line)
            if speaker:
                speakers.add(speaker)
        
        return sorted(list(speakers))
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return dict(self.extraction_stats)


# Example usage
if __name__ == "__main__":
    extractor = MemoryExtractor()
    
    # Sample transcript
    transcript = """
John Smith: Good morning everyone. Let's start with the project status update.

Sarah Johnson: Thanks John. I wanted to highlight that we've completed the API integration ahead of schedule.

John Smith: That's excellent news! We should celebrate this achievement.

Mike Chen: [00:05:30] I have a concern about the database performance. We might need to optimize our queries.

Sarah Johnson: That's a valid point. I suggest we schedule a technical review session this week.

John Smith: Agreed. Let's make that an action item. Mike, can you lead the performance review by Friday?

Mike Chen: I'll take care of it. I'll also document our findings and recommendations.

John Smith: Perfect. One more thing - we need to decide on the deployment strategy for next month.

Sarah Johnson: I recommend a phased rollout to minimize risk.

John Smith: That makes sense. Let's go with the phased approach. This is our official decision.
"""
    
    # Extract memories
    memories = extractor.extract_memories(
        transcript=transcript,
        meeting_id="meeting-001",
        project_id="project-001"
    )
    
    # Print results
    print(f"Extracted {len(memories)} memories:\n")
    for i, memory in enumerate(memories):
        print(f"{i+1}. [{memory.content_type.value}] {memory.speaker}: {memory.content}")
        if memory.priority:
            print(f"   Priority: {memory.priority.value}")
        print()
    
    # Print statistics
    print("Extraction statistics:")
    for content_type, count in extractor.get_extraction_stats().items():
        print(f"  {content_type}: {count}")
