"""
Memory extraction from meeting transcripts.

Reference: IMPLEMENTATION_GUIDE.md - Day 5: Extraction Pipeline
Extracts 6 types of memories from transcript text.
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from src.models.entities import Memory, MemoryType, ContentType

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Extracts structured memories from meeting transcripts.
    
    TODO Day 5:
    - [ ] Implement pattern matching for 6 memory types
    - [ ] Add speaker identification
    - [ ] Extract timestamps from transcript
    - [ ] Classify content types
    - [ ] Target: 10-15 memories/second
    """
    
    # Memory type patterns
    PATTERNS = {
        MemoryType.DECISION: [
            r"(?:we|I|let's)\s+(?:decided?|agreed?|will|should)\s+(?:to\s+)?(.+)",
            r"(?:the\s+)?decision\s+is\s+(?:to\s+)?(.+)",
            r"(?:we're|we\s+are)\s+going\s+(?:to|with)\s+(.+)",
        ],
        MemoryType.ACTION: [
            r"(?:I'll|I\s+will|you\s+will|[A-Z]\w+\s+will)\s+(.+)",
            r"(?:need\s+to|have\s+to|must|should)\s+(.+)",
            r"action\s+item:?\s*(.+)",
        ],
        MemoryType.IDEA: [
            r"(?:what\s+if|how\s+about|maybe\s+we\s+could)\s+(.+)",
            r"(?:idea|suggestion|proposal):?\s*(.+)",
            r"(?:we\s+could|should\s+consider)\s+(.+)",
        ],
        MemoryType.ISSUE: [
            r"(?:problem|issue|concern|challenge)\s+(?:is|with)\s+(.+)",
            r"(?:blocked|stuck|struggling)\s+(?:on|with)\s+(.+)",
            r"(?:risk|threat)\s+(?:of|that)\s+(.+)",
        ],
        MemoryType.QUESTION: [
            r"(?:question|wondering|curious)\s+(?:is|about)\s+(.+)\?",
            r"(?:what|how|why|when|where|who)\s+(.+)\?",
            r"(?:do|does|did|can|could|should)\s+(.+)\?",
        ],
        MemoryType.CONTEXT: [
            r"(.+)",  # Catch-all for general context
        ]
    }
    
    # Content type keywords
    CONTENT_KEYWORDS = {
        ContentType.TECHNICAL: ["code", "api", "database", "algorithm", "bug", "feature"],
        ContentType.STRATEGIC: ["strategy", "goal", "objective", "vision", "roadmap"],
        ContentType.OPERATIONAL: ["process", "workflow", "deployment", "release", "timeline"],
    }
    
    def extract_memories(self, transcript: str, meeting_id: str) -> List[Memory]:
        """
        Extract all memories from a transcript.
        
        Args:
            transcript: Full meeting transcript text
            meeting_id: ID of the meeting
            
        Returns:
            List of extracted Memory objects
            
        TODO Day 5:
        - [ ] Parse transcript into segments
        - [ ] Extract speaker and timestamp
        - [ ] Apply pattern matching
        - [ ] Classify memory and content types
        - [ ] Create Memory objects
        """
        memories = []
        
        # TODO Day 5: Parse transcript segments
        segments = self._parse_transcript(transcript)
        
        for segment in segments:
            # TODO Day 5: Extract memory from segment
            memory = self._extract_from_segment(segment, meeting_id)
            if memory:
                memories.append(memory)
        
        logger.info(f"Extracted {len(memories)} memories from transcript")
        return memories
    
    def _parse_transcript(self, transcript: str) -> List[Dict]:
        """
        Parse transcript into segments with metadata.
        
        TODO Day 5:
        - [ ] Split by speaker turns
        - [ ] Extract timestamps
        - [ ] Handle different transcript formats
        
        Returns list of dicts with:
        - text: segment text
        - speaker: speaker name
        - timestamp_ms: timestamp in milliseconds
        """
        segments = []
        
        # TODO Day 5: Implement transcript parsing
        # Simple line-by-line for now
        lines = transcript.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                segments.append({
                    'text': line.strip(),
                    'speaker': 'Unknown',  # TODO: Extract speaker
                    'timestamp_ms': i * 1000  # TODO: Extract real timestamp
                })
        
        return segments
    
    def _extract_from_segment(self, segment: Dict, meeting_id: str) -> Optional[Memory]:
        """
        Extract a memory from a transcript segment.
        
        TODO Day 5:
        - [ ] Apply pattern matching
        - [ ] Determine memory type
        - [ ] Classify content type
        - [ ] Calculate importance score
        """
        text = segment['text']
        
        # TODO Day 5: Try each memory type pattern
        memory_type = MemoryType.CONTEXT
        content = text
        
        for mem_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and mem_type != MemoryType.CONTEXT:
                    memory_type = mem_type
                    content = match.group(1) if match.groups() else text
                    break
            if memory_type != MemoryType.CONTEXT:
                break
        
        # TODO Day 5: Classify content type
        content_type = self._classify_content_type(content)
        
        # TODO Day 5: Calculate importance score
        importance = self._calculate_importance(memory_type, content)
        
        return Memory(
            meeting_id=meeting_id,
            content=content,
            speaker=segment['speaker'],
            timestamp_ms=segment['timestamp_ms'],
            memory_type=memory_type,
            content_type=content_type,
            importance_score=importance
        )
    
    def _classify_content_type(self, text: str) -> ContentType:
        """
        Classify content type based on keywords.
        
        TODO Day 5:
        - [ ] Check for keyword matches
        - [ ] Return most relevant type
        """
        text_lower = text.lower()
        
        for content_type, keywords in self.CONTENT_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return content_type
        
        return ContentType.GENERAL
    
    def _calculate_importance(self, memory_type: MemoryType, content: str) -> float:
        """
        Calculate importance score (0-1).
        
        TODO Day 5:
        - [ ] Weight by memory type
        - [ ] Consider content length
        - [ ] Apply business rules
        """
        # Base scores by type
        base_scores = {
            MemoryType.DECISION: 0.9,
            MemoryType.ACTION: 0.8,
            MemoryType.ISSUE: 0.7,
            MemoryType.IDEA: 0.6,
            MemoryType.QUESTION: 0.5,
            MemoryType.CONTEXT: 0.3
        }
        
        score = base_scores.get(memory_type, 0.5)
        
        # TODO Day 5: Adjust based on content
        # Length bonus (longer = more important)
        if len(content) > 100:
            score = min(1.0, score + 0.1)
        
        return score