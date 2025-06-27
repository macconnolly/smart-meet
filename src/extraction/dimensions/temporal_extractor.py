"""
Temporal dimension extractor for cognitive features.

This module extracts 4 temporal dimensions from memory content:
- Urgency: How time-sensitive the information is
- Deadline proximity: Closeness to deadlines or due dates
- Sequence position: Position in meeting or discussion flow
- Duration relevance: How long the information remains relevant
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalFeatures:
    """Extracted temporal features."""
    urgency: float  # 0-1: How urgent/time-sensitive
    deadline_proximity: float  # 0-1: How close to deadline
    sequence_position: float  # 0-1: Position in sequence
    duration_relevance: float  # 0-1: Long-term vs short-term relevance
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.urgency,
            self.deadline_proximity,
            self.sequence_position,
            self.duration_relevance
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'TemporalFeatures':
        """Create from numpy array."""
        if array.shape != (4,):
            raise ValueError(f"Array must be 4D, got {array.shape}")
        return cls(
            urgency=float(array[0]),
            deadline_proximity=float(array[1]),
            sequence_position=float(array[2]),
            duration_relevance=float(array[3])
        )


class TemporalDimensionExtractor:
    """
    Extracts temporal dimensions from memory content.
    
    Uses keyword matching, pattern recognition, and contextual
    analysis to determine temporal characteristics.
    """
    
    # Urgency keywords and phrases
    URGENCY_KEYWORDS = {
        "critical": 1.0,
        "urgent": 0.9,
        "asap": 0.9,
        "immediately": 0.9,
        "right away": 0.9,
        "time-sensitive": 0.8,
        "priority": 0.8,
        "deadline": 0.8,
        "by eod": 0.8,
        "by end of day": 0.8,
        "today": 0.7,
        "tomorrow": 0.6,
        "this week": 0.5,
        "next week": 0.4,
        "soon": 0.4,
        "when possible": 0.3,
        "eventually": 0.2,
        "someday": 0.1,
        "no rush": 0.1,
    }
    
    # Deadline patterns
    DEADLINE_PATTERNS = [
        (r"by\s+(\w+\s+\d{1,2})", "date"),  # by January 15
        (r"by\s+(\d{1,2}/\d{1,2})", "date"),  # by 1/15
        (r"due\s+(\w+\s+\d{1,2})", "date"),  # due March 1
        (r"deadline[:\s]+(\w+\s+\d{1,2})", "date"),  # deadline: April 20
        (r"by\s+(\w+day)", "weekday"),  # by Friday
        (r"by\s+next\s+(\w+day)", "next_weekday"),  # by next Monday
        (r"within\s+(\d+)\s+days?", "days"),  # within 3 days
        (r"within\s+(\d+)\s+weeks?", "weeks"),  # within 2 weeks
        (r"by\s+end\s+of\s+(\w+)", "period"),  # by end of month
    ]
    
    # Duration indicators
    DURATION_KEYWORDS = {
        # Long-term relevance
        "strategic": 0.9,
        "long-term": 0.9,
        "permanent": 0.9,
        "ongoing": 0.8,
        "continuous": 0.8,
        "always": 0.8,
        "policy": 0.8,
        "principle": 0.8,
        "framework": 0.7,
        "process": 0.7,
        "standard": 0.7,
        # Short-term relevance
        "temporary": 0.3,
        "interim": 0.3,
        "quick fix": 0.2,
        "for now": 0.2,
        "one-time": 0.2,
        "ad hoc": 0.2,
        "this meeting": 0.1,
        "just today": 0.1,
    }
    
    def __init__(self):
        """Initialize the temporal extractor."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), pattern_type)
            for pattern, pattern_type in self.DEADLINE_PATTERNS
        ]
    
    def extract(
        self,
        content: str,
        timestamp_ms: Optional[int] = None,
        meeting_duration_ms: Optional[int] = None,
        speaker: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> TemporalFeatures:
        """
        Extract temporal dimensions from memory content.
        
        Args:
            content: Memory content text
            timestamp_ms: When in the meeting this was said
            meeting_duration_ms: Total meeting duration
            speaker: Who said this (for authority weighting)
            content_type: Type of content (decisions may be more urgent)
            
        Returns:
            TemporalFeatures with 4 dimensions
        """
        # Extract individual features
        urgency = self._extract_urgency(content, content_type)
        deadline_proximity = self._extract_deadline_proximity(content)
        sequence_position = self._extract_sequence_position(
            timestamp_ms, meeting_duration_ms
        )
        duration_relevance = self._extract_duration_relevance(content, content_type)
        
        return TemporalFeatures(
            urgency=urgency,
            deadline_proximity=deadline_proximity,
            sequence_position=sequence_position,
            duration_relevance=duration_relevance
        )
    
    def _extract_urgency(self, content: str, content_type: Optional[str]) -> float:
        """
        Extract urgency level from content.
        
        Returns:
            Urgency score 0-1
        """
        content_lower = content.lower()
        
        # Check for urgency keywords
        max_urgency = 0.0
        for keyword, score in self.URGENCY_KEYWORDS.items():
            if keyword in content_lower:
                max_urgency = max(max_urgency, score)
        
        # Boost for certain content types
        if content_type:
            if content_type in ["action", "commitment", "risk"]:
                max_urgency = min(1.0, max_urgency * 1.2)
            elif content_type in ["decision"]:
                max_urgency = min(1.0, max_urgency * 1.1)
        
        # Check for exclamation marks (mild indicator)
        if "!" in content:
            max_urgency = min(1.0, max_urgency + 0.1)
        
        # Check for all caps words (strong indicator)
        words = content.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) > 0:
            max_urgency = min(1.0, max_urgency + 0.1 * min(len(caps_words), 3))
        
        return max_urgency
    
    def _extract_deadline_proximity(self, content: str) -> float:
        """
        Extract deadline proximity from content.
        
        Returns:
            Deadline proximity score 0-1 (1 = very close deadline)
        """
        # Look for deadline patterns
        for pattern, pattern_type in self._compiled_patterns:
            match = pattern.search(content)
            if match:
                return self._calculate_deadline_score(match.group(1), pattern_type)
        
        # No explicit deadline found
        return 0.0
    
    def _calculate_deadline_score(self, deadline_text: str, pattern_type: str) -> float:
        """
        Calculate deadline proximity score based on parsed deadline.
        
        Args:
            deadline_text: Extracted deadline text
            pattern_type: Type of deadline pattern
            
        Returns:
            Score 0-1 based on how close the deadline is
        """
        try:
            current_date = datetime.now()
            
            if pattern_type == "days":
                days = int(deadline_text)
                # Score based on days: 1 day = 0.9, 7 days = 0.5, 30 days = 0.1
                return max(0.1, 1.0 - (days / 35))
            
            elif pattern_type == "weeks":
                weeks = int(deadline_text)
                days = weeks * 7
                return max(0.1, 1.0 - (days / 35))
            
            elif pattern_type == "weekday":
                # Simple approximation - assume within next 7 days
                return 0.7
            
            elif pattern_type == "next_weekday":
                # Next week
                return 0.5
            
            elif pattern_type == "period":
                period = deadline_text.lower()
                if period in ["day", "today"]:
                    return 0.9
                elif period in ["week"]:
                    return 0.6
                elif period in ["month"]:
                    return 0.3
                elif period in ["quarter"]:
                    return 0.2
                elif period in ["year"]:
                    return 0.1
            
            # Default for unrecognized patterns
            return 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating deadline score: {e}")
            return 0.5
    
    def _extract_sequence_position(
        self,
        timestamp_ms: Optional[int],
        meeting_duration_ms: Optional[int]
    ) -> float:
        """
        Extract sequence position in meeting.
        
        Args:
            timestamp_ms: When this was said
            meeting_duration_ms: Total meeting duration
            
        Returns:
            Position score 0-1 (0 = beginning, 1 = end)
        """
        if timestamp_ms is None or meeting_duration_ms is None:
            # Default to middle if no timing info
            return 0.5
        
        if meeting_duration_ms <= 0:
            return 0.5
        
        # Calculate relative position
        position = timestamp_ms / meeting_duration_ms
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, position))
    
    def _extract_duration_relevance(
        self,
        content: str,
        content_type: Optional[str]
    ) -> float:
        """
        Extract how long the information remains relevant.
        
        Returns:
            Duration relevance score 0-1 (1 = long-term relevance)
        """
        content_lower = content.lower()
        
        # Check duration keywords
        max_score = 0.5  # Default to medium-term
        
        for keyword, score in self.DURATION_KEYWORDS.items():
            if keyword in content_lower:
                max_score = max(max_score, score)
        
        # Adjust based on content type
        if content_type:
            if content_type in ["principle", "framework", "policy"]:
                max_score = max(max_score, 0.8)
            elif content_type in ["decision", "commitment"]:
                max_score = max(max_score, 0.7)
            elif content_type in ["action", "task"]:
                max_score = min(max_score, 0.5)  # Tasks are usually short-term
        
        # Check for future tense (might indicate longer relevance)
        future_indicators = ["will", "going to", "plan to", "intend to", "strategy"]
        if any(indicator in content_lower for indicator in future_indicators):
            max_score = max(max_score, 0.6)
        
        return max_score
    
    def batch_extract(
        self,
        contents: List[str],
        timestamps_ms: Optional[List[int]] = None,
        meeting_duration_ms: Optional[int] = None,
        speakers: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract temporal features for multiple memories.
        
        Args:
            contents: List of memory contents
            timestamps_ms: List of timestamps
            meeting_duration_ms: Meeting duration
            speakers: List of speakers
            content_types: List of content types
            
        Returns:
            Array of shape (n_memories, 4)
        """
        n_memories = len(contents)
        
        # Prepare lists with None if not provided
        if timestamps_ms is None:
            timestamps_ms = [None] * n_memories
        if speakers is None:
            speakers = [None] * n_memories
        if content_types is None:
            content_types = [None] * n_memories
        
        # Extract features
        features = []
        for i in range(n_memories):
            temporal_features = self.extract(
                content=contents[i],
                timestamp_ms=timestamps_ms[i],
                meeting_duration_ms=meeting_duration_ms,
                speaker=speakers[i],
                content_type=content_types[i]
            )
            features.append(temporal_features.to_array())
        
        return np.vstack(features)


# Example usage and testing
if __name__ == "__main__":
    extractor = TemporalDimensionExtractor()
    
    # Test cases
    test_cases = [
        "This is urgent! We need to complete this by tomorrow.",
        "Let's establish a long-term strategic framework for the project.",
        "No rush, but please look into this when you have time.",
        "Critical: Submit the report by Friday EOD!",
        "This is our ongoing policy moving forward.",
        "Quick temporary fix for today's demo.",
    ]
    
    for content in test_cases:
        features = extractor.extract(content)
        print(f"\nContent: {content}")
        print(f"Urgency: {features.urgency:.2f}")
        print(f"Deadline Proximity: {features.deadline_proximity:.2f}")
        print(f"Sequence Position: {features.sequence_position:.2f}")
        print(f"Duration Relevance: {features.duration_relevance:.2f}")
