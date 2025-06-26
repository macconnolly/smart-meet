"""
Temporal dimension extractor for meeting memories.

Extracts 4D temporal features:
1. Urgency (0-1): How urgent/time-sensitive
2. Deadline proximity (0-1): How close to a deadline
3. Sequence position (0-1): Position in meeting timeline
4. Duration relevance (0-1): How long-term vs short-term
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TemporalExtractor:
    """
    Extracts temporal dimensions from text content.
    
    Uses keyword detection, pattern matching, and contextual analysis
    to determine temporal characteristics of memories.
    """
    
    # Urgency keywords with scores
    URGENCY_KEYWORDS = {
        # Critical urgency (0.9-1.0)
        'urgent': 1.0,
        'asap': 1.0,
        'immediately': 1.0,
        'critical': 0.95,
        'emergency': 0.95,
        'today': 0.9,
        'now': 0.9,
        
        # High urgency (0.7-0.9)
        'tomorrow': 0.8,
        'soon': 0.75,
        'quickly': 0.75,
        'priority': 0.7,
        'important': 0.7,
        
        # Medium urgency (0.5-0.7)
        'this week': 0.6,
        'next week': 0.55,
        'timely': 0.5,
        
        # Low urgency (0.2-0.5)
        'this month': 0.4,
        'next month': 0.3,
        'eventually': 0.2,
        'someday': 0.2
    }
    
    # Deadline patterns
    DEADLINE_PATTERNS = [
        r'by\s+(\w+\s+\d+)',  # by March 15
        r'due\s+(\w+\s+\d+)',  # due April 1
        r'deadline[:\s]+(\w+\s+\d+)',  # deadline: May 20
        r'before\s+(\w+\s+\d+)',  # before June 30
        r'no later than\s+(\w+\s+\d+)',  # no later than July 15
    ]
    
    def __init__(self):
        """Initialize the temporal extractor."""
        self.deadline_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEADLINE_PATTERNS]
    
    def extract(
        self,
        text: str,
        timestamp_ms: int = 0,
        meeting_duration_ms: int = 3600000  # 1 hour default
    ) -> np.ndarray:
        """
        Extract 4D temporal features from text.
        
        Args:
            text: Input text to analyze
            timestamp_ms: Timestamp within meeting (milliseconds)
            meeting_duration_ms: Total meeting duration (milliseconds)
            
        Returns:
            4D numpy array with temporal features
        """
        # TODO: Extract urgency score
        urgency = self._extract_urgency(text)
        
        # TODO: Extract deadline proximity
        deadline_proximity = self._extract_deadline_proximity(text)
        
        # TODO: Calculate sequence position
        sequence_position = self._calculate_sequence_position(timestamp_ms, meeting_duration_ms)
        
        # TODO: Determine duration relevance
        duration_relevance = self._extract_duration_relevance(text)
        
        # TODO: Compose and return 4D vector
        return np.array([
            urgency,
            deadline_proximity,
            sequence_position,
            duration_relevance
        ], dtype=np.float32)
    
    def _extract_urgency(self, text: str) -> float:
        """
        Extract urgency score from text.
        
        Args:
            text: Input text
            
        Returns:
            Urgency score between 0 and 1
        """
        text_lower = text.lower()
        
        # TODO: Check for urgency keywords
        # - Find all matching keywords
        # - Take maximum score if multiple matches
        # - Default to 0.3 if no matches
        
        urgency_score = 0.3  # Default neutral urgency
        
        # TODO: Implement keyword matching
        
        return np.clip(urgency_score, 0.0, 1.0)
    
    def _extract_deadline_proximity(self, text: str) -> float:
        """
        Extract deadline proximity from text.
        
        Args:
            text: Input text
            
        Returns:
            Deadline proximity score between 0 and 1
        """
        # TODO: Search for deadline patterns
        # - Use regex patterns to find dates
        # - Parse dates and calculate days until deadline
        # - Convert to proximity score (closer = higher)
        
        # TODO: Handle relative dates (next week, etc.)
        
        # Default: no specific deadline
        return 0.0
    
    def _calculate_sequence_position(self, timestamp_ms: int, duration_ms: int) -> float:
        """
        Calculate position in meeting sequence.
        
        Args:
            timestamp_ms: Current timestamp in meeting
            duration_ms: Total meeting duration
            
        Returns:
            Position score between 0 (start) and 1 (end)
        """
        # TODO: Handle edge cases
        # - Zero or negative duration
        # - Timestamp beyond duration
        
        if duration_ms <= 0:
            return 0.5
        
        # TODO: Calculate normalized position
        position = timestamp_ms / duration_ms
        return np.clip(position, 0.0, 1.0)
    
    def _extract_duration_relevance(self, text: str) -> float:
        """
        Extract duration relevance (long-term vs short-term).
        
        Args:
            text: Input text
            
        Returns:
            Duration relevance score (0=short-term, 1=long-term)
        """
        text_lower = text.lower()
        
        # TODO: Define patterns for different time horizons
        long_term_indicators = [
            'strategy', 'strategic', 'long-term', 'future', 'vision',
            'roadmap', 'years', 'annual', 'quarterly'
        ]
        
        short_term_indicators = [
            'today', 'tomorrow', 'immediate', 'quick', 'temporary',
            'workaround', 'hotfix', 'this week'
        ]
        
        # TODO: Count indicators and calculate score
        # - More long-term indicators = higher score
        # - More short-term indicators = lower score
        # - Balance if both present
        
        return 0.5  # Default: medium-term
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of temporal features.
        
        Returns:
            List of feature names
        """
        return [
            'urgency',
            'deadline_proximity',
            'sequence_position',
            'duration_relevance'
        ]


def test_temporal_extractor():
    """Test the temporal extractor."""
    print("Testing TemporalExtractor...")
    
    extractor = TemporalExtractor()
    
    # Test cases
    test_texts = [
        "This is urgent and needs to be done immediately!",
        "We should complete this by March 15",
        "Let's think about our long-term strategy",
        "Just a regular discussion point"
    ]
    
    for text in test_texts:
        features = extractor.extract(text)
        print(f"\nText: {text}")
        print(f"Features: {features}")
        print(f"Feature names: {extractor.get_feature_names()}")
    
    print("\n✅ TemporalExtractor tests completed!")


if __name__ == "__main__":
    test_temporal_extractor()