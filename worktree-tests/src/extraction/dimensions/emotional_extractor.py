"""
Emotional dimension extractor for cognitive features.

This module extracts 3 emotional dimensions from memory content:
- Polarity: Positive vs negative sentiment (-1 to 1, normalized to 0-1)
- Intensity: Strength of emotional content (0-1)
- Confidence: Certainty/confidence level in statements (0-1)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    raise ImportError(
        "VADER sentiment analyzer not installed. "
        "Please install: pip install vaderSentiment"
    )

logger = logging.getLogger(__name__)


@dataclass
class EmotionalFeatures:
    """Extracted emotional features."""
    polarity: float  # 0-1: Sentiment (0=negative, 0.5=neutral, 1=positive)
    intensity: float  # 0-1: Emotional intensity
    confidence: float  # 0-1: Confidence/certainty level
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.polarity,
            self.intensity,
            self.confidence
        ])


class EmotionalDimensionExtractor:
    """
    Extracts emotional dimensions from memory content.
    
    Uses VADER sentiment analysis combined with custom patterns
    for confidence and intensity detection.
    """
    
    # Confidence indicators
    CONFIDENCE_KEYWORDS = {
        # High confidence
        "definitely": 0.9,
        "certainly": 0.9,
        "absolutely": 0.9,
        "clearly": 0.85,
        "obviously": 0.85,
        "undoubtedly": 0.85,
        "sure": 0.8,
        "confident": 0.8,
        "certain": 0.8,
        "know": 0.75,
        "believe": 0.7,
        "think": 0.6,
        # Low confidence
        "maybe": 0.3,
        "perhaps": 0.3,
        "possibly": 0.3,
        "might": 0.35,
        "could": 0.35,
        "unsure": 0.2,
        "uncertain": 0.2,
        "doubt": 0.2,
        "guess": 0.25,
        "assume": 0.4,
    }
    
    # Intensity modifiers
    INTENSITY_MODIFIERS = {
        # Amplifiers
        "very": 1.5,
        "extremely": 2.0,
        "incredibly": 2.0,
        "really": 1.3,
        "so": 1.3,
        "quite": 1.2,
        "totally": 1.5,
        "completely": 1.5,
        "absolutely": 1.8,
        "utterly": 1.8,
        # Dampeners
        "somewhat": 0.7,
        "slightly": 0.5,
        "a bit": 0.6,
        "a little": 0.6,
        "fairly": 0.8,
        "rather": 0.8,
        "sort of": 0.6,
        "kind of": 0.6,
    }
    
    # Emotional intensity words
    EMOTION_WORDS = {
        # High intensity
        "love": 0.9,
        "hate": 0.9,
        "furious": 0.95,
        "ecstatic": 0.95,
        "devastated": 0.9,
        "thrilled": 0.85,
        "terrible": 0.85,
        "amazing": 0.85,
        "awful": 0.85,
        "fantastic": 0.8,
        "horrible": 0.8,
        "excellent": 0.75,
        "angry": 0.75,
        "excited": 0.7,
        "upset": 0.7,
        # Medium intensity
        "happy": 0.5,
        "sad": 0.5,
        "good": 0.4,
        "bad": 0.4,
        "pleased": 0.5,
        "disappointed": 0.5,
        "satisfied": 0.4,
        "concerned": 0.5,
        # Low intensity
        "okay": 0.2,
        "fine": 0.2,
        "alright": 0.2,
        "neutral": 0.1,
    }
    
    def __init__(self):
        """Initialize the emotional extractor."""
        self.vader = SentimentIntensityAnalyzer()
    
    def extract(
        self,
        content: str,
        speaker: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> EmotionalFeatures:
        """
        Extract emotional dimensions from memory content.
        
        Args:
            content: Memory content text
            speaker: Who said this (some speakers may be more confident)
            content_type: Type of content (affects interpretation)
            
        Returns:
            EmotionalFeatures with 3 dimensions
        """
        # Get VADER sentiment scores
        vader_scores = self.vader.polarity_scores(content)
        
        # Extract features
        polarity = self._extract_polarity(vader_scores)
        intensity = self._extract_intensity(content, vader_scores)
        confidence = self._extract_confidence(content, content_type)
        
        return EmotionalFeatures(
            polarity=polarity,
            intensity=intensity,
            confidence=confidence
        )
    
    def _extract_polarity(self, vader_scores: Dict[str, float]) -> float:
        """
        Extract sentiment polarity from VADER scores.
        
        Args:
            vader_scores: VADER sentiment scores
            
        Returns:
            Polarity score 0-1 (0=negative, 0.5=neutral, 1=positive)
        """
        # VADER compound score ranges from -1 to 1
        # Normalize to 0-1 range
        compound = vader_scores['compound']
        normalized = (compound + 1) / 2
        
        return normalized
    
    def _extract_intensity(
        self,
        content: str,
        vader_scores: Dict[str, float]
    ) -> float:
        """
        Extract emotional intensity from content.
        
        Args:
            content: Memory content
            vader_scores: VADER sentiment scores
            
        Returns:
            Intensity score 0-1
        """
        content_lower = content.lower()
        
        # Base intensity from VADER (how far from neutral)
        base_intensity = abs(vader_scores['compound'])
        
        # Check for emotion words
        emotion_score = 0.0
        for word, score in self.EMOTION_WORDS.items():
            if word in content_lower:
                emotion_score = max(emotion_score, score)
        
        # Check for intensity modifiers
        modifier_score = 1.0
        for modifier, multiplier in self.INTENSITY_MODIFIERS.items():
            if modifier in content_lower:
                modifier_score *= multiplier
        
        # Combine scores
        # Weight: 40% VADER, 40% emotion words, 20% modifiers
        intensity = (
            0.4 * base_intensity +
            0.4 * emotion_score +
            0.2 * min(1.0, modifier_score - 1.0)  # Convert multiplier to 0-1
        )
        
        # Check for exclamation marks (intensity indicator)
        exclamation_count = content.count('!')
        if exclamation_count > 0:
            intensity = min(1.0, intensity + 0.1 * min(exclamation_count, 3))
        
        # Check for capital letters (intensity indicator)
        if len(content) > 10:  # Avoid short messages
            capital_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if capital_ratio > 0.3:  # Significant capitals
                intensity = min(1.0, intensity + 0.2)
        
        return min(1.0, intensity)
    
    def _extract_confidence(
        self,
        content: str,
        content_type: Optional[str]
    ) -> float:
        """
        Extract confidence/certainty level from content.
        
        Args:
            content: Memory content
            content_type: Type of content
            
        Returns:
            Confidence score 0-1
        """
        content_lower = content.lower()
        
        # Check confidence keywords
        confidence_scores = []
        for keyword, score in self.CONFIDENCE_KEYWORDS.items():
            if keyword in content_lower:
                confidence_scores.append(score)
        
        # Use average of found confidence indicators
        if confidence_scores:
            base_confidence = np.mean(confidence_scores)
        else:
            # Default confidence based on content type
            if content_type in ["decision", "commitment", "finding"]:
                base_confidence = 0.7
            elif content_type in ["hypothesis", "assumption"]:
                base_confidence = 0.5
            elif content_type in ["question", "issue"]:
                base_confidence = 0.4
            else:
                base_confidence = 0.6
        
        # Check for hedging language (reduces confidence)
        hedging_patterns = [
            r'\b(if|whether|depending on|assuming|provided that)\b',
            r'\b(may|might|could|would)\b',
            r'\b(generally|usually|typically|often)\b',
        ]
        
        hedging_count = 0
        for pattern in hedging_patterns:
            hedging_count += len(re.findall(pattern, content_lower))
        
        if hedging_count > 0:
            # Reduce confidence based on hedging
            base_confidence *= (1.0 - min(0.3, hedging_count * 0.1))
        
        # Check for definitive language (increases confidence)
        definitive_patterns = [
            r'\b(will|must|shall|always|never)\b',
            r'\b(fact|proven|confirmed|verified)\b',
        ]
        
        definitive_count = 0
        for pattern in definitive_patterns:
            definitive_count += len(re.findall(pattern, content_lower))
        
        if definitive_count > 0:
            # Increase confidence based on definitive language
            base_confidence = min(1.0, base_confidence + definitive_count * 0.1)
        
        # Question marks reduce confidence
        if '?' in content:
            base_confidence *= 0.7
        
        return base_confidence
    
    def batch_extract(
        self,
        contents: List[str],
        speakers: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract emotional features for multiple memories.
        
        Args:
            contents: List of memory contents
            speakers: List of speakers
            content_types: List of content types
            
        Returns:
            Array of shape (n_memories, 3)
        """
        n_memories = len(contents)
        
        # Prepare lists with None if not provided
        if speakers is None:
            speakers = [None] * n_memories
        if content_types is None:
            content_types = [None] * n_memories
        
        # Extract features
        features = []
        for i in range(n_memories):
            emotional_features = self.extract(
                content=contents[i],
                speaker=speakers[i],
                content_type=content_types[i]
            )
            features.append(emotional_features.to_array())
        
        return np.vstack(features)


# Example usage and testing
if __name__ == "__main__":
    extractor = EmotionalDimensionExtractor()
    
    # Test cases
    test_cases = [
        "I absolutely love this idea! It's fantastic!",
        "I'm not sure about this approach, maybe we should reconsider.",
        "This is definitely the right decision. I'm confident it will work.",
        "I'm extremely disappointed with these results.",
        "The analysis clearly shows positive trends.",
        "Could we possibly explore other options?",
        "I hate to say this, but the project is failing.",
        "Fine, let's proceed with the plan.",
    ]
    
    for content in test_cases:
        features = extractor.extract(content)
        print(f"\nContent: {content}")
        print(f"Polarity: {features.polarity:.2f} (0=neg, 0.5=neutral, 1=pos)")
        print(f"Intensity: {features.intensity:.2f}")
        print(f"Confidence: {features.confidence:.2f}")
