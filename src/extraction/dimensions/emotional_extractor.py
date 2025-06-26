"""
Emotional dimension extractor using VADER sentiment analysis.

Extracts 3D emotional features:
1. Sentiment polarity (-1 to 1, normalized to 0-1)
2. Emotional intensity (0-1)
3. Sentiment confidence (0-1)
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class EmotionalExtractor:
    """
    Extracts emotional dimensions from text using VADER.
    
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    attuned to sentiments expressed in social media and works well for
    meeting transcripts.
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("Initialized VADER sentiment analyzer")
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")
            logger.info("Install with: pip install vaderSentiment")
            self.analyzer = None
    
    def extract(self, text: str) -> np.ndarray:
        """
        Extract 3D emotional features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            3D numpy array with emotional features:
            - Polarity: Sentiment direction (0=negative, 0.5=neutral, 1=positive)
            - Intensity: Strength of emotion (0=weak, 1=strong)
            - Confidence: Confidence in sentiment detection (0=low, 1=high)
        """
        if not self.analyzer:
            # Return neutral values if VADER not available
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        # TODO: Get VADER sentiment scores
        scores = self._get_sentiment_scores(text)
        
        # TODO: Extract polarity (normalize from [-1, 1] to [0, 1])
        polarity = self._normalize_polarity(scores['compound'])
        
        # TODO: Calculate intensity from individual scores
        intensity = self._calculate_intensity(scores)
        
        # TODO: Calculate confidence based on text characteristics
        confidence = self._calculate_confidence(text, scores)
        
        return np.array([polarity, intensity, confidence], dtype=np.float32)
    
    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get VADER sentiment scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'pos', 'neg', 'neu', 'compound' scores
        """
        try:
            # TODO: Use VADER to analyze sentiment
            # Returns dict with keys: pos, neg, neu, compound
            scores = self.analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            # Return neutral scores on error
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
    
    def _normalize_polarity(self, compound_score: float) -> float:
        """
        Normalize VADER compound score from [-1, 1] to [0, 1].
        
        Args:
            compound_score: VADER compound score
            
        Returns:
            Normalized polarity score
        """
        # TODO: Convert from [-1, 1] to [0, 1]
        # -1 (most negative) -> 0
        # 0 (neutral) -> 0.5
        # 1 (most positive) -> 1
        normalized = (compound_score + 1.0) / 2.0
        return np.clip(normalized, 0.0, 1.0)
    
    def _calculate_intensity(self, scores: Dict[str, float]) -> float:
        """
        Calculate emotional intensity from sentiment scores.
        
        Args:
            scores: VADER scores dictionary
            
        Returns:
            Intensity score between 0 and 1
        """
        # TODO: Calculate intensity based on:
        # - Sum of positive and negative scores (not neutral)
        # - Higher pos + neg = more intense emotion
        # - Pure neutral = low intensity
        
        pos = scores.get('pos', 0.0)
        neg = scores.get('neg', 0.0)
        neu = scores.get('neu', 1.0)
        
        # Intensity is high when emotions are strong (either positive or negative)
        emotional_content = pos + neg
        intensity = emotional_content  # Already normalized to [0, 1]
        
        return np.clip(intensity, 0.0, 1.0)
    
    def _calculate_confidence(self, text: str, scores: Dict[str, float]) -> float:
        """
        Calculate confidence in sentiment detection.
        
        Args:
            text: Original text
            scores: VADER scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # TODO: Calculate confidence based on:
        # - Text length (very short text = lower confidence)
        # - Presence of clear sentiment indicators
        # - Consistency of sentiment (not mixed signals)
        
        # Start with base confidence
        confidence = 0.5
        
        # Adjust based on text length
        word_count = len(text.split())
        if word_count < 3:
            confidence *= 0.5
        elif word_count > 10:
            confidence *= 1.2
        
        # Adjust based on sentiment clarity
        # Clear sentiment = one score dominates
        pos = scores.get('pos', 0.0)
        neg = scores.get('neg', 0.0)
        neu = scores.get('neu', 1.0)
        
        max_score = max(pos, neg, neu)
        if max_score > 0.7:
            confidence *= 1.3
        elif max_score < 0.4:
            confidence *= 0.7
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of emotional features.
        
        Returns:
            List of feature names
        """
        return [
            'sentiment_polarity',
            'emotional_intensity',
            'sentiment_confidence'
        ]
    
    def get_sentiment_label(self, polarity: float) -> str:
        """
        Get human-readable sentiment label.
        
        Args:
            polarity: Normalized polarity score (0-1)
            
        Returns:
            Sentiment label
        """
        if polarity < 0.3:
            return "negative"
        elif polarity < 0.7:
            return "neutral"
        else:
            return "positive"


def test_emotional_extractor():
    """Test the emotional extractor."""
    print("Testing EmotionalExtractor...")
    
    extractor = EmotionalExtractor()
    
    # Test cases with expected sentiments
    test_cases = [
        ("This is absolutely fantastic! Great work everyone!", "positive"),
        ("I'm really concerned about this approach.", "negative"),
        ("Let's review the current status.", "neutral"),
        ("This is terrible and completely unacceptable!", "negative"),
        ("I'm thrilled with the progress we've made!", "positive"),
    ]
    
    for text, expected in test_cases:
        features = extractor.extract(text)
        polarity, intensity, confidence = features
        label = extractor.get_sentiment_label(polarity)
        
        print(f"\nText: {text}")
        print(f"Expected: {expected}, Got: {label}")
        print(f"Polarity: {polarity:.3f}")
        print(f"Intensity: {intensity:.3f}")
        print(f"Confidence: {confidence:.3f}")
    
    print("\n✅ EmotionalExtractor tests completed!")


if __name__ == "__main__":
    test_emotional_extractor()