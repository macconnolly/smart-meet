from typing import Dict, Any
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')



class EmotionalDimensionExtractor:
    """
    Extracts emotional dimensions from text using VADER sentiment analysis.

    Dimensions (3D):
    - polarity: Overall sentiment (compound score, -1 to 1, normalized to 0-1)
    - intensity: Strength of emotion (max of positive/negative/neutral, normalized to 0-1)
    - confidence: How confident is the statement? (placeholder, always 0.5 for now)
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract(self, text: str, context: Dict[str, Any] = None) -> np.ndarray:
        """
        Extracts emotional dimensions from the given text.

        Args:
            text: The input text.
            context: Optional dictionary with additional context (not used for VADER).

        Returns:
            A 3D numpy array representing emotional dimensions.
        """
        if not text.strip():
            return np.array([0.5, 0.0, 0.5]) # Neutral polarity, no intensity, default confidence

        sentiment = self.analyzer.polarity_scores(text)

        # Polarity: compound score normalized from [-1, 1] to [0, 1]
        polarity = (sentiment['compound'] + 1) / 2

        # Intensity: max of positive, negative, or neutral scores
        intensity = max(sentiment['pos'], sentiment['neg'], sentiment['neu'])

        # Confidence: Placeholder for now, always 0.5
        confidence = 0.5

        return np.array([polarity, intensity, confidence])

if __name__ == "__main__":
    extractor = EmotionalDimensionExtractor()

    # Test cases
    print("--- Emotional Dimension Extraction Tests ---")

    # Test 1: Positive sentiment
    text1 = "I am extremely happy with the results! This is fantastic."
    dims1 = extractor.extract(text1)
    print(f"'{text1}' -> Polarity: {dims1[0]:.2f}, Intensity: {dims1[1]:.2f}, Confidence: {dims1[2]:.2f}")
    assert dims1[0] > 0.75 # Should be high positive
    assert dims1[1] > 0.5 # Should have some intensity

    # Test 2: Negative sentiment
    text2 = "This is a terrible idea and I am very disappointed."
    dims2 = extractor.extract(text2)
    print(f"'{text2}' -> Polarity: {dims2[0]:.2f}, Intensity: {dims2[1]:.2f}, Confidence: {dims2[2]:.2f}")
    assert dims2[0] < 0.25 # Should be high negative
    assert dims2[1] > 0.5 # Should have some intensity

    # Test 3: Neutral sentiment
    text3 = "The meeting is scheduled for tomorrow at 10 AM."
    dims3 = extractor.extract(text3)
    print(f"'{text3}' -> Polarity: {dims3[0]:.2f}, Intensity: {dims3[1]:.2f}, Confidence: {dims3[2]:.2f}")
    assert 0.4 < dims3[0] < 0.6 # Should be neutral
    assert dims3[1] > 0.5 # Neutral also has intensity

    # Test 4: Mixed sentiment (VADER handles this)
    text4 = "The product is good, but the service was awful."
    dims4 = extractor.extract(text4)
    print(f"'{text4}' -> Polarity: {dims4[0]:.2f}, Intensity: {dims4[1]:.2f}, Confidence: {dims4[2]:.2f}")
    assert 0.4 < dims4[0] < 0.6 # Should be somewhat neutral due to mixed

    # Test 5: Empty text
    text5 = ""
    dims5 = extractor.extract(text5)
    print(f"'{text5}' -> Polarity: {dims5[0]:.2f}, Intensity: {dims5[1]:.2f}, Confidence: {dims5[2]:.2f}")
    assert np.allclose(dims5, [0.5, 0.0, 0.5])

    print("\nAll emotional dimension tests passed!")
