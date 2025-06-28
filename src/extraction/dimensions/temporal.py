

import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

class TemporalDimensionExtractor:
    """
    Extracts temporal dimensions from text.

    Dimensions (4D):
    - urgency: How immediate is the action/information? (0-1)
    - deadline_proximity: How close is a mentioned deadline? (0-1)
    - sequence_position: Where does this memory fit in a sequence? (0-1)
    - duration_relevance: How relevant is the duration mentioned? (0-1)
    """

    def __init__(self):
        self.urgency_keywords = ["urgent", "asap", "immediately", "now", "critical", "soon"]
        self.deadline_keywords = ["by", "before", "due", "deadline"]
        self.duration_keywords = ["minutes", "hours", "days", "weeks", "months", "years"]

    def extract(self, text: str, context: Dict[str, Any] = None) -> np.ndarray:
        """
        Extracts temporal dimensions from the given text and context.

        Args:
            text: The input text.
            context: Optional dictionary with additional context, e.g., current_time, total_duration.

        Returns:
            A 4D numpy array representing temporal dimensions.
        """
        urgency = self._calculate_urgency(text)
        deadline_proximity = self._calculate_deadline_proximity(text, context)
        sequence_position = self._calculate_sequence_position(context)
        duration_relevance = self._calculate_duration_relevance(text)

        return np.array([urgency, deadline_proximity, sequence_position, duration_relevance])

    def _calculate_urgency(self, text: str) -> float:
        """Calculates urgency based on keywords."""
        text_lower = text.lower()
        for keyword in self.urgency_keywords:
            if keyword in text_lower:
                return 1.0  # High urgency
        return 0.0

    def _calculate_deadline_proximity(self, text: str, context: Dict[str, Any] = None) -> float:
        """Calculates deadline proximity based on mentioned dates/times."""
        current_time = context.get("current_time", datetime.now()) if context else datetime.now()
        
        # Look for explicit dates (e.g., "by July 1st", "due next Friday")
        # This is a simplified example, a real implementation would use a more robust NLP date parser
        match = re.search(r"(by|due|before)\s+((\w+\s+\d{1,2}(?:st|nd|rd|th)?)|(next\s+\w+))", text, re.IGNORECASE)
        if match:
            # Placeholder: In a real system, parse the date and compare to current_time
            # For now, if a deadline is mentioned, assume some proximity
            return 0.8 # High proximity if a deadline is explicitly mentioned

        return 0.0 # No deadline mentioned or implied

    def _calculate_sequence_position(self, context: Dict[str, Any] = None) -> float:
        """Calculates sequence position (e.g., for memories in a meeting transcript)."""
        if context and "total_memories" in context and "current_memory_index" in context:
            total = context["total_memories"]
            current = context["current_memory_index"]
            if total > 0:
                return current / total
        return 0.5  # Default to middle if no context

    def _calculate_duration_relevance(self, text: str) -> float:
        """Calculates relevance of mentioned durations."""
        text_lower = text.lower()
        for keyword in self.duration_keywords:
            if keyword in text_lower:
                # Simple heuristic: presence of duration keyword implies some relevance
                return 0.7
        return 0.0

if __name__ == "__main__":
    extractor = TemporalDimensionExtractor()

    # Test cases
    print("--- Temporal Dimension Extraction Tests ---")

    # Test 1: High urgency
    text1 = "We need to complete this task immediately."
    dims1 = extractor.extract(text1)
    print(f"'{text1}' -> Urgency: {dims1[0]:.2f}, Deadline: {dims1[1]:.2f}, Sequence: {dims1[2]:.2f}, Duration: {dims1[3]:.2f}")
    assert dims1[0] == 1.0

    # Test 2: Deadline proximity
    text2 = "The report is due by next Friday."
    dims2 = extractor.extract(text2)
    print(f"'{text2}' -> Urgency: {dims2[0]:.2f}, Deadline: {dims2[1]:.2f}, Sequence: {dims2[2]:.2f}, Duration: {dims2[3]:.2f}")
    assert dims2[1] > 0.0

    # Test 3: Sequence position with context
    text3 = "This is the final point of discussion."
    context3 = {"current_memory_index": 9, "total_memories": 10}
    dims3 = extractor.extract(text3, context3)
    print(f"'{text3}' (context: {context3}) -> Urgency: {dims3[0]:.2f}, Deadline: {dims3[1]:.2f}, Sequence: {dims3[2]:.2f}, Duration: {dims3[3]:.2f}")
    assert dims3[2] == 0.9

    # Test 4: Duration relevance
    text4 = "The meeting will last for two hours."
    dims4 = extractor.extract(text4)
    print(f"'{text4}' -> Urgency: {dims4[0]:.2f}, Deadline: {dims4[1]:.2f}, Sequence: {dims4[2]:.2f}, Duration: {dims4[3]:.2f}")
    assert dims4[3] > 0.0

    # Test 5: No temporal indicators
    text5 = "This is a general statement."
    dims5 = extractor.extract(text5)
    print(f"'{text5}' -> Urgency: {dims5[0]:.2f}, Deadline: {dims5[1]:.2f}, Sequence: {dims5[2]:.2f}, Duration: {dims5[3]:.2f}")
    assert np.allclose(dims5, [0.0, 0.0, 0.5, 0.0])

    print("\nAll temporal dimension tests passed!")

