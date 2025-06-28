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
import calendar
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
    
    # Urgency keywords and phrases with refined scoring
    URGENCY_KEYWORDS = {
        # Critical urgency (0.9-1.0)
        "emergency": 1.0,
        "critical": 1.0,
        "crisis": 1.0,
        "urgent": 0.95,
        "asap": 0.95,
        "immediately": 0.95,
        "right now": 0.95,
        "right away": 0.95,
        "this instant": 0.95,
        "drop everything": 0.95,
        
        # High urgency (0.7-0.9)
        "time-sensitive": 0.85,
        "high priority": 0.85,
        "priority": 0.8,
        "pressing": 0.8,
        "deadline": 0.8,
        "eod": 0.8,
        "end of day": 0.8,
        "by today": 0.8,
        "today": 0.75,
        "before close": 0.75,
        "quick": 0.7,
        "quickly": 0.7,
        "fast": 0.7,
        
        # Medium urgency (0.4-0.7)
        "tomorrow": 0.65,
        "by tomorrow": 0.65,
        "this week": 0.55,
        "next few days": 0.5,
        "within days": 0.5,
        "coming days": 0.5,
        "soon": 0.45,
        "shortly": 0.45,
        "before long": 0.4,
        
        # Low urgency (0.2-0.4)
        "next week": 0.35,
        "this month": 0.3,
        "when possible": 0.25,
        "when convenient": 0.2,
        "at your convenience": 0.2,
        
        # Very low urgency (0.0-0.2)
        "eventually": 0.15,
        "someday": 0.1,
        "no rush": 0.05,
        "no hurry": 0.05,
        "whenever": 0.05,
        "not urgent": 0.0,
    }
    
    # Refined deadline patterns with better date parsing
    DEADLINE_PATTERNS = [
        # Specific dates
        (r"by\s+(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", "full_date"),
        (r"due\s+(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", "full_date"),
        (r"deadline[:\s]+(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", "full_date"),
        
        # Month and day
        (r"by\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})", "month_day"),
        (r"due\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})", "month_day"),
        
        # Relative dates
        (r"by\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", "weekday"),
        (r"by\s+next\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", "next_weekday"),
        (r"within\s+(\d+)\s+hours?", "hours"),
        (r"within\s+(\d+)\s+days?", "days"),
        (r"within\s+(\d+)\s+weeks?", "weeks"),
        (r"within\s+(\d+)\s+months?", "months"),
        
        # Time periods
        (r"by\s+end\s+of\s+(day|week|month|quarter|year)", "period"),
        (r"before\s+end\s+of\s+(day|week|month|quarter|year)", "period"),
        (r"by\s+(eod|eow|eom|eoq|eoy)", "period_abbrev"),
        
        # Specific times
        (r"by\s+(\d{1,2}):(\d{2})\s*(am|pm)?", "time"),
        (r"before\s+(\d{1,2}):(\d{2})\s*(am|pm)?", "time"),
    ]
    
    # Duration relevance indicators with refined scoring
    DURATION_KEYWORDS = {
        # Long-term relevance (0.8-1.0)
        "permanent": 1.0,
        "perpetual": 1.0,
        "forever": 0.95,
        "always": 0.95,
        "strategic": 0.9,
        "long-term": 0.9,
        "long term": 0.9,
        "enduring": 0.85,
        "lasting": 0.85,
        "ongoing": 0.85,
        "continuous": 0.85,
        "sustained": 0.8,
        "persistent": 0.8,
        
        # Policy/framework indicators (0.7-0.9)
        "policy": 0.85,
        "framework": 0.85,
        "principle": 0.85,
        "standard": 0.8,
        "guideline": 0.8,
        "protocol": 0.8,
        "procedure": 0.75,
        "process": 0.75,
        "methodology": 0.75,
        "best practice": 0.7,
        
        # Medium-term relevance (0.4-0.7)
        "project": 0.6,
        "phase": 0.55,
        "milestone": 0.55,
        "quarterly": 0.5,
        "annual": 0.5,
        "yearly": 0.5,
        "seasonal": 0.45,
        "periodic": 0.45,
        
        # Short-term relevance (0.1-0.4)
        "temporary": 0.3,
        "interim": 0.3,
        "provisional": 0.3,
        "short-term": 0.25,
        "short term": 0.25,
        "quick fix": 0.2,
        "for now": 0.2,
        "one-time": 0.15,
        "one time": 0.15,
        "once-off": 0.15,
        "ad hoc": 0.15,
        "this meeting": 0.1,
        "just today": 0.1,
        "right now": 0.1,
    }
    
    # Sequence indicators for better position detection
    SEQUENCE_INDICATORS = {
        # Beginning indicators
        "first": 0.1,
        "start": 0.1,
        "begin": 0.1,
        "kick off": 0.1,
        "opening": 0.1,
        "initial": 0.15,
        "introduce": 0.15,
        
        # Middle indicators
        "next": 0.5,
        "then": 0.5,
        "continue": 0.5,
        "proceed": 0.5,
        "furthermore": 0.5,
        "additionally": 0.5,
        
        # Ending indicators
        "finally": 0.9,
        "lastly": 0.9,
        "conclude": 0.85,
        "wrap up": 0.85,
        "summary": 0.85,
        "in summary": 0.9,
        "to summarize": 0.9,
        "in conclusion": 0.95,
        "closing": 0.9,
    }
    
    def __init__(self):
        """Initialize the temporal extractor with compiled patterns."""
        # Compile all regex patterns for efficiency
        self._compiled_deadline_patterns = [
            (re.compile(pattern, re.IGNORECASE), pattern_type)
            for pattern, pattern_type in self.DEADLINE_PATTERNS
        ]
        
        # Compile urgency phrase patterns for multi-word matching
        self._urgency_phrase_patterns = []
        for phrase, score in self.URGENCY_KEYWORDS.items():
            if ' ' in phrase:  # Multi-word phrases
                pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                self._urgency_phrase_patterns.append((pattern, score))
        
        # Cache for month name to number conversion
        self._month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 9, 'november': 11, 'december': 12
        }
        
        # Weekday to number mapping
        self._weekday_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
    
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
        # Extract individual features with enhanced methods
        urgency = self._extract_urgency_enhanced(content, content_type)
        deadline_proximity = self._extract_deadline_proximity_enhanced(content)
        sequence_position = self._extract_sequence_position_enhanced(
            content, timestamp_ms, meeting_duration_ms
        )
        duration_relevance = self._extract_duration_relevance_enhanced(content, content_type)
        
        # Apply contextual adjustments
        urgency, deadline_proximity = self._apply_contextual_adjustments(
            urgency, deadline_proximity, content_type, speaker
        )
        
        return TemporalFeatures(
            urgency=urgency,
            deadline_proximity=deadline_proximity,
            sequence_position=sequence_position,
            duration_relevance=duration_relevance
        )
    
    def _extract_urgency_enhanced(self, content: str, content_type: Optional[str]) -> float:
        """
        Enhanced urgency extraction with phrase matching and context awareness.
        
        Returns:
            Urgency score 0-1
        """
        content_lower = content.lower()
        
        # Find all urgency scores
        urgency_scores = []
        
        # Check multi-word phrases first (higher priority)
        for pattern, score in self._urgency_phrase_patterns:
            if pattern.search(content_lower):
                urgency_scores.append(score)
        
        # Check single words
        words = content_lower.split()
        for word in words:
            # Remove punctuation for better matching
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.URGENCY_KEYWORDS:
                urgency_scores.append(self.URGENCY_KEYWORDS[clean_word])
        
        # Calculate base urgency
        if urgency_scores:
            # Use weighted average: highest score gets more weight
            urgency_scores.sort(reverse=True)
            weights = [1.0 / (i + 1) for i in range(len(urgency_scores))]
            weighted_sum = sum(s * w for s, w in zip(urgency_scores, weights))
            weight_total = sum(weights)
            base_urgency = weighted_sum / weight_total
        else:
            base_urgency = 0.3  # Default baseline
        
        # Content type boost
        content_type_boosts = {
            "action": 1.2,
            "commitment": 1.2,
            "decision": 1.15,
            "risk": 1.15,
            "issue": 1.1,
            "deliverable": 1.1,
            "milestone": 1.1,
        }
        
        if content_type and content_type in content_type_boosts:
            base_urgency *= content_type_boosts[content_type]
        
        # Linguistic intensity indicators
        intensity_boost = 0.0
        
        # Exclamation marks
        exclamation_count = content.count('!')
        if exclamation_count > 0:
            intensity_boost += min(0.15, exclamation_count * 0.05)
        
        # All caps words (excluding common acronyms)
        words = content.split()
        caps_words = [
            w for w in words 
            if w.isupper() and len(w) > 2 and not self._is_common_acronym(w)
        ]
        if caps_words:
            intensity_boost += min(0.15, len(caps_words) * 0.05)
        
        # Question marks in urgent context
        if '?' in content and any(word in content_lower for word in ['urgent', 'asap', 'deadline']):
            intensity_boost += 0.1
        
        # Repetition of urgency words
        urgency_word_count = sum(
            1 for word in words 
            if re.sub(r'[^\w\s]', '', word.lower()) in self.URGENCY_KEYWORDS
        )
        if urgency_word_count > 1:
            intensity_boost += min(0.1, (urgency_word_count - 1) * 0.05)
        
        # Combine scores
        final_urgency = min(1.0, base_urgency + intensity_boost)
        
        return final_urgency
    
    def _extract_deadline_proximity_enhanced(self, content: str) -> float:
        """
        Enhanced deadline extraction with better date parsing and scoring.
        
        Returns:
            Deadline proximity score 0-1 (1 = very close deadline)
        """
        # Track all found deadlines
        deadline_scores = []
        
        for pattern, pattern_type in self._compiled_deadline_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                score = self._calculate_deadline_score_enhanced(
                    match, pattern_type, content
                )
                if score is not None:
                    deadline_scores.append(score)
        
        # Check for implicit deadlines
        implicit_score = self._check_implicit_deadlines(content)
        if implicit_score > 0:
            deadline_scores.append(implicit_score)
        
        # Return the most urgent (highest) deadline score
        if deadline_scores:
            return max(deadline_scores)
        
        return 0.0
    
    def _calculate_deadline_score_enhanced(
        self, 
        match: re.Match, 
        pattern_type: str,
        full_content: str
    ) -> Optional[float]:
        """
        Enhanced deadline scoring with context awareness.
        
        Args:
            match: Regex match object
            pattern_type: Type of deadline pattern
            full_content: Full content for context
            
        Returns:
            Score 0-1 based on deadline proximity, or None if parsing fails
        """
        try:
            current_date = datetime.now()
            deadline_date = None
            
            if pattern_type == "full_date":
                # Parse full date
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                if year < 100:  # Two-digit year
                    year += 2000
                deadline_date = datetime(year, month, day)
                
            elif pattern_type == "month_day":
                # Parse month and day
                month_name = match.group(1).lower()
                day = int(match.group(2))
                month = self._month_map.get(month_name)
                if month:
                    year = current_date.year
                    # If date has passed this year, assume next year
                    deadline_date = datetime(year, month, day)
                    if deadline_date < current_date:
                        deadline_date = datetime(year + 1, month, day)
            
            elif pattern_type == "weekday":
                # Calculate next occurrence of weekday
                weekday_name = match.group(1).lower()
                target_weekday = self._weekday_map.get(weekday_name)
                if target_weekday is not None:
                    days_ahead = (target_weekday - current_date.weekday()) % 7
                    if days_ahead == 0:  # Today, assume next week
                        days_ahead = 7
                    deadline_date = current_date + timedelta(days=days_ahead)
            
            elif pattern_type == "next_weekday":
                # Next occurrence (7-13 days)
                weekday_name = match.group(1).lower()
                target_weekday = self._weekday_map.get(weekday_name)
                if target_weekday is not None:
                    days_ahead = (target_weekday - current_date.weekday()) % 7 + 7
                    deadline_date = current_date + timedelta(days=days_ahead)
            
            elif pattern_type == "hours":
                hours = int(match.group(1))
                deadline_date = current_date + timedelta(hours=hours)
            
            elif pattern_type == "days":
                days = int(match.group(1))
                deadline_date = current_date + timedelta(days=days)
            
            elif pattern_type == "weeks":
                weeks = int(match.group(1))
                deadline_date = current_date + timedelta(weeks=weeks)
            
            elif pattern_type == "months":
                months = int(match.group(1))
                # Approximate months as 30 days
                deadline_date = current_date + timedelta(days=months * 30)
            
            elif pattern_type == "period":
                period = match.group(1).lower()
                period_days = {
                    "day": 1,
                    "week": 7,
                    "month": 30,
                    "quarter": 90,
                    "year": 365
                }
                days = period_days.get(period, 7)
                deadline_date = current_date + timedelta(days=days)
            
            elif pattern_type == "period_abbrev":
                abbrev = match.group(1).lower()
                abbrev_days = {
                    "eod": 1,
                    "eow": 7,
                    "eom": 30,
                    "eoq": 90,
                    "eoy": 365
                }
                days = abbrev_days.get(abbrev, 7)
                deadline_date = current_date + timedelta(days=days)
            
            elif pattern_type == "time":
                # Time today or tomorrow
                hour = int(match.group(1))
                minute = int(match.group(2))
                am_pm = match.group(3)
                
                if am_pm and am_pm.lower() == 'pm' and hour < 12:
                    hour += 12
                elif am_pm and am_pm.lower() == 'am' and hour == 12:
                    hour = 0
                
                deadline_date = current_date.replace(hour=hour, minute=minute)
                if deadline_date < current_date:
                    deadline_date += timedelta(days=1)
            
            # Calculate proximity score
            if deadline_date:
                days_until = (deadline_date - current_date).total_seconds() / 86400
                
                # Apply context modifiers
                modifier = self._get_deadline_context_modifier(full_content)
                
                # Score calculation with exponential decay
                if days_until <= 0:
                    score = 1.0  # Overdue
                elif days_until <= 1:
                    score = 0.95
                elif days_until <= 3:
                    score = 0.85
                elif days_until <= 7:
                    score = 0.7
                elif days_until <= 14:
                    score = 0.5
                elif days_until <= 30:
                    score = 0.3
                elif days_until <= 90:
                    score = 0.15
                else:
                    score = 0.05
                
                return min(1.0, score * modifier)
            
        except Exception as e:
            logger.debug(f"Error parsing deadline: {e}")
        
        return None
    
    def _check_implicit_deadlines(self, content: str) -> float:
        """Check for implicit deadline indicators."""
        content_lower = content.lower()
        
        implicit_patterns = [
            (r"before\s+(?:the|our)\s+(?:meeting|presentation|demo|review)", 0.8),
            (r"in time for\s+(?:the|our)", 0.75),
            (r"ready\s+(?:by|for)\s+(?:the|our)", 0.7),
            (r"need(?:s|ed)?\s+to\s+be\s+(?:done|ready|complete)", 0.65),
            (r"expect(?:ing|ed)?\s+(?:by|before)", 0.6),
        ]
        
        max_score = 0.0
        for pattern, score in implicit_patterns:
            if re.search(pattern, content_lower):
                max_score = max(max_score, score)
        
        return max_score
    
    def _get_deadline_context_modifier(self, content: str) -> float:
        """Get modifier based on deadline context."""
        content_lower = content.lower()
        
        # Strict deadline indicators
        if any(word in content_lower for word in ['must', 'mandatory', 'required', 'critical']):
            return 1.2
        
        # Flexible deadline indicators
        if any(word in content_lower for word in ['try to', 'if possible', 'ideally', 'preferably']):
            return 0.8
        
        return 1.0
    
    def _extract_sequence_position_enhanced(
        self,
        content: str,
        timestamp_ms: Optional[int],
        meeting_duration_ms: Optional[int]
    ) -> float:
        """
        Enhanced sequence position extraction using both timestamp and content.
        
        Args:
            content: Memory content
            timestamp_ms: When this was said
            meeting_duration_ms: Total meeting duration
            
        Returns:
            Position score 0-1 (0 = beginning, 1 = end)
        """
        # Base position from timestamp
        if timestamp_ms is not None and meeting_duration_ms is not None and meeting_duration_ms > 0:
            timestamp_position = timestamp_ms / meeting_duration_ms
            timestamp_weight = 0.7
        else:
            timestamp_position = 0.5
            timestamp_weight = 0.0
        
        # Content-based position detection
        content_lower = content.lower()
        sequence_scores = []
        
        for phrase, score in self.SEQUENCE_INDICATORS.items():
            if phrase in content_lower:
                sequence_scores.append(score)
        
        # Calculate content position
        if sequence_scores:
            content_position = np.mean(sequence_scores)
            content_weight = 1.0 - timestamp_weight
        else:
            content_position = 0.5
            content_weight = 0.0
        
        # Check for explicit meeting phase indicators
        phase_modifiers = {
            "agenda": 0.1,
            "kick.?off": 0.05,
            "introduct": 0.1,
            "overview": 0.15,
            "update": 0.3,
            "discuss": 0.5,
            "next steps": 0.8,
            "action items": 0.85,
            "wrap.?up": 0.9,
            "adjourn": 0.95,
        }
        
        for pattern, modifier in phase_modifiers.items():
            if re.search(pattern, content_lower):
                # Give phase indicators high weight
                phase_weight = 0.3
                timestamp_weight *= (1 - phase_weight)
                content_weight *= (1 - phase_weight)
                
                final_position = (
                    timestamp_position * timestamp_weight +
                    content_position * content_weight +
                    modifier * phase_weight
                )
                return max(0.0, min(1.0, final_position))
        
        # Combine timestamp and content positions
        final_position = (
            timestamp_position * timestamp_weight +
            content_position * content_weight
        )
        
        return max(0.0, min(1.0, final_position))
    
    def _extract_duration_relevance_enhanced(
        self,
        content: str,
        content_type: Optional[str]
    ) -> float:
        """
        Enhanced duration relevance extraction with better context understanding.
        
        Returns:
            Duration relevance score 0-1 (1 = long-term relevance)
        """
        content_lower = content.lower()
        
        # Find all duration indicators
        duration_scores = []
        
        # Check duration keywords
        for keyword, score in self.DURATION_KEYWORDS.items():
            if keyword in content_lower:
                duration_scores.append(score)
        
        # Base score from keywords
        if duration_scores:
            # Weight towards higher scores for long-term relevance
            duration_scores.sort(reverse=True)
            weights = [1.0 / (i + 1) for i in range(len(duration_scores))]
            weighted_sum = sum(s * w for s, w in zip(duration_scores, weights))
            weight_total = sum(weights)
            base_score = weighted_sum / weight_total
        else:
            # Default based on content type
            content_type_defaults = {
                "principle": 0.8,
                "framework": 0.8,
                "policy": 0.8,
                "strategy": 0.75,
                "decision": 0.7,
                "commitment": 0.65,
                "insight": 0.6,
                "finding": 0.6,
                "recommendation": 0.55,
                "action": 0.4,
                "task": 0.35,
                "issue": 0.3,
                "question": 0.3,
            }
            base_score = content_type_defaults.get(content_type, 0.5)
        
        # Analyze temporal scope indicators
        scope_modifiers = self._analyze_temporal_scope(content_lower)
        
        # Check for change/evolution indicators
        change_indicators = [
            "going forward", "from now on", "moving forward",
            "new approach", "new process", "new standard",
            "change", "update", "revise", "evolve"
        ]
        
        if any(indicator in content_lower for indicator in change_indicators):
            base_score = max(base_score, 0.6)
        
        # Apply future tense boost
        future_patterns = [
            r'\bwill\s+\w+',
            r'\bgoing\s+to\s+\w+',
            r'\bplan(?:s|ning)?\s+to',
            r'\bintend(?:s|ing)?\s+to',
            r'\bstrateg(?:y|ic)',
            r'\broadmap',
            r'\bvision',
        ]
        
        future_count = sum(1 for pattern in future_patterns if re.search(pattern, content_lower))
        if future_count > 0:
            future_boost = min(0.15, future_count * 0.05)
            base_score = min(1.0, base_score + future_boost)
        
        # Apply past tense reduction for temporary items
        past_patterns = [
            r'\bwas\s+\w+',
            r'\bwere\s+\w+',
            r'\bhad\s+\w+',
            r'\bused\s+to',
            r'\bpreviously',
        ]
        
        past_count = sum(1 for pattern in past_patterns if re.search(pattern, content_lower))
        if past_count > future_count:  # More past than future focus
            base_score *= 0.8
        
        # Combine with scope modifiers
        final_score = min(1.0, base_score * scope_modifiers)
        
        return final_score
    
    def _analyze_temporal_scope(self, content_lower: str) -> float:
        """Analyze the temporal scope mentioned in content."""
        scope_patterns = {
            # Broad scope (multiplier > 1)
            r'\b(?:company|organization).?wide\b': 1.2,
            r'\b(?:enterprise|global|universal)\b': 1.2,
            r'\b(?:across\s+all|throughout)\b': 1.15,
            r'\b(?:standard\s+practice|best\s+practice)\b': 1.15,
            
            # Narrow scope (multiplier < 1)
            r'\b(?:this\s+project|current\s+sprint)\b': 0.8,
            r'\b(?:pilot|trial|test|experiment)\b': 0.7,
            r'\b(?:specific\s+to|only\s+for|just\s+for)\b': 0.7,
            r'\b(?:exception|special\s+case)\b': 0.6,
        }
        
        multiplier = 1.0
        for pattern, modifier in scope_patterns.items():
            if re.search(pattern, content_lower):
                multiplier *= modifier
        
        return multiplier
    
    def _apply_contextual_adjustments(
        self,
        urgency: float,
        deadline_proximity: float,
        content_type: Optional[str],
        speaker: Optional[str]
    ) -> Tuple[float, float]:
        """Apply contextual adjustments to urgency and deadline scores."""
        # If high deadline proximity but low urgency, boost urgency
        if deadline_proximity > 0.7 and urgency < 0.5:
            urgency = min(1.0, urgency + (deadline_proximity - 0.7))
        
        # If high urgency but no deadline, slightly reduce urgency
        if urgency > 0.7 and deadline_proximity < 0.1:
            urgency *= 0.9
        
        # Content type relationships
        if content_type == "action" and deadline_proximity < 0.3:
            # Actions without deadlines get moderate urgency
            urgency = max(urgency, 0.5)
            deadline_proximity = max(deadline_proximity, 0.3)
        
        return urgency, deadline_proximity
    
    def _is_common_acronym(self, word: str) -> bool:
        """Check if a word is a common acronym."""
        common_acronyms = {
            'API', 'UI', 'UX', 'SQL', 'HTML', 'CSS', 'JSON', 'XML',
            'CEO', 'CTO', 'CFO', 'VP', 'HR', 'IT', 'QA', 'PM',
            'KPI', 'ROI', 'B2B', 'B2C', 'SaaS', 'MVP', 'POC',
            'USA', 'UK', 'EU', 'GDP', 'AI', 'ML', 'VR', 'AR'
        }
        return word in common_acronyms
    
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
