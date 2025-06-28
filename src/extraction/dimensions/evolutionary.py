"""
Evolutionary dimension extractor for meeting content analysis.

Extracts 3 evolutionary dimensions:
- Change Rate (0-1): Speed of evolution/change
- Innovation Level (0-1): Degree of novelty and innovation
- Adaptation Need (0-1): Requirement for adjustment and flexibility
"""

import re
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta

from ...models.entities import Memory, ContentType


class EvolutionaryDimensionExtractor:
    """Extracts evolutionary and change-related dimensions."""
    
    # Change rate indicators
    CHANGE_RATE_PATTERNS = {
        'high': [
            r'\b(rapid[ly]?|quick[ly]?|fast|swift[ly]?|immediate[ly]?)\b',
            r'\b(asap|urgent[ly]?|critical|time.?sensitive|deadline)\b',
            r'\b(accelerat[eds]?|speed[s]?\s+up|ramp[s]?\s+up|scale[s]?\s+fast)\b',
            r'\b(daily|hourly|real.?time|continuous[ly]?)\b',
            r'\b(volatile|dynamic|flux|turbulent)\b',
        ],
        'medium': [
            r'\b(weekly|bi.?weekly|monthly|quarterly)\b',
            r'\b(moderate[ly]?|steady|gradual[ly]?|progressive[ly]?)\b',
            r'\b(evolv[es]?|develop[s]?|progress[es]?|advance[s]?)\b',
            r'\b(iterative[ly]?|incremental[ly]?|phased?)\b',
        ],
        'low': [
            r'\b(slow[ly]?|gradual[ly]?|eventual[ly]?|long.?term)\b',
            r'\b(stable|static|fixed|constant|unchanging)\b',
            r'\b(annual[ly]?|year[ly]?|rare[ly]?|seldom)\b',
            r'\b(maintain|preserve|sustain|keep)\b',
        ]
    }
    
    # Innovation indicators
    INNOVATION_PATTERNS = {
        'high': [
            r'\b(innovati[veon]+|breakthrough|revolutionary|pioneering)\b',
            r'\b(first.?of.?its.?kind|never.?before|unprecedented|unique)\b',
            r'\b(disrupt[iveon]+|transform[ativeon]+|game.?chang[ering]+)\b',
            r'\b(cutting.?edge|state.?of.?the.?art|next.?gen[eration]*)\b',
            r'\b(experiment[al]?|prototype|beta|alpha|proof.?of.?concept)\b',
        ],
        'medium': [
            r'\b(new|novel|fresh|different|alternative)\b',
            r'\b(improv[eds]?|enhanc[eds]?|upgrad[eds]?|optimiz[eds]?)\b',
            r'\b(modern[ize]?|update[ds]?|refactor[eds]?|redesign[eds]?)\b',
            r'\b(creative[ly]?|inventive[ly]?|original)\b',
        ],
        'low': [
            r'\b(traditional|conventional|standard|typical|usual)\b',
            r'\b(proven|established|tested|reliable|trusted)\b',
            r'\b(legacy|existing|current|maintain|preserve)\b',
            r'\b(minor|small|incremental|routine)\b',
        ]
    }
    
    # Adaptation need indicators
    ADAPTATION_PATTERNS = {
        'high': [
            r'\b(pivot|adapt|adjust|flex[ible]+|agile)\b',
            r'\b(chang[eing]+\s+requirements?|moving\s+target|evolving\s+needs?)\b',
            r'\b(uncertain[ty]?|unpredictable|volatile|dynamic)\b',
            r'\b(contingency|backup|fallback|plan\s+[bc])\b',
            r'\b(resilient|robust|fault.?tolerant)\b',
        ],
        'medium': [
            r'\b(configurable|customizable|modular|extensible)\b',
            r'\b(may\s+change|might\s+evolve|could\s+shift)\b',
            r'\b(flexible|adaptable|adjustable|scalable)\b',
            r'\b(version[eds]?|iteration[s]?|revision[s]?)\b',
        ],
        'low': [
            r'\b(fixed|rigid|strict|firm|immutable)\b',
            r'\b(stable|predictable|consistent|reliable)\b',
            r'\b(standard[ized]?|uniform|homogeneous)\b',
            r'\b(final|definitive|permanent|unchangeable)\b',
        ]
    }
    
    # Time-based change indicators
    TIME_CHANGE_KEYWORDS = [
        'evolve', 'change', 'shift', 'transform', 'migrate',
        'transition', 'progress', 'advance', 'develop', 'grow'
    ]
    
    def extract(
        self,
        text: str,
        timestamp_ms: Optional[int] = None,
        previous_versions: Optional[List[Memory]] = None,
        project_timeline: Optional[Dict[str, Any]] = None,
        content_type: Optional[ContentType] = None
    ) -> np.ndarray:
        """
        Extract evolutionary dimensions from text.
        
        Args:
            text: The content to analyze
            timestamp_ms: Timestamp of the memory
            previous_versions: Previous versions of related memories
            project_timeline: Timeline context of the project
            content_type: Type of content
            
        Returns:
            numpy array of shape (3,) with values [change_rate, innovation_level, adaptation_need]
        """
        text_lower = text.lower()
        
        # Calculate change rate
        change_rate = self._calculate_change_rate(
            text_lower,
            timestamp_ms,
            previous_versions,
            project_timeline
        )
        
        # Calculate innovation level
        innovation_level = self._calculate_innovation_level(
            text_lower,
            content_type,
            project_timeline
        )
        
        # Calculate adaptation need
        adaptation_need = self._calculate_adaptation_need(
            text_lower,
            content_type,
            project_timeline
        )
        
        return np.array([change_rate, innovation_level, adaptation_need], dtype=np.float32)
    
    def _calculate_change_rate(
        self,
        text: str,
        timestamp_ms: Optional[int] = None,
        previous_versions: Optional[List[Memory]] = None,
        project_timeline: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate rate of change based on text and temporal context."""
        score = 0.3  # Base score
        
        # Check change rate patterns
        high_count = sum(1 for pattern in self.CHANGE_RATE_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.CHANGE_RATE_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.CHANGE_RATE_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.35 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.15 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.2 * min(low_count / 2, 1.0)
        
        # Check for time change keywords
        change_count = sum(1 for keyword in self.TIME_CHANGE_KEYWORDS 
                          if keyword in text)
        score += 0.1 * min(change_count / 3, 1.0)
        
        # Analyze version frequency if available
        if previous_versions and timestamp_ms:
            current_time = datetime.fromtimestamp(timestamp_ms / 1000)
            recent_versions = [
                v for v in previous_versions 
                if (current_time - datetime.fromtimestamp(v.timestamp_ms / 1000)).days < 30
            ]
            if len(recent_versions) > 5:
                score += 0.2  # High change frequency
            elif len(recent_versions) > 2:
                score += 0.1  # Moderate change frequency
        
        # Consider project phase
        if project_timeline:
            phase = project_timeline.get('current_phase', '')
            if phase in ['startup', 'growth', 'expansion']:
                score += 0.15
            elif phase in ['mature', 'stable', 'maintenance']:
                score -= 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_innovation_level(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        project_timeline: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate innovation and novelty level."""
        score = 0.25  # Base score
        
        # Check innovation patterns
        high_count = sum(1 for pattern in self.INNOVATION_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.INNOVATION_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.INNOVATION_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.4 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.2 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.15 * min(low_count / 2, 1.0)
        
        # Boost for idea content type
        if content_type == ContentType.IDEA:
            score += 0.15
        
        # Check for research/exploration keywords
        research_keywords = [
            'research', 'explore', 'investigate', 'study', 'analyze',
            'experiment', 'test', 'trial', 'pilot', 'poc'
        ]
        research_count = sum(1 for keyword in research_keywords if keyword in text)
        score += 0.1 * min(research_count / 2, 1.0)
        
        # Consider project type
        if project_timeline:
            project_type = project_timeline.get('type', '')
            if project_type in ['r&d', 'innovation', 'research']:
                score += 0.2
            elif project_type in ['maintenance', 'support', 'operations']:
                score -= 0.15
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_adaptation_need(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        project_timeline: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate need for adaptation and flexibility."""
        score = 0.3  # Base score
        
        # Check adaptation patterns
        high_count = sum(1 for pattern in self.ADAPTATION_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.ADAPTATION_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.ADAPTATION_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.35 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.15 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.2 * min(low_count / 2, 1.0)
        
        # Check for conditional language
        conditional_patterns = [
            r'\b(if|when|unless|provided|assuming|depending)\b',
            r'\b(might|may|could|possibly|potentially)\b',
            r'\b(alternative|option|choice|scenario|case)\b'
        ]
        conditional_count = sum(1 for pattern in conditional_patterns 
                               if re.search(pattern, text, re.IGNORECASE))
        score += 0.1 * min(conditional_count / 4, 1.0)
        
        # Issues and questions suggest adaptation needs
        if content_type in [ContentType.ISSUE, ContentType.QUESTION]:
            score += 0.1
        
        # Market/environment factors
        if project_timeline:
            market_volatility = project_timeline.get('market_volatility', 'medium')
            if market_volatility == 'high':
                score += 0.2
            elif market_volatility == 'low':
                score -= 0.1
            
            # Regulatory environment
            if project_timeline.get('regulatory_changes', False):
                score += 0.15
        
        return np.clip(score, 0.0, 1.0)