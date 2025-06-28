"""
Causal dimension extractor for meeting content analysis.

Extracts 3 causal dimensions:
- Dependencies (0-1): Level of interdependence with other items
- Impact (0-1): Potential effect magnitude
- Risk Factors (0-1): Uncertainty and risk level
"""

import re
from typing import List, Optional, Dict, Any
import numpy as np

from ...models.entities import Memory, ContentType


class CausalDimensionExtractor:
    """Extracts causal relationships and impact dimensions."""
    
    # Dependency indicators
    DEPENDENCY_PATTERNS = {
        'high': [
            r'\b(depend[s]?\s+on|requires?|prerequisite|blocker|blocking)\b',
            r'\b(must\s+have|need[s]?\s+to\s+be|contingent\s+on|relies?\s+on)\b',
            r'\b(can\'t\s+proceed|stuck\s+until|waiting\s+for|blocked\s+by)\b',
            r'\b(critical\s+path|bottleneck|dependency\s+chain)\b',
        ],
        'medium': [
            r'\b(related\s+to|connected\s+with|influences?|affects?)\b',
            r'\b(should\s+coordinate|align\s+with|integrate\s+with)\b',
            r'\b(downstream|upstream|parallel\s+work)\b',
        ],
        'low': [
            r'\b(independent|standalone|separate|isolated)\b',
            r'\b(can\s+proceed|self.?contained|autonomous)\b',
            r'\b(optional|nice\s+to\s+have|when\s+possible)\b',
        ]
    }
    
    # Impact indicators
    IMPACT_PATTERNS = {
        'high': [
            r'\b(critical|crucial|vital|essential|fundamental)\b',
            r'\b(major\s+impact|significant\s+effect|game.?changer)\b',
            r'\b(transform|revolutionize|disrupt|breakthrough)\b',
            r'\b(company.?wide|organization.?wide|strategic)\b',
            r'\b(million|billion|massive|enormous)\b',
        ],
        'medium': [
            r'\b(important|substantial|notable|meaningful)\b',
            r'\b(affect[s]?\s+multiple|cross.?functional|department.?wide)\b',
            r'\b(thousand|significant|considerable)\b',
            r'\b(improve|enhance|optimize|streamline)\b',
        ],
        'low': [
            r'\b(minor|small|slight|minimal|negligible)\b',
            r'\b(local|limited|specific|narrow)\b',
            r'\b(tweak|adjust|fine.?tune|polish)\b',
            r'\b(cosmetic|superficial|surface.?level)\b',
        ]
    }
    
    # Risk indicators
    RISK_PATTERNS = {
        'high': [
            r'\b(high\s+risk|dangerous|hazardous|threat)\b',
            r'\b(could\s+fail|might\s+break|unstable|fragile)\b',
            r'\b(unknown|uncertain|unpredictable|volatile)\b',
            r'\b(experimental|untested|prototype|beta)\b',
            r'\b(compliance|legal|regulatory|security)\s+risk\b',
        ],
        'medium': [
            r'\b(some\s+risk|moderate\s+risk|manageable\s+risk)\b',
            r'\b(challenges?|concerns?|issues?|problems?)\b',
            r'\b(may|might|could|possibly)\b',
            r'\b(assumptions?|hypothesis|theory|estimate)\b',
        ],
        'low': [
            r'\b(low\s+risk|safe|proven|tested|stable)\b',
            r'\b(confident|certain|sure|guaranteed)\b',
            r'\b(standard|routine|typical|normal)\b',
            r'\b(well.?understood|documented|established)\b',
        ]
    }
    
    # Causal relationship words
    CAUSAL_KEYWORDS = [
        'because', 'therefore', 'hence', 'thus', 'consequently',
        'as a result', 'leads to', 'causes', 'results in', 'due to',
        'owing to', 'thanks to', 'since', 'so', 'accordingly'
    ]
    
    def extract(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        linked_memories: Optional[List[str]] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract causal dimensions from text.
        
        Args:
            text: The content to analyze
            content_type: Type of content (decision, action, etc.)
            linked_memories: IDs of related memories
            project_context: Additional project context
            
        Returns:
            numpy array of shape (3,) with values [dependencies, impact, risk_factors]
        """
        text_lower = text.lower()
        
        # Calculate dependency score
        dependencies = self._calculate_dependencies(
            text_lower, 
            linked_memories,
            project_context
        )
        
        # Calculate impact score
        impact = self._calculate_impact(
            text_lower,
            content_type,
            project_context
        )
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(
            text_lower,
            content_type,
            project_context
        )
        
        return np.array([dependencies, impact, risk_factors], dtype=np.float32)
    
    def _calculate_dependencies(
        self,
        text: str,
        linked_memories: Optional[List[str]] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate dependency level based on text and context."""
        score = 0.2  # Base score
        
        # Check dependency patterns
        high_count = sum(1 for pattern in self.DEPENDENCY_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.DEPENDENCY_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.DEPENDENCY_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.35 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.2 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.15 * min(low_count / 2, 1.0)
        
        # Check for causal keywords
        causal_count = sum(1 for keyword in self.CAUSAL_KEYWORDS 
                          if keyword in text)
        score += 0.1 * min(causal_count / 3, 1.0)
        
        # Boost based on linked memories
        if linked_memories:
            link_boost = min(len(linked_memories) / 5, 1.0) * 0.2
            score += link_boost
        
        # Consider project complexity
        if project_context:
            if project_context.get('team_size', 0) > 10:
                score += 0.1
            if project_context.get('integration_points', 0) > 5:
                score += 0.15
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_impact(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate potential impact magnitude."""
        score = 0.25  # Base score
        
        # Check impact patterns
        high_count = sum(1 for pattern in self.IMPACT_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.IMPACT_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.IMPACT_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.4 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.2 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.1 * min(low_count / 2, 1.0)
        
        # Boost for decision content
        if content_type == ContentType.DECISION:
            score += 0.15
        elif content_type == ContentType.ACTION:
            score += 0.1
        
        # Check for quantitative impact
        number_pattern = r'\b\d+[%$kKmMbB]?\b'
        if re.search(number_pattern, text):
            score += 0.1
        
        # Consider project scope
        if project_context:
            scope = project_context.get('scope', '')
            if scope in ['company-wide', 'enterprise', 'global']:
                score += 0.2
            elif scope in ['department', 'team']:
                score += 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_risk_factors(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate risk and uncertainty level."""
        score = 0.3  # Base score
        
        # Check risk patterns
        high_count = sum(1 for pattern in self.RISK_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.RISK_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.RISK_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.35 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.15 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.2 * min(low_count / 2, 1.0)
        
        # Check for uncertainty markers
        uncertainty_markers = [
            r'\b(maybe|perhaps|possibly|potentially|might)\b',
            r'\b(assume|guess|think|believe|hope)\b',
            r'\b(if|whether|depends?|contingent)\b',
            r'\b(TBD|TBA|unknown|unclear|undefined)\b'
        ]
        uncertainty_count = sum(1 for pattern in uncertainty_markers 
                               if re.search(pattern, text, re.IGNORECASE))
        score += 0.1 * min(uncertainty_count / 4, 1.0)
        
        # Consider issue/question content types
        if content_type in [ContentType.ISSUE, ContentType.QUESTION]:
            score += 0.1
        
        # Project maturity affects risk
        if project_context:
            maturity = project_context.get('maturity', 'unknown')
            if maturity in ['prototype', 'alpha', 'beta']:
                score += 0.15
            elif maturity in ['stable', 'mature', 'production']:
                score -= 0.1
        
        return np.clip(score, 0.0, 1.0)