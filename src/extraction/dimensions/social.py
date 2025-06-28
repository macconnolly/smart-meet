"""
Social dimension extractor for meeting content analysis.

Extracts 3 social dimensions:
- Authority (0-1): Level of decision-making power
- Influence (0-1): Impact on team dynamics
- Team Dynamics (0-1): Collaboration and interaction level
"""

import re
from typing import List, Optional, Dict, Any
import numpy as np

from ...models.entities import Memory, ContentType


class SocialDimensionExtractor:
    """Extracts social dimensions from meeting content."""
    
    # Authority indicators
    AUTHORITY_PATTERNS = {
        'high': [
            r'\b(decide[ds]?|decision|approved?|authorize[ds]?|direct[eds]?)\b',
            r'\b(mandate[ds]?|order[eds]?|instruct[eds]?|command[eds]?)\b',
            r'\b(CEO|CTO|VP|director|manager|lead|head)\b',
            r'\b(final say|executive decision|strategic direction)\b',
        ],
        'medium': [
            r'\b(recommend[eds]?|suggest[eds]?|propose[ds]?|advise[ds]?)\b',
            r'\b(should|ought to|need to|must)\b',
            r'\b(team lead|coordinator|supervisor)\b',
        ],
        'low': [
            r'\b(think|believe|feel|wonder|maybe|perhaps)\b',
            r'\b(could|might|possibly|potentially)\b',
            r'\b(team member|contributor|participant)\b',
        ]
    }
    
    # Influence indicators
    INFLUENCE_PATTERNS = {
        'high': [
            r'\b(convinced?|persuaded?|influenced?|changed?\s+mind)\b',
            r'\b(everyone agreed|consensus reached|team aligned)\b',
            r'\b(key insight|breakthrough|game.?changer)\b',
            r'\b(shifted|transformed|revolutionized)\b',
        ],
        'medium': [
            r'\b(good point|interesting|worth considering)\b',
            r'\b(affects?|impacts?|influences?|shapes?)\b',
            r'\b(contribut[eds]?|add[eds]?|brought up)\b',
        ],
        'low': [
            r'\b(mentioned|noted|observed|pointed out)\b',
            r'\b(fyi|for what it\'s worth|just saying)\b',
            r'\b(minor|small|slight)\b',
        ]
    }
    
    # Team dynamics indicators
    COLLABORATION_KEYWORDS = [
        'we', 'us', 'our', 'team', 'together', 'collaborate',
        'cooperation', 'joint', 'shared', 'collective', 'group'
    ]
    
    INTERACTION_PATTERNS = [
        r'\b(agree[ds]?|align[eds]?|support[eds]?|endorse[ds]?)\b',
        r'\b(build[s]?\s+on|expand[eds]?\s+on|add[eds]?\s+to)\b',
        r'\b(discuss[eds]?|debate[ds]?|dialogue|conversation)\b',
        r'\b(feedback|input|thoughts|ideas|suggestions)\b',
    ]
    
    def extract(
        self,
        text: str,
        speaker_role: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        meeting_participants: Optional[List[str]] = None,
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract social dimensions from text.
        
        Args:
            text: The content to analyze
            speaker_role: Role of the speaker (e.g., "Manager", "Developer")
            content_type: Type of content (decision, action, etc.)
            meeting_participants: List of participants in the meeting
            interaction_context: Additional context about interactions
            
        Returns:
            numpy array of shape (3,) with values [authority, influence, team_dynamics]
        """
        text_lower = text.lower()
        
        # Calculate authority score
        authority = self._calculate_authority(text_lower, speaker_role, content_type)
        
        # Calculate influence score
        influence = self._calculate_influence(
            text_lower, 
            meeting_participants,
            interaction_context
        )
        
        # Calculate team dynamics score
        team_dynamics = self._calculate_team_dynamics(
            text_lower,
            meeting_participants,
            interaction_context
        )
        
        return np.array([authority, influence, team_dynamics], dtype=np.float32)
    
    def _calculate_authority(
        self,
        text: str,
        speaker_role: Optional[str] = None,
        content_type: Optional[ContentType] = None
    ) -> float:
        """Calculate authority level based on language and context."""
        score = 0.3  # Base score
        
        # Check authority patterns
        high_count = sum(1 for pattern in self.AUTHORITY_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.AUTHORITY_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.AUTHORITY_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.3 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.15 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.1 * min(low_count / 3, 1.0)
        
        # Boost for speaker role
        if speaker_role:
            role_lower = speaker_role.lower()
            if any(title in role_lower for title in ['ceo', 'cto', 'vp', 'director', 'head']):
                score += 0.25
            elif any(title in role_lower for title in ['manager', 'lead', 'supervisor']):
                score += 0.15
        
        # Boost for decision content
        if content_type == ContentType.DECISION:
            score += 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_influence(
        self,
        text: str,
        meeting_participants: Optional[List[str]] = None,
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate influence on team and decisions."""
        score = 0.25  # Base score
        
        # Check influence patterns
        high_count = sum(1 for pattern in self.INFLUENCE_PATTERNS['high'] 
                        if re.search(pattern, text, re.IGNORECASE))
        medium_count = sum(1 for pattern in self.INFLUENCE_PATTERNS['medium'] 
                          if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in self.INFLUENCE_PATTERNS['low'] 
                       if re.search(pattern, text, re.IGNORECASE))
        
        # Weight based on pattern matches
        if high_count > 0:
            score += 0.35 * min(high_count / 2, 1.0)
        if medium_count > 0:
            score += 0.2 * min(medium_count / 3, 1.0)
        if low_count > 0:
            score -= 0.05 * min(low_count / 3, 1.0)
        
        # Check if others are mentioned (indicates influence)
        if meeting_participants and len(meeting_participants) > 1:
            mentions = sum(1 for p in meeting_participants if p.lower() in text)
            if mentions > 1:
                score += 0.15
        
        # Boost based on interaction context
        if interaction_context:
            if interaction_context.get('responses_generated', 0) > 2:
                score += 0.1
            if interaction_context.get('ideas_adopted', False):
                score += 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_team_dynamics(
        self,
        text: str,
        meeting_participants: Optional[List[str]] = None,
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate team collaboration and interaction level."""
        score = 0.2  # Base score
        
        # Count collaboration keywords
        collab_count = sum(1 for word in self.COLLABORATION_KEYWORDS 
                          if f' {word} ' in f' {text} ')
        score += 0.3 * min(collab_count / 5, 1.0)
        
        # Check interaction patterns
        interaction_count = sum(1 for pattern in self.INTERACTION_PATTERNS 
                               if re.search(pattern, text, re.IGNORECASE))
        score += 0.25 * min(interaction_count / 3, 1.0)
        
        # Boost for multiple participants mentioned
        if meeting_participants and len(meeting_participants) > 2:
            participant_mentions = sum(1 for p in meeting_participants 
                                     if p.lower() in text)
            if participant_mentions >= 2:
                score += 0.15
        
        # Check for inclusive language
        inclusive_patterns = [
            r'\b(everyone|all of us|the team|collectively)\b',
            r'\b(let\'s|shall we|can we|should we)\b',
            r'\b(thoughts\?|ideas\?|feedback\?|input\?)\b'
        ]
        inclusive_count = sum(1 for pattern in inclusive_patterns 
                             if re.search(pattern, text, re.IGNORECASE))
        score += 0.1 * min(inclusive_count / 2, 1.0)
        
        return np.clip(score, 0.0, 1.0)