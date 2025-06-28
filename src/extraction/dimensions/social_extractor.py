"""
Social dimension extractor for cognitive features.

This module extracts 3 social dimensions from memory content:
- Authority: Speaker's authority level (0-1)
- Influence: Potential influence on team/decisions (0-1)
- Team dynamics: Team interaction quality (0-1)

NOTE: This is a placeholder implementation that returns default values.
Full implementation will analyze speaker roles, interaction patterns, etc.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SocialFeatures:
    """Extracted social features."""
    authority: float  # 0-1: Speaker's authority level
    influence: float  # 0-1: Potential influence on others
    team_dynamics: float  # 0-1: Team interaction quality
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.authority,
            self.influence,
            self.team_dynamics
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'SocialFeatures':
        """Create from numpy array."""
        if array.shape != (3,):
            raise ValueError(f"Array must be 3D, got {array.shape}")
        return cls(
            authority=float(array[0]),
            influence=float(array[1]),
            team_dynamics=float(array[2])
        )


class SocialDimensionExtractor:
    """
    Extracts social dimensions from memory content.
    
    PLACEHOLDER IMPLEMENTATION: Returns default values of 0.5 for all dimensions.
    
    Future implementation will analyze:
    - Speaker roles and organizational hierarchy
    - Decision-making patterns
    - Team interaction quality
    - Communication patterns
    - Influence indicators
    """
    
    def __init__(self):
        """Initialize the social extractor."""
        logger.info("SocialDimensionExtractor initialized (placeholder implementation)")
    
    def extract(
        self,
        content: str,
        speaker: Optional[str] = None,
        speaker_role: Optional[str] = None,
        participants: Optional[List[str]] = None,
        content_type: Optional[str] = None
    ) -> SocialFeatures:
        """
        Extract social dimensions from memory content.
        
        Args:
            content: Memory content text
            speaker: Who said this
            speaker_role: Role of the speaker (e.g., "manager", "developer")
            participants: List of meeting participants
            content_type: Type of content
            
        Returns:
            SocialFeatures with 3 dimensions
        """
        content_lower = content.lower()

        # Authority
        authority_score = 0.5
        authority_keywords = ["decided by", "approved by", "responsible for", "mandate", "lead", "manager", "director", "head of"]
        for keyword in authority_keywords:
            if keyword in content_lower:
                authority_score += 0.1
        
        if speaker_role:
            if "manager" in speaker_role.lower() or "lead" in speaker_role.lower() or "director" in speaker_role.lower():
                authority_score += 0.2
            elif "executive" in speaker_role.lower() or "ceo" in speaker_role.lower():
                authority_score += 0.3
        
        if content_type == "decision":
            authority_score += 0.1

        # Influence
        influence_score = 0.5
        influence_keywords = ["suggested", "convinced", "led to", "proposed", "recommend", "persuaded", "advocated"]
        for keyword in influence_keywords:
            if keyword in content_lower:
                influence_score += 0.1
        
        if content_type == "action":
            influence_score += 0.1

        # Team Dynamics
        team_dynamics_score = 0.5
        positive_team_keywords = ["team", "collaboration", "working together", "consensus", "agreement", "support", "aligned"]
        negative_team_keywords = ["disagreement", "conflict", "issue", "blocker"]

        for keyword in positive_team_keywords:
            if keyword in content_lower:
                team_dynamics_score += 0.05
        for keyword in negative_team_keywords:
            if keyword in content_lower:
                team_dynamics_score -= 0.05
        
        # Normalize scores to be within [0, 1]
        authority_score = np.clip(authority_score, 0.0, 1.0)
        influence_score = np.clip(influence_score, 0.0, 1.0)
        team_dynamics_score = np.clip(team_dynamics_score, 0.0, 1.0)
            
        return SocialFeatures(
            authority=float(authority_score),
            influence=float(influence_score),
            team_dynamics=float(team_dynamics_score)
        )
    
    def batch_extract(
        self,
        contents: List[str],
        speakers: Optional[List[str]] = None,
        speaker_roles: Optional[List[str]] = None,
        participants_lists: Optional[List[List[str]]] = None,
        content_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract social features for multiple memories.
        
        Args:
            contents: List of memory contents
            speakers: List of speakers
            speaker_roles: List of speaker roles
            participants_lists: List of participant lists
            content_types: List of content types
            
        Returns:
            Array of shape (n_memories, 3)
        """
        n_memories = len(contents)
        
        # Prepare lists with None if not provided
        if speakers is None:
            speakers = [None] * n_memories
        if speaker_roles is None:
            speaker_roles = [None] * n_memories
        if participants_lists is None:
            participants_lists = [None] * n_memories
        if content_types is None:
            content_types = [None] * n_memories
        
        # Extract features
        features = []
        for i in range(n_memories):
            social_features = self.extract(
                content=contents[i],
                speaker=speakers[i],
                speaker_role=speaker_roles[i],
                participants=participants_lists[i],
                content_type=content_types[i]
            )
            features.append(social_features.to_array())
        
        return np.vstack(features)