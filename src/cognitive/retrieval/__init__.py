"""
Cognitive retrieval system with activation, similarity, and bridge discovery.

This module contains:
- ContextualRetrieval: High-level retrieval coordinator
- SimilaritySearch: Cosine similarity with recency bias
- BridgeDiscovery: Serendipitous connection finding
"""

from .contextual_retrieval import (
    ContextualRetrieval,
    ContextualRetrievalResult
)

from .similarity_search import (
    SimilaritySearch,
    SearchResult
)

from .bridge_discovery import (
    SimpleBridgeDiscovery,
    BridgeMemory
)

__all__ = [
    # Contextual retrieval
    "ContextualRetrieval",
    "ContextualRetrievalResult",
    
    # Similarity search
    "SimilaritySearch",
    "SearchResult",
    
    # Bridge discovery
    "SimpleBridgeDiscovery",
    "BridgeMemory"
]
