"""
Dimension caching for improved performance.

This module provides caching for dimension extraction results to avoid
recomputing dimensions for identical content.
"""

import logging
import hashlib
import time
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
import json
import numpy as np
from collections import OrderedDict

from .dimension_analyzer import CognitiveDimensions

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the dimension cache."""
    dimensions: CognitiveDimensions
    content_hash: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dimensions": self.dimensions.to_array().tolist(),
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        dimensions_array = np.array(data["dimensions"])
        dimensions = CognitiveDimensions.from_array(dimensions_array)
        
        return cls(
            dimensions=dimensions,
            content_hash=data["content_hash"],
            timestamp=data["timestamp"],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time())
        )


class DimensionCache:
    """
    LRU cache for dimension extraction results.
    
    Caches cognitive dimensions based on content hash to avoid
    recomputing for identical content.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600,
        enable_stats: bool = True
    ):
        """
        Initialize the dimension cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_stats: Whether to track cache statistics
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_stats = enable_stats
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        logger.info(
            f"DimensionCache initialized with max_size={max_size}, "
            f"ttl={ttl_seconds}s"
        )
    
    def _compute_hash(
        self,
        content: str,
        context_dict: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute hash for content and context.
        
        Args:
            content: Memory content
            context_dict: Serializable context dictionary
            
        Returns:
            Hash string
        """
        # Create a stable string representation
        hash_input = content
        
        if context_dict:
            # Sort keys for stable hashing
            sorted_context = json.dumps(context_dict, sort_keys=True)
            hash_input = f"{content}||{sorted_context}"
        
        # Use SHA256 for good distribution
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def get(
        self,
        content: str,
        context_dict: Optional[Dict[str, Any]] = None
    ) -> Optional[CognitiveDimensions]:
        """
        Get dimensions from cache.
        
        Args:
            content: Memory content
            context_dict: Serializable context dictionary
            
        Returns:
            Cached CognitiveDimensions or None if not found/expired
        """
        content_hash = self._compute_hash(content, context_dict)
        
        if content_hash not in self._cache:
            if self.enable_stats:
                self._stats["misses"] += 1
            return None
        
        entry = self._cache[content_hash]
        
        # Check TTL
        age = time.time() - entry.timestamp
        if age > self.ttl_seconds:
            # Expired
            del self._cache[content_hash]
            if self.enable_stats:
                self._stats["expired"] += 1
                self._stats["misses"] += 1
            return None
        
        # Update access info and move to end (most recent)
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._cache.move_to_end(content_hash)
        
        if self.enable_stats:
            self._stats["hits"] += 1
        
        return entry.dimensions
    
    def put(
        self,
        content: str,
        dimensions: CognitiveDimensions,
        context_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store dimensions in cache.
        
        Args:
            content: Memory content
            dimensions: Extracted dimensions
            context_dict: Serializable context dictionary
        """
        content_hash = self._compute_hash(content, context_dict)
        
        # Check if we need to evict
        if len(self._cache) >= self.max_size and content_hash not in self._cache:
            # Evict least recently used (first item)
            evicted_key = next(iter(self._cache))
            del self._cache[evicted_key]
            if self.enable_stats:
                self._stats["evictions"] += 1
        
        # Create and store entry
        entry = CacheEntry(
            dimensions=dimensions,
            content_hash=content_hash,
            timestamp=time.time()
        )
        
        self._cache[content_hash] = entry
        # Move to end (most recent)
        self._cache.move_to_end(content_hash)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("DimensionCache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "size": self.size(),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "expired": self._stats["expired"],
            "total_requests": total_requests
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
    
    def get_top_accessed(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N most accessed entries.
        
        Args:
            n: Number of entries to return
            
        Returns:
            List of (content_hash, access_count) tuples
        """
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            (entry.content_hash, entry.access_count)
            for _, entry in sorted_entries[:n]
        ]
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def save_to_dict(self) -> Dict[str, Any]:
        """
        Save cache to dictionary for persistence.
        
        Returns:
            Dictionary representation of cache
        """
        return {
            "entries": {
                key: entry.to_dict()
                for key, entry in self._cache.items()
            },
            "stats": self._stats,
            "config": {
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load cache from dictionary.
        
        Args:
            data: Dictionary representation of cache
        """
        self._cache.clear()
        
        # Load entries
        for key, entry_data in data.get("entries", {}).items():
            try:
                entry = CacheEntry.from_dict(entry_data)
                self._cache[key] = entry
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
        
        # Load stats
        if "stats" in data:
            self._stats.update(data["stats"])
        
        logger.info(f"Loaded {len(self._cache)} entries from cache data")


# Global cache instance
_dimension_cache: Optional[DimensionCache] = None


def get_dimension_cache(
    max_size: int = 10000,
    ttl_seconds: float = 3600
) -> DimensionCache:
    """
    Get or create the global dimension cache.
    
    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL for entries
        
    Returns:
        DimensionCache instance
    """
    global _dimension_cache
    
    if _dimension_cache is None:
        _dimension_cache = DimensionCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    
    return _dimension_cache


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from .dimension_analyzer import get_dimension_analyzer, DimensionExtractionContext
    
    async def test_cache():
        # Create cache and analyzer
        cache = get_dimension_cache(max_size=100, ttl_seconds=60)
        analyzer = get_dimension_analyzer()
        
        # Test content
        test_contents = [
            "We need to implement caching for better performance.",
            "The deadline for this feature is next Friday.",
            "I'm very excited about the new architecture!",
            "We need to implement caching for better performance.",  # Duplicate
        ]
        
        # Extract dimensions with caching
        for i, content in enumerate(test_contents):
            context = DimensionExtractionContext(
                content_type="insight",
                timestamp_ms=i * 1000
            )
            
            # Create context dict for caching
            context_dict = {
                "content_type": context.content_type,
                "timestamp_ms": context.timestamp_ms
            }
            
            # Check cache first
            cached_dims = cache.get(content, context_dict)
            
            if cached_dims:
                print(f"\nContent {i+1}: Cache HIT")
                dimensions = cached_dims
            else:
                print(f"\nContent {i+1}: Cache MISS")
                dimensions = await analyzer.analyze(content, context)
                cache.put(content, dimensions, context_dict)
            
            print(f"  Dimensions: {dimensions.to_dict()}")
        
        # Show cache stats
        print(f"\nCache stats: {cache.get_stats()}")
    
    # Run test
    asyncio.run(test_cache())