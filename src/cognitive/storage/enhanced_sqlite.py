"""
SQLite persistence layer for cognitive memory metadata and connection graph.

This module implements structured data storage using SQLite with a complete
database schema for memory metadata, connection graphs, bridge cache, and
retrieval statistics to support the cognitive memory system.

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Enhanced Persistence
"""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from ...models.entities import Memory, MemoryConnection
from ...storage.sqlite.repositories import MemoryRepository, MemoryConnectionRepository
from ...storage.sqlite.connection import DatabaseConnection


class EnhancedDatabaseManager:
    """SQLite database manager with cognitive memory enhancements."""

    def __init__(self, db_path: str = "data/cognitive_memory.db"):
        """
        Initialize enhanced database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with enhanced schema
        self._initialize_enhanced_database()

    def _initialize_enhanced_database(self) -> None:
        """Initialize database with cognitive memory enhancements."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create cognitive memory metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cognitive_metadata (
                        memory_id TEXT PRIMARY KEY,
                        cognitive_embedding TEXT,  -- JSON array of 400D vector
                        decay_rate REAL DEFAULT 0.1,
                        importance_score REAL DEFAULT 0.0,
                        consolidation_status TEXT DEFAULT 'none',
                        consolidation_date REAL,
                        source_episodic_id TEXT,
                        last_activated REAL,
                        activation_count INTEGER DEFAULT 0,
                        bridge_potential REAL DEFAULT 0.0,
                        FOREIGN KEY (memory_id) REFERENCES memories(id)
                    )
                """)

                # Create bridge cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bridge_cache (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        bridge_score REAL NOT NULL,
                        novelty_score REAL NOT NULL,
                        connection_potential REAL NOT NULL,
                        explanation TEXT,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        FOREIGN KEY (source_id) REFERENCES memories(id),
                        FOREIGN KEY (target_id) REFERENCES memories(id)
                    )
                """)

                # Create retrieval statistics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retrieval_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT NOT NULL,
                        query_vector TEXT,  -- JSON array
                        retrieval_method TEXT NOT NULL,
                        memories_retrieved INTEGER NOT NULL,
                        core_memories INTEGER DEFAULT 0,
                        peripheral_memories INTEGER DEFAULT 0,
                        bridge_memories INTEGER DEFAULT 0,
                        retrieval_time_ms REAL NOT NULL,
                        parameters TEXT,  -- JSON parameters
                        created_at REAL NOT NULL
                    )
                """)

                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cognitive_decay 
                    ON cognitive_metadata(decay_rate, importance_score)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cognitive_consolidation 
                    ON cognitive_metadata(consolidation_status, consolidation_date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bridge_scores 
                    ON bridge_cache(bridge_score DESC, expires_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_retrieval_time 
                    ON retrieval_stats(created_at DESC)
                """)

                conn.commit()
                logger.info("Enhanced database schema initialized", db_path=str(self.db_path))

        except Exception as e:
            logger.error("Failed to initialize enhanced database", error=str(e))
            raise

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper context management."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access

            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            # Set performance optimizations
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA temp_store = MEMORY")

            yield conn

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Database operation failed", error=str(e))
            raise
        finally:
            if conn:
                conn.close()

    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space and optimize performance."""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuumed successfully")
                return True
        except Exception as e:
            logger.error("Failed to vacuum database", error=str(e))
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get enhanced database statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}
                
                # Basic table counts
                tables = [
                    "memories",
                    "memory_connections",
                    "cognitive_metadata",
                    "bridge_cache",
                    "retrieval_stats",
                ]

                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                        count = cursor.fetchone()["count"]
                        stats[f"{table}_count"] = count
                    except:
                        stats[f"{table}_count"] = 0

                # Cognitive-specific stats
                cursor.execute("""
                    SELECT 
                        consolidation_status,
                        COUNT(*) as count
                    FROM cognitive_metadata
                    GROUP BY consolidation_status
                """)
                consolidation_stats = {}
                for row in cursor.fetchall():
                    consolidation_stats[row["consolidation_status"]] = row["count"]
                stats["consolidation_breakdown"] = consolidation_stats

                # Get database file size
                stats["database_size_bytes"] = self.db_path.stat().st_size

                return stats

        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            return {"error": str(e)}


class CognitiveMetadataStore:
    """Enhanced metadata storage for cognitive memory features."""

    def __init__(self, db_manager: EnhancedDatabaseManager):
        """Initialize cognitive metadata store."""
        self.db_manager = db_manager

    def store_cognitive_metadata(
        self, 
        memory_id: str,
        cognitive_embedding: Optional[np.ndarray] = None,
        decay_rate: float = 0.1,
        importance_score: float = 0.0,
        consolidation_status: str = "none",
        source_episodic_id: Optional[str] = None
    ) -> bool:
        """Store cognitive metadata for a memory."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Serialize embedding if provided
                embedding_json = None
                if cognitive_embedding is not None:
                    embedding_json = json.dumps(cognitive_embedding.tolist())

                cursor.execute("""
                    INSERT OR REPLACE INTO cognitive_metadata (
                        memory_id, cognitive_embedding, decay_rate,
                        importance_score, consolidation_status,
                        consolidation_date, source_episodic_id,
                        last_activated, activation_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    embedding_json,
                    decay_rate,
                    importance_score,
                    consolidation_status,
                    time.time() if consolidation_status == "consolidated" else None,
                    source_episodic_id,
                    time.time(),
                    0
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(
                "Failed to store cognitive metadata",
                memory_id=memory_id,
                error=str(e)
            )
            return False

    def get_cognitive_metadata(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cognitive metadata for a memory."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM cognitive_metadata WHERE memory_id = ?
                """, (memory_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Deserialize embedding if present
                metadata = dict(row)
                if metadata.get("cognitive_embedding"):
                    try:
                        embedding_list = json.loads(metadata["cognitive_embedding"])
                        metadata["cognitive_embedding"] = np.array(embedding_list)
                    except:
                        metadata["cognitive_embedding"] = None

                return metadata

        except Exception as e:
            logger.error(
                "Failed to get cognitive metadata",
                memory_id=memory_id,
                error=str(e)
            )
            return None

    def update_activation_stats(
        self, 
        memory_id: str,
        increment_count: bool = True
    ) -> bool:
        """Update activation statistics for a memory."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if increment_count:
                    cursor.execute("""
                        UPDATE cognitive_metadata
                        SET activation_count = activation_count + 1,
                            last_activated = ?
                        WHERE memory_id = ?
                    """, (time.time(), memory_id))
                else:
                    cursor.execute("""
                        UPDATE cognitive_metadata
                        SET last_activated = ?
                        WHERE memory_id = ?
                    """, (time.time(), memory_id))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(
                "Failed to update activation stats",
                memory_id=memory_id,
                error=str(e)
            )
            return False

    def get_consolidation_candidates(
        self,
        min_activation_count: int = 3,
        min_importance_score: float = 0.6,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get memories that are candidates for consolidation."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT 
                        cm.*,
                        m.content,
                        m.memory_type
                    FROM cognitive_metadata cm
                    JOIN memories m ON cm.memory_id = m.id
                    WHERE cm.consolidation_status = 'none'
                    AND cm.activation_count >= ?
                    AND cm.importance_score >= ?
                    ORDER BY cm.importance_score DESC, cm.activation_count DESC
                    LIMIT ?
                """, (min_activation_count, min_importance_score, limit))

                candidates = []
                for row in cursor.fetchall():
                    candidates.append(dict(row))

                return candidates

        except Exception as e:
            logger.error("Failed to get consolidation candidates", error=str(e))
            return []


class BridgeCacheStore:
    """Storage for discovered bridge memories with expiration."""

    def __init__(self, db_manager: EnhancedDatabaseManager):
        """Initialize bridge cache store."""
        self.db_manager = db_manager
        self.default_ttl_hours = 24  # Default cache TTL

    def store_bridge(
        self,
        source_id: str,
        target_id: str,
        bridge_score: float,
        novelty_score: float,
        connection_potential: float,
        explanation: str,
        ttl_hours: Optional[float] = None
    ) -> bool:
        """Store a discovered bridge in cache."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                now = time.time()
                ttl = ttl_hours or self.default_ttl_hours
                expires_at = now + (ttl * 3600)

                # Generate unique ID
                bridge_id = f"{source_id}_{target_id}_{int(now)}"

                cursor.execute("""
                    INSERT OR REPLACE INTO bridge_cache (
                        id, source_id, target_id, bridge_score,
                        novelty_score, connection_potential,
                        explanation, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bridge_id,
                    source_id,
                    target_id,
                    bridge_score,
                    novelty_score,
                    connection_potential,
                    explanation,
                    now,
                    expires_at
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error("Failed to store bridge", error=str(e))
            return False

    def get_bridges_for_memories(
        self,
        memory_ids: List[str],
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get cached bridges for a set of memories."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Clean expired bridges first
                self._clean_expired_bridges(cursor)

                # Get active bridges
                placeholders = ",".join(["?" for _ in memory_ids])
                cursor.execute(f"""
                    SELECT * FROM bridge_cache
                    WHERE (source_id IN ({placeholders}) 
                           OR target_id IN ({placeholders}))
                    AND bridge_score >= ?
                    AND expires_at > ?
                    ORDER BY bridge_score DESC
                """, memory_ids + memory_ids + [min_score, time.time()])

                bridges = []
                for row in cursor.fetchall():
                    bridges.append(dict(row))

                return bridges

        except Exception as e:
            logger.error("Failed to get bridges", error=str(e))
            return []

    def _clean_expired_bridges(self, cursor: sqlite3.Cursor) -> None:
        """Remove expired bridges from cache."""
        cursor.execute("""
            DELETE FROM bridge_cache
            WHERE expires_at <= ?
        """, (time.time(),))


class RetrievalStatsTracker:
    """Track retrieval statistics for analysis and optimization."""

    def __init__(self, db_manager: EnhancedDatabaseManager):
        """Initialize retrieval stats tracker."""
        self.db_manager = db_manager

    def track_retrieval(
        self,
        query_id: str,
        query_vector: Optional[np.ndarray],
        retrieval_method: str,
        memories_retrieved: int,
        core_memories: int = 0,
        peripheral_memories: int = 0,
        bridge_memories: int = 0,
        retrieval_time_ms: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track a retrieval operation."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Serialize query vector if provided
                query_vector_json = None
                if query_vector is not None:
                    query_vector_json = json.dumps(query_vector.tolist())

                # Serialize parameters
                params_json = json.dumps(parameters) if parameters else None

                cursor.execute("""
                    INSERT INTO retrieval_stats (
                        query_id, query_vector, retrieval_method,
                        memories_retrieved, core_memories,
                        peripheral_memories, bridge_memories,
                        retrieval_time_ms, parameters, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id,
                    query_vector_json,
                    retrieval_method,
                    memories_retrieved,
                    core_memories,
                    peripheral_memories,
                    bridge_memories,
                    retrieval_time_ms,
                    params_json,
                    time.time()
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error("Failed to track retrieval", error=str(e))
            return False

    def get_retrieval_analytics(
        self,
        hours_back: float = 24.0,
        retrieval_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get retrieval analytics for the specified time period."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                since_time = time.time() - (hours_back * 3600)

                # Build query
                query = """
                    SELECT 
                        retrieval_method,
                        COUNT(*) as total_retrievals,
                        AVG(memories_retrieved) as avg_memories,
                        AVG(core_memories) as avg_core,
                        AVG(peripheral_memories) as avg_peripheral,
                        AVG(bridge_memories) as avg_bridges,
                        AVG(retrieval_time_ms) as avg_time_ms,
                        MIN(retrieval_time_ms) as min_time_ms,
                        MAX(retrieval_time_ms) as max_time_ms
                    FROM retrieval_stats
                    WHERE created_at >= ?
                """
                params = [since_time]

                if retrieval_method:
                    query += " AND retrieval_method = ?"
                    params.append(retrieval_method)

                query += " GROUP BY retrieval_method"

                cursor.execute(query, params)

                analytics = {
                    "time_period_hours": hours_back,
                    "methods": {}
                }

                for row in cursor.fetchall():
                    method_stats = dict(row)
                    method = method_stats.pop("retrieval_method")
                    analytics["methods"][method] = method_stats

                return analytics

        except Exception as e:
            logger.error("Failed to get retrieval analytics", error=str(e))
            return {"error": str(e)}


class EnhancedConnectionGraphStore:
    """Enhanced connection graph storage with cognitive features."""

    def __init__(
        self, 
        db_manager: EnhancedDatabaseManager,
        connection_repo: MemoryConnectionRepository
    ):
        """Initialize enhanced connection graph store."""
        self.db_manager = db_manager
        self.connection_repo = connection_repo

    async def add_cognitive_connection(
        self,
        source_id: str,
        target_id: str,
        connection_strength: float,
        connection_type: str,
        shared_stakeholders: Optional[List[str]] = None,
        shared_deliverables: Optional[List[str]] = None,
        meeting_proximity: float = 0.0,
        cross_meeting_link: bool = False
    ) -> bool:
        """Add a connection with cognitive metadata."""
        try:
            # Create base connection
            connection = MemoryConnection(
                source_id=source_id,
                target_id=target_id,
                connection_strength=connection_strength,
                connection_type=connection_type
            )

            # Add cognitive metadata
            connection.metadata = {
                "shared_stakeholders": shared_stakeholders or [],
                "shared_deliverables": shared_deliverables or [],
                "meeting_proximity": meeting_proximity,
                "cross_meeting_link": cross_meeting_link
            }

            # Store using repository
            return await self.connection_repo.create(connection)

        except Exception as e:
            logger.error(
                "Failed to add cognitive connection",
                source_id=source_id,
                target_id=target_id,
                error=str(e)
            )
            return False

    async def get_stakeholder_connections(
        self,
        stakeholder_name: str,
        project_id: str,
        min_strength: float = 0.0
    ) -> List[MemoryConnection]:
        """Get all connections between memories mentioning a stakeholder."""
        try:
            connections = await self.connection_repo.get_all()
            
            # Filter connections that mention the stakeholder
            stakeholder_connections = []
            for conn in connections:
                if conn.metadata and "shared_stakeholders" in conn.metadata:
                    if stakeholder_name in conn.metadata["shared_stakeholders"]:
                        if conn.connection_strength >= min_strength:
                            stakeholder_connections.append(conn)

            return stakeholder_connections

        except Exception as e:
            logger.error(
                "Failed to get stakeholder connections",
                stakeholder=stakeholder_name,
                error=str(e)
            )
            return []

    async def get_deliverable_network(
        self,
        deliverable_id: str
    ) -> Dict[str, List[MemoryConnection]]:
        """Get network of memories connected to a deliverable."""
        try:
            connections = await self.connection_repo.get_all()
            
            # Build deliverable network
            network = {
                "direct": [],      # Directly linked to deliverable
                "indirect": [],    # Linked through other memories
                "related": []      # Share the deliverable reference
            }

            for conn in connections:
                if conn.metadata and "shared_deliverables" in conn.metadata:
                    if deliverable_id in conn.metadata["shared_deliverables"]:
                        network["related"].append(conn)

            return network

        except Exception as e:
            logger.error(
                "Failed to get deliverable network",
                deliverable_id=deliverable_id,
                error=str(e)
            )
            return {"direct": [], "indirect": [], "related": []}


def create_enhanced_sqlite_persistence(
    db_path: str = "data/cognitive_memory.db",
) -> tuple[
    CognitiveMetadataStore, 
    BridgeCacheStore, 
    RetrievalStatsTracker,
    EnhancedConnectionGraphStore
]:
    """
    Factory function to create enhanced SQLite persistence components.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of persistence components
    """
    db_manager = EnhancedDatabaseManager(db_path)
    
    # Create stores
    metadata_store = CognitiveMetadataStore(db_manager)
    bridge_store = BridgeCacheStore(db_manager)
    stats_tracker = RetrievalStatsTracker(db_manager)
    
    # For connection graph, we need the repository
    # This is a placeholder - in real implementation, get from DI
    connection_repo = None  # Will be injected
    graph_store = EnhancedConnectionGraphStore(db_manager, connection_repo)
    
    return metadata_store, bridge_store, stats_tracker, graph_store
