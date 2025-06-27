"""
Database initialization script for Cognitive Meeting Intelligence.

This script creates and initializes the SQLite database with proper
schema, indexes, and sample data for development and testing.
"""

import asyncio
import logging
from pathlib import Path
import sqlite3
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# TODO: Import these when implemented (Day 1)
# from src.core.config import ConfigManager
# from src.storage.sqlite.memory_repository import initialize_database

# Temporary implementation until Day 1
class ConfigManager:
    async def load_config(self):
        class Config:
            class Database:
                sqlite_path = "data/memories.db"
            database = Database()
        return Config()

async def initialize_database(db_path: str):
    """Initialize database with schema.sql"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read and execute schema
    schema_path = Path(__file__).parent.parent / "src" / "storage" / "sqlite" / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()


async def main():
    """
    @TODO: Main database initialization function.
    
    AGENTIC EMPOWERMENT: Automated database setup enables
    quick deployment and consistent development environments.
    """
    print("🔧 Initializing Cognitive Meeting Intelligence Database")
    
    try:
        # TODO: Load configuration
        config_manager = ConfigManager()
        config = await config_manager.load_config()
        
        # TODO: Ensure data directory exists
        db_path = Path(config.database.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Database path: {db_path}")
        
        # TODO: Check if database already exists
        if db_path.exists():
            response = input("Database already exists. Recreate? (y/N): ")
            if response.lower() != 'y':
                print("❌ Database initialization cancelled")
                return
            
            # Backup existing database
            backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            db_path.rename(backup_path)
            print(f"💾 Existing database backed up to: {backup_path}")
        
        # TODO: Initialize database schema
        print("🏗️  Creating database schema...")
        await initialize_database(str(db_path))
        
        # TODO: Create indexes for performance
        print("📊 Creating database indexes...")
        await create_performance_indexes(str(db_path))
        
        # TODO: Insert sample data if requested
        response = input("Insert sample data for testing? (y/N): ")
        if response.lower() == 'y':
            print("📝 Inserting sample data...")
            await insert_sample_data(str(db_path))
        
        # TODO: Verify database setup
        print("✅ Verifying database setup...")
        await verify_database_setup(str(db_path))
        
        print("🎉 Database initialization completed successfully!")
        print(f"📍 Database location: {db_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        logging.exception("Database initialization error")
        return 1
    
    return 0


async def create_performance_indexes(db_path: str):
    """
    @TODO: Create database indexes for optimal performance.
    
    AGENTIC EMPOWERMENT: Proper indexing ensures fast queries
    and system responsiveness under load.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # TODO: Memory table indexes
        indexes = [
            # Primary access patterns
            "CREATE INDEX IF NOT EXISTS idx_memories_meeting_id ON memories(meeting_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_content_type ON memories(content_type)",
            
            # Consolidation queries
            "CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count)",
            "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)",
            
            # Connection queries (updated table name)
            "CREATE INDEX IF NOT EXISTS idx_connections_source ON memory_connections(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_connections_target ON memory_connections(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_connections_type ON memory_connections(connection_type)",
            
            # Meeting queries
            "CREATE INDEX IF NOT EXISTS idx_meetings_start_time ON meetings(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at)",
            
            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_memories_type_date ON memories(memory_type, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_meeting_type ON memories(meeting_id, memory_type)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
            print(f"  ✓ Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
        
        conn.commit()
        
    finally:
        conn.close()


async def insert_sample_data(db_path: str):
    """
    @TODO: Insert sample data for development and testing.
    
    AGENTIC EMPOWERMENT: Sample data enables immediate testing
    and development without requiring real meeting data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # TODO: Sample meetings
        sample_meetings = [
            {
                'id': 'meeting_001',
                'title': 'Product Strategy Q1 Planning',
                'start_time': '2024-01-15 10:00:00',
                'end_time': '2024-01-15 11:30:00',
                'participants_json': '["Alice Johnson", "Bob Smith", "Charlie Davis"]',
                'transcript_path': 'data/transcripts/meeting_001.txt',
                'metadata_json': '{"location": "Conference Room A", "recorded": true}'
            },
            {
                'id': 'meeting_002', 
                'title': 'Technical Architecture Review',
                'start_time': '2024-01-16 14:00:00',
                'end_time': '2024-01-16 16:00:00',
                'participants_json': '["David Wilson", "Eve Brown", "Frank Miller"]',
                'transcript_path': 'data/transcripts/meeting_002.txt',
                'metadata_json': '{"location": "Virtual", "recorded": true}'
            },
            {
                'id': 'meeting_003',
                'title': 'Weekly Team Sync',
                'start_time': '2024-01-17 09:00:00',
                'end_time': '2024-01-17 09:30:00', 
                'participants_json': '["Alice Johnson", "Bob Smith", "Grace Lee"]',
                'transcript_path': None,
                'metadata_json': '{"location": "Virtual", "recorded": false}'
            }
        ]
        
        for meeting in sample_meetings:
            cursor.execute("""
                INSERT INTO meetings (id, title, start_time, end_time, participants_json, transcript_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting['id'], meeting['title'], meeting['start_time'], 
                meeting['end_time'], meeting['participants_json'], 
                meeting['transcript_path'], meeting['metadata_json']
            ))
        
        print(f"  ✓ Inserted {len(sample_meetings)} sample meetings")
        
        # TODO: Sample memories
        sample_memories = [
            {
                'id': 'mem_001',
                'content': 'We decided to prioritize the mobile app development for Q1, targeting iOS first with Android following in Q2.',
                'memory_type': 'decision',
                'content_type': 'strategic',
                'meeting_id': 'meeting_001',
                'speaker': 'Alice Johnson',
                'timestamp_ms': 900000,  # 15 minutes into meeting
                'level': 2,  # L2 episodic
                'qdrant_id': 'qdrant_mem_001',
                'dimensions_json': '{"temporal": [0.8, 0.9, 0.2, 0.5], "emotional": [0.3, 0.7, 0.9], "social": [0.9, 0.8, 0.6], "causal": [0.7, 0.5, 0.8], "evolutionary": [0.6, 0.4, 0.7]}',
                'importance_score': 0.95,
                'decay_rate': 0.1,
                'access_count': 0
            },
            {
                'id': 'mem_002',
                'content': 'Bob raised concerns about the current database performance under high load conditions.',
                'memory_type': 'issue', 
                'content_type': 'technical',
                'meeting_id': 'meeting_002',
                'speaker': 'Bob Smith',
                'timestamp_ms': 1800000,  # 30 minutes into meeting
                'level': 2,  # L2 episodic
                'qdrant_id': 'qdrant_mem_002',
                'dimensions_json': '{"temporal": [0.6, 0.7, 0.5, 0.3], "emotional": [0.7, 0.6, 0.5], "social": [0.7, 0.6, 0.5], "causal": [0.8, 0.7, 0.6], "evolutionary": [0.5, 0.6, 0.4]}',
                'importance_score': 0.88,
                'decay_rate': 0.1,
                'access_count': 0
            },
            {
                'id': 'mem_003',
                'content': 'Action item: Charlie to research serverless architecture options and present findings next week.',
                'memory_type': 'action',
                'content_type': 'technical', 
                'meeting_id': 'meeting_002',
                'speaker': 'Charlie Davis',
                'timestamp_ms': 3600000,  # 60 minutes into meeting
                'level': 2,  # L2 episodic
                'qdrant_id': 'qdrant_mem_003',
                'dimensions_json': '{"temporal": [0.9, 0.8, 0.7, 0.9], "emotional": [0.4, 0.6, 0.8], "social": [0.8, 0.7, 0.7], "causal": [0.6, 0.8, 0.7], "evolutionary": [0.7, 0.5, 0.6]}',
                'importance_score': 0.92,
                'decay_rate': 0.1,
                'access_count': 0
            }
        ]
        
        for memory in sample_memories:
            cursor.execute("""
                INSERT INTO memories (
                    id, meeting_id, content, speaker, timestamp_ms,
                    memory_type, content_type, level, qdrant_id, dimensions_json,
                    importance_score, decay_rate, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory['id'], memory['meeting_id'], memory['content'],
                memory['speaker'], memory['timestamp_ms'], memory['memory_type'],
                memory['content_type'], memory['level'], memory['qdrant_id'],
                memory['dimensions_json'], memory['importance_score'],
                memory['decay_rate'], memory['access_count']
            ))
        
        print(f"  ✓ Inserted {len(sample_memories)} sample memories")
        
        # TODO: Sample connections
        sample_connections = [
            {
                'source_id': 'mem_001',
                'target_id': 'mem_003', 
                'connection_type': 'causal',
                'connection_strength': 0.75
            }
        ]
        
        for conn in sample_connections:
            cursor.execute("""
                INSERT INTO memory_connections (source_id, target_id, connection_type, connection_strength)
                VALUES (?, ?, ?, ?)
            """, (
                conn['source_id'], conn['target_id'], conn['connection_type'],
                conn['connection_strength']
            ))
        
        print(f"  ✓ Inserted {len(sample_connections)} sample connections")
        
        conn.commit()
        
    finally:
        conn.close()


async def verify_database_setup(db_path: str):
    """
    @TODO: Verify database setup and schema correctness.
    
    AGENTIC EMPOWERMENT: Database verification ensures the
    system is properly configured and ready for operation.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # TODO: Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['memories', 'memory_connections', 'meetings', 'search_history', 'system_metadata']
        missing_tables = set(expected_tables) - set(tables)
        
        if missing_tables:
            raise Exception(f"Missing tables: {missing_tables}")
        
        print(f"  ✓ All required tables present: {', '.join(expected_tables)}")
        
        # TODO: Check indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        required_indexes = ['idx_memories_meeting_id', 'idx_memories_created_at']
        missing_indexes = set(required_indexes) - set(indexes)
        
        if missing_indexes:
            print(f"  ⚠️  Missing some indexes: {missing_indexes}")
        else:
            print(f"  ✓ Performance indexes created")
        
        # TODO: Test basic queries
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM meetings") 
        meeting_count = cursor.fetchone()[0]
        
        print(f"  ✓ Database contains {memory_count} memories and {meeting_count} meetings")
        
        # TODO: Test foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        
        if fk_violations:
            print(f"  ⚠️  Foreign key violations: {len(fk_violations)}")
        else:
            print(f"  ✓ Foreign key constraints verified")
        
    finally:
        conn.close()


if __name__ == '__main__':
    # TODO: Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # TODO: Run initialization
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

