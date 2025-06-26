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

from src.core.config import ConfigManager
from src.storage.sqlite.memory_repository import initialize_database


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
            
            # Relationship queries
            "CREATE INDEX IF NOT EXISTS idx_relationships_memory1 ON relationships(memory_id_1)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_memory2 ON relationships(memory_id_2)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)",
            
            # Meeting queries
            "CREATE INDEX IF NOT EXISTS idx_meetings_date ON meetings(date)",
            "CREATE INDEX IF NOT EXISTS idx_meetings_type ON meetings(meeting_type)",
            
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
                'date': '2024-01-15 10:00:00',
                'participants': '["Alice Johnson", "Bob Smith", "Charlie Davis"]',
                'duration': 90,
                'meeting_type': 'strategy',
                'metadata_json': '{"location": "Conference Room A", "recorded": true}'
            },
            {
                'id': 'meeting_002', 
                'title': 'Technical Architecture Review',
                'date': '2024-01-16 14:00:00',
                'participants': '["David Wilson", "Eve Brown", "Frank Miller"]',
                'duration': 120,
                'meeting_type': 'technical',
                'metadata_json': '{"location": "Virtual", "recorded": true}'
            },
            {
                'id': 'meeting_003',
                'title': 'Weekly Team Sync',
                'date': '2024-01-17 09:00:00', 
                'participants': '["Alice Johnson", "Bob Smith", "Grace Lee"]',
                'duration': 30,
                'meeting_type': 'sync',
                'metadata_json': '{"location": "Virtual", "recorded": false}'
            }
        ]
        
        for meeting in sample_meetings:
            cursor.execute("""
                INSERT INTO meetings (id, title, date, participants, duration, meeting_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting['id'], meeting['title'], meeting['date'], 
                meeting['participants'], meeting['duration'], 
                meeting['meeting_type'], meeting['metadata_json']
            ))
        
        print(f"  ✓ Inserted {len(sample_meetings)} sample meetings")
        
        # TODO: Sample memories
        sample_memories = [
            {
                'id': 'mem_001',
                'content': 'We decided to prioritize the mobile app development for Q1, targeting iOS first with Android following in Q2.',
                'memory_type': 'EPISODIC',
                'content_type': 'DECISION',
                'meeting_id': 'meeting_001',
                'speaker_id': 'Alice Johnson',
                'confidence': 0.95,
                'created_at': '2024-01-15 10:15:00',
                'access_count': 0,
                'metadata_json': '{"importance": "high", "stakeholders": ["mobile_team", "product_team"]}'
            },
            {
                'id': 'mem_002',
                'content': 'Bob raised concerns about the current database performance under high load conditions.',
                'memory_type': 'EPISODIC', 
                'content_type': 'DISCUSSION',
                'meeting_id': 'meeting_002',
                'speaker_id': 'Bob Smith',
                'confidence': 0.88,
                'created_at': '2024-01-16 14:30:00',
                'access_count': 0,
                'metadata_json': '{"topic": "performance", "urgency": "medium"}'
            },
            {
                'id': 'mem_003',
                'content': 'Action item: Charlie to research serverless architecture options and present findings next week.',
                'memory_type': 'EPISODIC',
                'content_type': 'ACTION_ITEM', 
                'meeting_id': 'meeting_002',
                'speaker_id': 'Charlie Davis',
                'confidence': 0.92,
                'created_at': '2024-01-16 15:00:00',
                'access_count': 0,
                'metadata_json': '{"assignee": "Charlie Davis", "due_date": "2024-01-23", "status": "pending"}'
            }
        ]
        
        for memory in sample_memories:
            cursor.execute("""
                INSERT INTO memories (
                    id, content, memory_type, content_type, meeting_id, 
                    speaker_id, confidence, created_at, updated_at, 
                    access_count, last_accessed, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory['id'], memory['content'], memory['memory_type'],
                memory['content_type'], memory['meeting_id'], memory['speaker_id'],
                memory['confidence'], memory['created_at'], memory['created_at'],
                memory['access_count'], None, memory['metadata_json']
            ))
        
        print(f"  ✓ Inserted {len(sample_memories)} sample memories")
        
        # TODO: Sample relationships
        sample_relationships = [
            {
                'memory_id_1': 'mem_001',
                'memory_id_2': 'mem_003', 
                'relationship_type': 'RELATED_TO',
                'strength': 0.75,
                'created_at': '2024-01-16 15:30:00'
            }
        ]
        
        for rel in sample_relationships:
            cursor.execute("""
                INSERT INTO relationships (memory_id_1, memory_id_2, relationship_type, strength, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                rel['memory_id_1'], rel['memory_id_2'], rel['relationship_type'],
                rel['strength'], rel['created_at']
            ))
        
        print(f"  ✓ Inserted {len(sample_relationships)} sample relationships")
        
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
        
        expected_tables = ['memories', 'relationships', 'meetings']
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

