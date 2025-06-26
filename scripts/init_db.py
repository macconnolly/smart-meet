#\!/usr/bin/env python3
"""
Initialize SQLite database for Cognitive Meeting Intelligence.

Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
This script creates the database schema and initializes system metadata.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.sqlite.connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_database():
    """
    Initialize the SQLite database with schema.
    
    TODO Day 1:
    - [ ] Execute schema.sql file
    - [ ] Verify all tables created
    - [ ] Insert initial system metadata
    - [ ] Add error handling and rollback
    """
    db = get_db()
    
    try:
        logger.info("Initializing database...")
        
        # TODO Day 1: Read and execute schema.sql
        schema_path = Path(__file__).parent.parent / "src/storage/sqlite/schema.sql"
        
        # TODO Day 1: Execute schema
        await db.execute_schema(str(schema_path))
        
        # TODO Day 1: Insert initial metadata
        # await db.execute_update(
        #     "INSERT INTO system_metadata (key, value) VALUES (?, ?)",
        #     ("schema_version", "1.0")
        # )
        
        # TODO Day 1: Verify tables exist
        # tables = await db.execute_query(
        #     "SELECT name FROM sqlite_master WHERE type='table'"
        # )
        
        logger.info("Database initialized successfully\!")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        await db.close()


def main():
    """Main entry point."""
    # TODO Day 1: Add command line arguments for custom db path
    asyncio.run(init_database())


if __name__ == "__main__":
    main()
EOF < /dev/null
