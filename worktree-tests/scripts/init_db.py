"""
Database initialization script for the Cognitive Meeting Intelligence system.

This script creates the database schema and initializes the system with
proper tables, indexes, and initial metadata.

Usage:
    python scripts/init_db.py [--db-path PATH]
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite.connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_database(db_path: str = "data/cognitive.db") -> None:
    """
    Initialize the database with schema and initial data.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        logger.info(f"Initializing database at: {db_path}")
        
        # Create database connection
        db_connection = DatabaseConnection(db_path=db_path)
        
        # Execute schema
        schema_path = "src/storage/sqlite/schema.sql"
        logger.info(f"Executing schema from: {schema_path}")
        await db_connection.execute_schema(schema_path)
        
        # Verify tables were created
        tables_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """
        
        tables = await db_connection.execute_query(tables_query)
        table_names = [t['name'] for t in tables]
        
        expected_tables = [
            'deliverables',
            'meeting_series',
            'meetings',
            'memories',
            'memory_connections',
            'projects',
            'search_history',
            'stakeholders',
            'system_metadata'
        ]
        
        # Check all expected tables exist
        missing_tables = set(expected_tables) - set(table_names)
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            raise RuntimeError(f"Failed to create all tables. Missing: {missing_tables}")
        
        logger.info(f"Successfully created {len(table_names)} tables")
        for table in table_names:
            if table != 'sqlite_sequence':  # Skip SQLite internal table
                logger.info(f"  ✓ {table}")
        
        # Verify indexes were created
        indexes_query = """
            SELECT name FROM sqlite_master 
            WHERE type='index' 
            AND sql IS NOT NULL
            ORDER BY name
        """
        
        indexes = await db_connection.execute_query(indexes_query)
        logger.info(f"Created {len(indexes)} indexes")
        
        # Verify system metadata
        metadata_query = "SELECT key, value FROM system_metadata ORDER BY key"
        metadata = await db_connection.execute_query(metadata_query)
        
        logger.info("System metadata:")
        for row in metadata:
            logger.info(f"  {row['key']}: {row['value']}")
        
        # Close connection
        await db_connection.close()
        
        logger.info("✅ Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def verify_database(db_path: str = "data/cognitive.db") -> None:
    """
    Verify the database structure and integrity.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        logger.info("Verifying database integrity...")
        
        db_connection = DatabaseConnection(db_path=db_path)
        
        # Run integrity check
        integrity_check = await db_connection.execute_query("PRAGMA integrity_check")
        if integrity_check[0]['integrity_check'] != 'ok':
            raise RuntimeError(f"Database integrity check failed: {integrity_check}")
        
        logger.info("✓ Database integrity check passed")
        
        # Check foreign keys are enabled
        foreign_keys = await db_connection.execute_query("PRAGMA foreign_keys")
        if foreign_keys[0]['foreign_keys'] != 1:
            logger.warning("Foreign keys are not enabled!")
        else:
            logger.info("✓ Foreign keys enabled")
        
        # Check WAL mode
        journal_mode = await db_connection.execute_query("PRAGMA journal_mode")
        if journal_mode[0]['journal_mode'].lower() == 'wal':
            logger.info("✓ WAL mode enabled")
        else:
            logger.warning(f"Journal mode is {journal_mode[0]['journal_mode']}, not WAL")
        
        await db_connection.close()
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize the Cognitive Meeting Intelligence database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/cognitive.db",
        help="Path to the SQLite database file (default: data/cognitive.db)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing database, don't initialize"
    )
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run async initialization
    if args.verify_only:
        asyncio.run(verify_database(args.db_path))
    else:
        asyncio.run(init_database(args.db_path))


if __name__ == "__main__":
    main()
