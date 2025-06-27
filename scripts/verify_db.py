#!/usr/bin/env python3
"""Verify database contents."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.storage.sqlite.connection import DatabaseConnection


async def main():
    db = DatabaseConnection("data/memories.db")

    print("📊 Database Verification Report")
    print("=" * 60)

    # Check memories with details
    memories = await db.execute_query(
        """
        SELECT m.id, m.content, m.content_type, m.priority, m.status, 
               m.qdrant_id, p.name as project_name
        FROM memories m 
        JOIN projects p ON m.project_id = p.id
    """
    )

    print(f"\n📝 Memories ({len(memories)} total):")
    for mem in memories:
        print(f"  • ID: {mem['id']} | Type: {mem['content_type']} | Priority: {mem['priority']}")
        print(f"    Content: {mem['content'][:80]}...")
        print(f"    Status: {mem['status']} | Qdrant ID: {mem['qdrant_id']}")
        print()

    # Check connections
    connections = await db.execute_query(
        """
        SELECT mc.*, m1.content as source_content, m2.content as target_content
        FROM memory_connections mc
        JOIN memories m1 ON mc.source_id = m1.id
        JOIN memories m2 ON mc.target_id = m2.id
    """
    )

    print(f"\n🔗 Memory Connections ({len(connections)} total):")
    for conn in connections:
        print(f"  • {conn['source_id']} → {conn['target_id']}")
        print(f"    Type: {conn['connection_type']} | Strength: {conn['connection_strength']}")
        print(f"    Source: {conn['source_content'][:50]}...")
        print(f"    Target: {conn['target_content'][:50]}...")
        print()

    # Check meetings
    meetings = await db.execute_query(
        """
        SELECT id, title, meeting_type, meeting_category, 
               (strftime('%s', end_time) - strftime('%s', start_time))/60 as duration_minutes
        FROM meetings
    """
    )

    print(f"\n📅 Meetings ({len(meetings)} total):")
    for meet in meetings:
        print(f"  • {meet['id']}: {meet['title']}")
        print(f"    Type: {meet['meeting_type']} | Category: {meet['meeting_category']}")
        print(f"    Duration: {meet['duration_minutes']} minutes")
        print()

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
