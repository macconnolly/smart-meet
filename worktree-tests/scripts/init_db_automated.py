#!/usr/bin/env python3
"""
Automated database initialization script for Cognitive Meeting Intelligence.
Non-interactive version for automated setup.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.storage.sqlite.connection import DatabaseConnection  # noqa: E402
from src.models.entities import (  # noqa: E402
    Project,
    ProjectType,
    ProjectStatus,
    Meeting,
    MeetingType,
    MeetingCategory,
    Memory,
    MemoryType,
    ContentType,
    Priority,
    Status,
    Stakeholder,
    StakeholderType,
    InfluenceLevel,
    EngagementLevel,
    Deliverable,
    DeliverableType,
    DeliverableStatus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main(force_recreate: bool = False, insert_sample_data: bool = True):
    """Main database initialization function."""
    print("🔧 Initializing Cognitive Meeting Intelligence Database")
    print("   Enhanced for Strategy Consulting Context")
    print()

    try:
        # Database configuration
        db_path = "data/memories.db"
        db = DatabaseConnection(db_path)

        # Check if database exists
        db_file = Path(db_path)
        if db_file.exists() and force_recreate:
            # Backup existing database
            backup_path = db_file.with_suffix(
                f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
            )
            db_file.rename(backup_path)
            print(f"💾 Existing database backed up to: {backup_path}")

        # Initialize database schema
        print("🏗️  Creating database schema...")
        await db.execute_schema()

        # Verify schema creation
        print("✅ Verifying database setup...")
        await verify_database_setup(db)

        # Insert sample data if requested
        if insert_sample_data:
            print("\n📊 Inserting sample data...")
            await insert_sample_data_func(db)

        print("\n🎉 Database initialization completed successfully!")
        print(f"📍 Database location: {Path(db_path).absolute()}")
        
        # Test basic query
        print("\n🔍 Testing database queries...")
        test_results = await db.execute_query("SELECT COUNT(*) as count FROM memories")
        print(f"   ✓ Memories table contains {test_results[0]['count']} records")
        
        test_results = await db.execute_query("SELECT COUNT(*) as count FROM meetings")
        print(f"   ✓ Meetings table contains {test_results[0]['count']} records")
        
        test_results = await db.execute_query("SELECT COUNT(*) as count FROM projects")
        print(f"   ✓ Projects table contains {test_results[0]['count']} records")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return 1
    finally:
        await db.close()

    return 0


async def verify_database_setup(db: DatabaseConnection):
    """Verify database setup and schema correctness."""
    # Check tables exist
    tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    tables = await db.execute_query(tables_query)
    table_names = [t["name"] for t in tables]

    expected_tables = [
        "deliverables",
        "meeting_series",
        "meetings",
        "memories",
        "memory_connections",
        "projects",
        "search_history",
        "stakeholders",
        "system_metadata",
    ]

    missing_tables = set(expected_tables) - set(table_names)
    if missing_tables:
        raise Exception(f"Missing tables: {missing_tables}")

    print(f"  ✓ All {len(expected_tables)} required tables present")

    # Check indexes
    indexes_query = "SELECT name FROM sqlite_master WHERE type='index'"
    indexes = await db.execute_query(indexes_query)
    index_names = [idx["name"] for idx in indexes]

    print(f"  ✓ Created {len(index_names)} performance indexes")

    # Verify system metadata
    metadata = await db.execute_query("SELECT * FROM system_metadata")
    for row in metadata:
        print(f"  ✓ {row['key']}: {row['value']}")


async def insert_sample_data_func(db: DatabaseConnection):
    """Insert sample consulting project data."""
    # Sample project
    project = Project(
        id="proj_001",
        name="Digital Transformation Strategy",
        client_name="Acme Corporation",
        project_type=ProjectType.TRANSFORMATION,
        status=ProjectStatus.ACTIVE,
        start_date=datetime.now() - timedelta(days=30),
        project_manager="Sarah Chen",
        engagement_code="ACME-2024-DT-001",
        budget_hours=800,
        consumed_hours=120,
    )

    await db.execute_update(
        """INSERT INTO projects (id, name, client_name, project_type, status,
           start_date, project_manager, engagement_code, budget_hours, consumed_hours,
           metadata_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            project.id,
            project.name,
            project.client_name,
            project.project_type.value,
            project.status.value,
            project.start_date.isoformat(),
            project.project_manager,
            project.engagement_code,
            project.budget_hours,
            project.consumed_hours,
            json.dumps(project.metadata),
            project.created_at.isoformat(),
            project.updated_at.isoformat(),
        ),
    )
    print(f"  ✓ Created project: {project.name}")

    # Sample stakeholders
    stakeholders = [
        Stakeholder(
            id="stake_001",
            project_id=project.id,
            name="John Martinez",
            organization="Acme Corporation",
            role="CEO",
            stakeholder_type=StakeholderType.CLIENT_SPONSOR,
            influence_level=InfluenceLevel.HIGH,
            engagement_level=EngagementLevel.CHAMPION,
            email="john.martinez@acme.com",
        ),
        Stakeholder(
            id="stake_002",
            project_id=project.id,
            name="Lisa Wong",
            organization="Acme Corporation",
            role="CTO",
            stakeholder_type=StakeholderType.CLIENT_TEAM,
            influence_level=InfluenceLevel.HIGH,
            engagement_level=EngagementLevel.SUPPORTIVE,
            email="lisa.wong@acme.com",
        ),
        Stakeholder(
            id="stake_003",
            project_id=project.id,
            name="Michael Brown",
            organization="McKinsey & Company",
            role="Partner",
            stakeholder_type=StakeholderType.CONSULTANT_PARTNER,
            influence_level=InfluenceLevel.HIGH,
            engagement_level=EngagementLevel.CHAMPION,
        ),
    ]

    for stakeholder in stakeholders:
        await db.execute_update(
            """INSERT INTO stakeholders (id, project_id, name, organization, role,
               stakeholder_type, influence_level, engagement_level, email, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stakeholder.id,
                stakeholder.project_id,
                stakeholder.name,
                stakeholder.organization,
                stakeholder.role,
                stakeholder.stakeholder_type.value,
                stakeholder.influence_level.value,
                stakeholder.engagement_level.value,
                stakeholder.email,
                stakeholder.notes,
                stakeholder.created_at.isoformat(),
            ),
        )
    print(f"  ✓ Created {len(stakeholders)} stakeholders")

    # Sample meetings
    meetings = [
        Meeting(
            id="meet_001",
            project_id=project.id,
            title="Kickoff Meeting with CEO",
            meeting_type=MeetingType.CLIENT_WORKSHOP,
            meeting_category=MeetingCategory.EXTERNAL,
            start_time=datetime.now() - timedelta(days=28),
            end_time=datetime.now() - timedelta(days=28, hours=-2),
            participants=[
                {"name": "John Martinez", "role": "client", "organization": "Acme"},
                {"name": "Sarah Chen", "role": "consultant", "organization": "McKinsey"},
                {"name": "Michael Brown", "role": "consultant", "organization": "McKinsey"},
            ],
            transcript_path="data/transcripts/meet_001.txt",
            key_decisions=[
                "Proceed with 3-phase transformation",
                "Focus on customer experience first",
            ],
            action_items=[
                {
                    "task": "Prepare current state assessment",
                    "owner": "Sarah Chen",
                    "due": "2024-02-01",
                },
                {"task": "Schedule stakeholder interviews", "owner": "Team", "due": "2024-01-25"},
            ],
        ),
        Meeting(
            id="meet_002",
            project_id=project.id,
            title="Technical Architecture Review",
            meeting_type=MeetingType.EXPERT_INTERVIEW,
            meeting_category=MeetingCategory.EXTERNAL,
            start_time=datetime.now() - timedelta(days=21),
            end_time=datetime.now() - timedelta(days=21, hours=-1.5),
            participants=[
                {"name": "Lisa Wong", "role": "client", "organization": "Acme"},
                {"name": "David Kim", "role": "consultant", "organization": "McKinsey"},
                {"name": "External Expert", "role": "expert", "organization": "Cloud Architect"},
            ],
            transcript_path="data/transcripts/meet_002.txt",
        ),
        Meeting(
            id="meet_003",
            project_id=project.id,
            title="Weekly SteerCo Update",
            meeting_type=MeetingType.CLIENT_STEERING,
            meeting_category=MeetingCategory.EXTERNAL,
            is_recurring=True,
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now() - timedelta(days=7, hours=-1),
            participants=[
                {"name": "John Martinez", "role": "client", "organization": "Acme"},
                {"name": "Lisa Wong", "role": "client", "organization": "Acme"},
                {"name": "Michael Brown", "role": "consultant", "organization": "McKinsey"},
                {"name": "Sarah Chen", "role": "consultant", "organization": "McKinsey"},
            ],
            transcript_path="data/transcripts/meet_003.txt",
        ),
    ]

    for meeting in meetings:
        await db.execute_update(
            """INSERT INTO meetings (id, project_id, title, meeting_type, meeting_category,
               is_recurring, recurring_series_id, start_time, end_time, participants_json,
               transcript_path, agenda_json, key_decisions_json, action_items_json,
               metadata_json, created_at, processed_at, memory_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meeting.id,
                meeting.project_id,
                meeting.title,
                meeting.meeting_type.value,
                meeting.meeting_category.value,
                meeting.is_recurring,
                meeting.recurring_series_id,
                meeting.start_time.isoformat(),
                meeting.end_time.isoformat() if meeting.end_time else None,
                json.dumps(meeting.participants),
                meeting.transcript_path,
                json.dumps(meeting.agenda),
                json.dumps(meeting.key_decisions),
                json.dumps(meeting.action_items),
                json.dumps(meeting.metadata),
                meeting.created_at.isoformat(),
                meeting.processed_at.isoformat() if meeting.processed_at else None,
                meeting.memory_count,
            ),
        )
    print(f"  ✓ Created {len(meetings)} meetings")

    # Sample deliverable - Create before memories that reference it
    deliverable = Deliverable(
        id="deliv_001",
        project_id=project.id,
        name="Customer Experience Transformation Roadmap",
        deliverable_type=DeliverableType.ROADMAP,
        status=DeliverableStatus.IN_PROGRESS,
        due_date=datetime.now() + timedelta(days=14),
        owner="Sarah Chen",
        reviewer="Michael Brown",
        description="Comprehensive roadmap for Phase 1 of digital transformation",
        acceptance_criteria={
            "timeline": "3-6 month implementation plan",
            "budget": "Cost estimates for each initiative",
            "metrics": "Success KPIs defined",
        },
    )

    await db.execute_update(
        """INSERT INTO deliverables (id, project_id, name, deliverable_type, status,
           due_date, owner, reviewer, version, file_path, description,
           acceptance_criteria_json, dependencies_json, created_at, delivered_at, approved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            deliverable.id,
            deliverable.project_id,
            deliverable.name,
            deliverable.deliverable_type.value,
            deliverable.status.value,
            deliverable.due_date.isoformat() if deliverable.due_date else None,
            deliverable.owner,
            deliverable.reviewer,
            deliverable.version,
            deliverable.file_path,
            deliverable.description,
            json.dumps(deliverable.acceptance_criteria),
            json.dumps(deliverable.dependencies),
            deliverable.created_at.isoformat(),
            deliverable.delivered_at.isoformat() if deliverable.delivered_at else None,
            deliverable.approved_at.isoformat() if deliverable.approved_at else None,
        ),
    )
    print(f"  ✓ Created deliverable: {deliverable.name}")

    # Sample memories
    memories = [
        Memory(
            id="mem_001",
            meeting_id="meet_001",
            project_id=project.id,
            content=(
                "We've decided to implement the digital transformation in three phases: "
                "customer experience, operational efficiency, and innovation platform"
            ),
            speaker="John Martinez",
            speaker_role="client",
            timestamp_ms=900000,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DECISION,
            priority=Priority.HIGH,
            status=Status.COMPLETED,
            level=2,
            qdrant_id="qdrant_mem_001",
            dimensions_json=json.dumps(
                {
                    "temporal": [0.9, 0.8, 0.3, 0.7],
                    "emotional": [0.7, 0.8, 0.9],
                    "social": [0.9, 0.8, 0.7],
                    "causal": [0.8, 0.7, 0.6],
                    "strategic": [0.95, 0.9, 0.85],
                }
            ),
            importance_score=0.95,
        ),
        Memory(
            id="mem_002",
            meeting_id="meet_002",
            project_id=project.id,
            content=(
                "The current monolithic architecture is a major bottleneck for scaling. "
                "We need to consider microservices migration"
            ),
            speaker="Lisa Wong",
            speaker_role="client",
            timestamp_ms=1800000,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.ISSUE,
            priority=Priority.HIGH,
            status=Status.OPEN,
            owner="David Kim",
            level=2,
            qdrant_id="qdrant_mem_002",
            dimensions_json=json.dumps(
                {
                    "temporal": [0.7, 0.6, 0.8, 0.5],
                    "emotional": [0.6, 0.5, 0.4],
                    "social": [0.7, 0.6, 0.6],
                    "causal": [0.9, 0.8, 0.7],
                    "strategic": [0.8, 0.75, 0.7],
                }
            ),
            importance_score=0.88,
        ),
        Memory(
            id="mem_003",
            meeting_id="meet_001",
            project_id=project.id,
            content=(
                "Create a detailed roadmap for the customer experience "
                "transformation phase by end of month"
            ),
            speaker="Sarah Chen",
            speaker_role="consultant",
            timestamp_ms=3600000,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.ACTION,
            priority=Priority.HIGH,
            status=Status.IN_PROGRESS,
            owner="Sarah Chen",
            due_date=datetime.now() + timedelta(days=14),
            level=2,
            qdrant_id="qdrant_mem_003",
            dimensions_json=json.dumps(
                {
                    "temporal": [0.95, 0.9, 0.7, 0.8],
                    "emotional": [0.5, 0.6, 0.7],
                    "social": [0.8, 0.7, 0.75],
                    "causal": [0.7, 0.8, 0.6],
                    "strategic": [0.85, 0.8, 0.9],
                }
            ),
            importance_score=0.92,
            deliverable_id="deliv_001",  # Will be created below
        ),
    ]

    for memory in memories:
        await db.execute_update(
            """INSERT INTO memories (id, meeting_id, project_id, content, speaker,
               speaker_role, timestamp, memory_type, content_type, priority, status,
               due_date, owner, level, qdrant_id, dimensions_json, importance_score,
               decay_rate, access_count, last_accessed, created_at, parent_id, deliverable_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.meeting_id,
                memory.project_id,
                memory.content,
                memory.speaker,
                memory.speaker_role,
                memory.timestamp_ms / 1000.0,
                memory.memory_type.value,
                memory.content_type.value,
                memory.priority.value if memory.priority else None,
                memory.status.value if memory.status else None,
                memory.due_date.isoformat() if memory.due_date else None,
                memory.owner,
                memory.level,
                memory.qdrant_id,
                memory.dimensions_json,
                memory.importance_score,
                memory.decay_rate,
                memory.access_count,
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.created_at.isoformat(),
                memory.parent_id,
                memory.deliverable_id,
            ),
        )
    print(f"  ✓ Created {len(memories)} memories")

    # Sample memory connections
    await db.execute_update(
        """INSERT INTO memory_connections (source_id, target_id, connection_strength,
           connection_type, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("mem_001", "mem_003", 0.85, "supports", datetime.now().isoformat()),
    )

    await db.execute_update(
        """INSERT INTO memory_connections (source_id, target_id, connection_strength,
           connection_type, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("mem_002", "mem_001", 0.7, "blocks", datetime.now().isoformat()),
    )
    print("  ✓ Created memory connections")

    print("\n📊 Sample consulting project data inserted successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Cognitive Meeting Intelligence Database")
    parser.add_argument("--force", action="store_true", help="Force recreate database")
    parser.add_argument("--no-sample-data", action="store_true", help="Skip sample data insertion")
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(main(
        force_recreate=args.force,
        insert_sample_data=not args.no_sample_data
    ))
    sys.exit(exit_code)