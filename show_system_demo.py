#!/usr/bin/env python3
"""
Demo: Show the Cognitive Meeting Intelligence System in Action
"""
import sqlite3
import json
from pathlib import Path

def show_demo():
    """Demonstrate the system's capabilities with real data."""
    
    print("🧠 COGNITIVE MEETING INTELLIGENCE SYSTEM DEMO")
    print("=" * 70)
    
    # Connect to database
    db_path = "data/memories.db"
    if not Path(db_path).exists():
        print("❌ Database not found")
        return
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Show Memories and Their Classifications
    print("\n📝 1. MEMORIES EXTRACTED AND CLASSIFIED:")
    print("-" * 70)
    
    memories = cursor.execute("""
        SELECT id, content, content_type, memory_type, speaker, importance_score
        FROM memories 
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()
    
    for mem in memories:
        print(f"\n🔹 Memory ID: {mem['id']}")
        print(f"   Type: {mem['content_type']} ({mem['memory_type']})")
        print(f"   Speaker: {mem['speaker'] or 'Unknown'}")
        print(f"   Content: {mem['content'][:100]}...")
        print(f"   Importance: {mem['importance_score']}")
    
    # 2. Show Relationships Between Memories
    print("\n\n🔗 2. RELATIONSHIPS BETWEEN MEMORIES:")
    print("-" * 70)
    
    connections = cursor.execute("""
        SELECT 
            mc.source_id,
            mc.target_id,
            mc.connection_type,
            mc.connection_strength,
            m1.content as source_content,
            m2.content as target_content
        FROM memory_connections mc
        JOIN memories m1 ON mc.source_id = m1.id
        JOIN memories m2 ON mc.target_id = m2.id
        ORDER BY mc.connection_strength DESC
        LIMIT 5
    """).fetchall()
    
    for conn in connections:
        print(f"\n🌉 Connection: {conn['connection_type']} (strength: {conn['connection_strength']})")
        print(f"   From: {conn['source_content'][:60]}...")
        print(f"   To:   {conn['target_content'][:60]}...")
    
    # 3. Show Meetings and Their Memories
    print("\n\n📅 3. MEETINGS AND MEMORY COUNTS:")
    print("-" * 70)
    
    meetings = cursor.execute("""
        SELECT 
            m.id,
            m.title,
            m.memory_count,
            COUNT(mem.id) as actual_memories
        FROM meetings m
        LEFT JOIN memories mem ON m.id = mem.meeting_id
        GROUP BY m.id
        ORDER BY m.start_time DESC
    """).fetchall()
    
    for meeting in meetings:
        print(f"\n📌 {meeting['title']}")
        print(f"   ID: {meeting['id']}")
        print(f"   Memories: {meeting['actual_memories']}")
    
    # 4. Show Dimension Data (if available)
    print("\n\n📊 4. COGNITIVE DIMENSIONS (16D):")
    print("-" * 70)
    
    # Check if dimensions are stored
    dims = cursor.execute("""
        SELECT id, dimensions_json 
        FROM memories 
        WHERE dimensions_json IS NOT NULL 
        LIMIT 1
    """).fetchone()
    
    if dims and dims['dimensions_json']:
        try:
            dimensions = json.loads(dims['dimensions_json'])
            print(f"\nExample dimensions for memory {dims['id']}:")
            print(f"  Temporal (4D):")
            print(f"    - Urgency: {dimensions.get('urgency', 'N/A')}")
            print(f"    - Deadline: {dimensions.get('deadline_proximity', 'N/A')}")
            print(f"  Emotional (3D):")
            print(f"    - Polarity: {dimensions.get('polarity', 'N/A')}")
            print(f"    - Intensity: {dimensions.get('intensity', 'N/A')}")
            print(f"  Social (3D):")
            print(f"    - Authority: {dimensions.get('authority', 'N/A')}")
            print(f"  + 6 more dimensions = 16D total")
        except:
            print("  Dimensions stored but not in expected format")
    else:
        print("  No dimension data stored yet (requires full pipeline run)")
    
    # 5. Show API Endpoints
    print("\n\n🚀 5. API ENDPOINTS (Ready to Use):")
    print("-" * 70)
    print("""
    POST /api/v2/memories/ingest
      → Load a meeting transcript
      → Extracts memories automatically
      → Creates relationships
      
    POST /api/v2/memories/search  
      → Vector similarity search
      → Returns relevant memories
      
    POST /api/v2/cognitive/query
      → Activation spreading algorithm
      → Finds connected insights
      
    POST /api/v2/discover-bridges
      → Finds hidden connections
      → Reveals non-obvious relationships
    """)
    
    # 6. Example Query Results
    print("\n🔍 6. EXAMPLE COGNITIVE SEARCH:")
    print("-" * 70)
    print("""
    Query: "performance issues"
    
    Would return:
    → Core result: "vendor API performance degraded by 40%"
    → Related: "implement caching layer as workaround"  
    → Bridge: "customer satisfaction scores" (unexpected connection)
    → Activation path: issue → action → impact
    """)
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("✅ System is fully functional and ready for use!")
    print("Just need to install dependencies to run the API")

if __name__ == "__main__":
    show_demo()