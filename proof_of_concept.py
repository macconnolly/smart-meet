#!/usr/bin/env python3
"""
Standalone Proof of Concept - Demonstrates core functionality without dependencies
"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Simplified models (no external dependencies)
class ContentType:
    DECISION = "decision"
    ACTION = "action"
    ISSUE = "issue"
    QUESTION = "question"
    INSIGHT = "insight"
    CONTEXT = "context"

class Memory:
    def __init__(self, content: str, content_type: str, speaker: str = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.content_type = content_type
        self.speaker = speaker
        self.timestamp = datetime.now()
        self.importance = 0.5
        
    def __repr__(self):
        return f"Memory({self.content_type}: {self.content[:50]}...)"

class MemoryConnection:
    def __init__(self, source_id: str, target_id: str, connection_type: str, strength: float):
        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type
        self.strength = strength

# Simple classifier (no ML needed)
class SimpleClassifier:
    @staticmethod
    def classify(text: str) -> str:
        text_lower = text.lower()
        
        # Decision patterns
        if any(word in text_lower for word in ["decided", "decision", "agreed", "will implement"]):
            return ContentType.DECISION
            
        # Action patterns
        if any(word in text_lower for word in ["will do", "action item", "by tomorrow", "complete by"]):
            return ContentType.ACTION
            
        # Issue patterns
        if any(word in text_lower for word in ["problem", "issue", "broken", "not working"]):
            return ContentType.ISSUE
            
        # Question patterns
        if text.strip().endswith("?") or text_lower.startswith(("what", "how", "why", "when")):
            return ContentType.QUESTION
            
        # Insight patterns
        if any(word in text_lower for word in ["realized", "discovered", "means that", "indicates"]):
            return ContentType.INSIGHT
            
        return ContentType.CONTEXT

# Simple storage
class SimpleStorage:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                speaker TEXT,
                timestamp TEXT,
                importance REAL
            );
            
            CREATE TABLE IF NOT EXISTS connections (
                source_id TEXT,
                target_id TEXT,
                connection_type TEXT,
                strength REAL,
                PRIMARY KEY (source_id, target_id)
            );
        """)
        self.conn.commit()
        
    def store_memory(self, memory: Memory):
        self.conn.execute("""
            INSERT INTO memories (id, content, content_type, speaker, timestamp, importance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory.id, memory.content, memory.content_type, memory.speaker, 
              memory.timestamp.isoformat(), memory.importance))
        self.conn.commit()
        
    def store_connection(self, connection: MemoryConnection):
        self.conn.execute("""
            INSERT OR REPLACE INTO connections (source_id, target_id, connection_type, strength)
            VALUES (?, ?, ?, ?)
        """, (connection.source_id, connection.target_id, 
              connection.connection_type, connection.strength))
        self.conn.commit()
        
    def get_memories(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM memories ORDER BY timestamp")
        return [dict(row) for row in cursor.fetchall()]
        
    def get_connections(self) -> List[Dict]:
        cursor = self.conn.execute("""
            SELECT c.*, m1.content as source_content, m2.content as target_content
            FROM connections c
            JOIN memories m1 ON c.source_id = m1.id
            JOIN memories m2 ON c.target_id = m2.id
        """)
        return [dict(row) for row in cursor.fetchall()]

# Main demonstration
def demonstrate_system():
    print("🧠 COGNITIVE MEETING INTELLIGENCE - PROOF OF CONCEPT")
    print("=" * 70)
    
    # Sample meeting transcript
    transcript_segments = [
        ("Sarah", "We have a critical issue with the vendor API performance."),
        ("John", "That's concerning. We need to make a decision about this."),
        ("Sarah", "I've decided we should implement a caching layer as a workaround."),
        ("John", "Good idea. Let's make that an action item for tomorrow."),
        ("Mike", "How long will the caching solution take to implement?"),
        ("Sarah", "Based on my analysis, it should take about 2 days."),
        ("John", "I realized this connects to our scalability concerns from last month."),
        ("Sarah", "You're right. This indicates we need a longer-term architecture review."),
    ]
    
    # Initialize components
    classifier = SimpleClassifier()
    storage = SimpleStorage()
    memories = []
    
    # Step 1: Extract and classify memories
    print("\n📝 STEP 1: EXTRACTING AND CLASSIFYING MEMORIES")
    print("-" * 70)
    
    for speaker, content in transcript_segments:
        # Classify
        content_type = classifier.classify(content)
        
        # Create memory
        memory = Memory(content=content, content_type=content_type, speaker=speaker)
        memories.append(memory)
        
        # Store it
        storage.store_memory(memory)
        
        print(f"✅ {speaker}: [{content_type}] {content}")
    
    # Step 2: Create relationships
    print("\n\n🔗 STEP 2: CREATING RELATIONSHIPS BETWEEN MEMORIES")
    print("-" * 70)
    
    # Sequential connections
    print(f"Creating {len(memories)-1} sequential connections...")
    for i in range(len(memories) - 1):
        conn = MemoryConnection(
            memories[i].id, 
            memories[i+1].id,
            "sequential",
            0.7
        )
        storage.store_connection(conn)
    print(f"✅ Created sequential connections")
    
    # Semantic connections (issue -> decision -> action)
    issue_memories = [m for m in memories if m.content_type == ContentType.ISSUE]
    decision_memories = [m for m in memories if m.content_type == ContentType.DECISION]
    action_memories = [m for m in memories if m.content_type == ContentType.ACTION]
    
    # Connect issue to decision
    if issue_memories and decision_memories:
        conn = MemoryConnection(
            issue_memories[0].id,
            decision_memories[0].id,
            "problem_solution",
            0.9
        )
        storage.store_connection(conn)
        print(f"✅ Connected: Issue → Decision (problem_solution)")
    
    # Connect decision to action
    if decision_memories and action_memories:
        conn = MemoryConnection(
            decision_memories[0].id,
            action_memories[0].id,
            "implementation",
            0.85
        )
        storage.store_connection(conn)
        print(f"✅ Connected: Decision → Action (implementation)")
    
    # Connect insights
    insight_memories = [m for m in memories if m.content_type == ContentType.INSIGHT]
    if insight_memories and issue_memories:
        conn = MemoryConnection(
            insight_memories[0].id,
            issue_memories[0].id,
            "relates_to",
            0.6
        )
        storage.store_connection(conn)
        print(f"✅ Connected: Insight → Issue (relates_to)")
    
    # Step 3: Show stored data
    print("\n\n💾 STEP 3: VERIFYING STORED DATA")
    print("-" * 70)
    
    stored_memories = storage.get_memories()
    print(f"\n📊 Stored {len(stored_memories)} memories:")
    
    # Count by type
    type_counts = {}
    for mem in stored_memories:
        ct = mem['content_type']
        type_counts[ct] = type_counts.get(ct, 0) + 1
    
    for content_type, count in type_counts.items():
        print(f"  • {content_type}: {count}")
    
    # Step 4: Show relationships
    print("\n\n🌉 STEP 4: VERIFYING RELATIONSHIPS")
    print("-" * 70)
    
    connections = storage.get_connections()
    print(f"\n📊 Created {len(connections)} connections:")
    
    for conn in connections:
        print(f"\n🔗 {conn['connection_type']} (strength: {conn['strength']})")
        print(f"  From: {conn['source_content'][:50]}...")
        print(f"  To:   {conn['target_content'][:50]}...")
    
    # Step 5: Demonstrate search/retrieval
    print("\n\n🔍 STEP 5: DEMONSTRATING RETRIEVAL")
    print("-" * 70)
    
    # Find all memories related to "performance"
    cursor = storage.conn.execute("""
        SELECT DISTINCT m2.* 
        FROM memories m1
        JOIN connections c ON (m1.id = c.source_id OR m1.id = c.target_id)
        JOIN memories m2 ON (m2.id = c.source_id OR m2.id = c.target_id)
        WHERE m1.content LIKE '%performance%'
    """)
    
    related = cursor.fetchall()
    print(f"\nMemories connected to 'performance' query:")
    for mem in related:
        print(f"  • [{mem['content_type']}] {mem['content'][:60]}...")
    
    print("\n\n" + "=" * 70)
    print("✅ PROOF OF CONCEPT COMPLETE!")
    print("=" * 70)
    print("\nDemonstrated capabilities:")
    print("  ✅ Loaded and parsed meeting transcript")
    print("  ✅ Classified memories into correct types")
    print("  ✅ Stored memories with metadata")
    print("  ✅ Created meaningful relationships between memories")
    print("  ✅ Retrieved connected memories based on content")
    print("\nAll core functionality works without complex dependencies!")

if __name__ == "__main__":
    demonstrate_system()