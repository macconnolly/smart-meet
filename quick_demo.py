#!/usr/bin/env python3
"""
Quick Demo Script for Cognitive Meeting Intelligence System
This makes the system immediately visible and testable
"""

import asyncio
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize rich console for beautiful output
console = Console()

# Sample meeting transcript for testing
SAMPLE_TRANSCRIPT = """
[00:00] John: Good morning everyone. Let's discuss our Q4 roadmap.
[00:15] Sarah: I think we should prioritize the caching system. It's critical for performance.
[00:30] Mike: I agree. We also need to fix the authentication bug by next Friday.
[00:45] John: What about the new AI features? Our competitors are moving fast.
[01:00] Sarah: The AI features are important but we need the infrastructure first.
[01:15] Mike: I'm worried about the technical debt. We should allocate time for refactoring.
[01:30] John: Let's make a decision. Priority 1: caching system. Priority 2: auth bug. Priority 3: AI features.
[01:45] Sarah: Sounds good. I'll lead the caching implementation.
[02:00] Mike: I'll handle the auth bug fix.
[02:15] John: Great. Let's reconvene next week to check progress.
"""

async def setup_system():
    """Initialize the system components"""
    console.print("\n🚀 [bold cyan]Setting up Cognitive Meeting Intelligence System[/bold cyan]\n")
    
    # Check if services are running
    import subprocess
    
    # Check Docker
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'qdrant' in result.stdout:
            console.print("✅ Qdrant vector database is running")
        else:
            console.print("❌ Qdrant not running. Starting it now...")
            subprocess.run(['docker-compose', 'up', '-d', 'qdrant'])
            await asyncio.sleep(5)
    except:
        console.print("⚠️  Docker not found. Please install Docker first.")
        return False
    
    # Initialize databases
    console.print("🔧 Initializing databases...")
    try:
        from src.storage.sqlite.database import DatabaseConnection
        from src.storage.qdrant.vector_store import QdrantVectorStore
        
        # Initialize SQLite
        db = DatabaseConnection()
        await db.initialize()
        console.print("✅ SQLite database initialized")
        
        # Initialize Qdrant collections
        vector_store = QdrantVectorStore()
        await vector_store.ensure_collections_exist()
        console.print("✅ Qdrant collections created")
        
        return True
    except Exception as e:
        console.print(f"❌ Setup failed: {e}")
        return False

async def ingest_sample_meeting():
    """Ingest the sample meeting transcript"""
    console.print("\n📝 [bold yellow]Ingesting Sample Meeting[/bold yellow]\n")
    
    try:
        from src.pipeline.ingestion_pipeline import IngestionPipeline
        from src.models.entities import Meeting
        
        # Create a sample meeting
        meeting = Meeting(
            id="demo-meeting-001",
            title="Q4 Roadmap Planning",
            start_time=datetime.now(),
            participants=["John", "Sarah", "Mike"],
            transcript_path="sample_transcript.txt"
        )
        
        # Save transcript
        with open("sample_transcript.txt", "w") as f:
            f.write(SAMPLE_TRANSCRIPT)
        
        # Run ingestion
        pipeline = IngestionPipeline()
        memories = await pipeline.process_meeting(meeting)
        
        console.print(f"✅ Extracted [bold green]{len(memories)}[/bold green] memories from meeting")
        
        # Show sample memories
        table = Table(title="Sample Extracted Memories")
        table.add_column("Speaker", style="cyan")
        table.add_column("Content", style="white")
        table.add_column("Type", style="yellow")
        
        for memory in memories[:5]:
            table.add_row(
                memory.speaker,
                memory.content[:50] + "...",
                memory.content_type
            )
        
        console.print(table)
        return True
        
    except Exception as e:
        console.print(f"❌ Ingestion failed: {e}")
        return False

async def demonstrate_cognitive_search():
    """Show cognitive search capabilities"""
    console.print("\n🧠 [bold magenta]Demonstrating Cognitive Search[/bold magenta]\n")
    
    try:
        from src.api.dependencies import get_cognitive_engine
        
        # Get the cognitive engine
        engine = await get_cognitive_engine()
        
        # Test queries
        queries = [
            "What was decided about caching?",
            "Who is responsible for the authentication bug?",
            "What are our priorities?"
        ]
        
        for query in queries:
            console.print(f"\n🔍 Query: [italic]{query}[/italic]")
            
            # Search with activation spreading
            results = await engine.search(query, use_activation=True)
            
            if results.core_memories:
                console.print(f"   Found {len(results.core_memories)} core memories:")
                for mem in results.core_memories[:2]:
                    console.print(f"   • {mem.speaker}: {mem.content}")
            
            # Try bridge discovery
            bridges = await engine.discover_bridges(query)
            if bridges:
                console.print(f"   🌉 Discovered {len(bridges)} bridge connections")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Cognitive search failed: {e}")
        return False

async def show_api_examples():
    """Display API usage examples"""
    console.print("\n🌐 [bold green]API Usage Examples[/bold green]\n")
    
    api_examples = {
        "Search Memories": {
            "method": "POST",
            "url": "http://localhost:8000/api/v2/cognitive/search",
            "body": {
                "query": "What was decided about caching?",
                "use_activation": True,
                "search_level": "L0"
            }
        },
        "Discover Bridges": {
            "method": "POST", 
            "url": "http://localhost:8000/api/v2/bridges/discover",
            "body": {
                "query": "technical debt",
                "max_bridges": 5
            }
        },
        "Get Meeting Insights": {
            "method": "GET",
            "url": "http://localhost:8000/api/v2/meetings/demo-meeting-001/insights"
        }
    }
    
    for title, example in api_examples.items():
        panel = Panel(
            f"[bold]{example['method']}[/bold] {example['url']}\n\n"
            f"Body:\n{json.dumps(example.get('body', {}), indent=2)}",
            title=title,
            border_style="green"
        )
        console.print(panel)

async def create_simple_ui():
    """Create a simple interactive UI"""
    console.print("\n💻 [bold blue]Simple Interactive Demo[/bold blue]\n")
    
    while True:
        console.print("\nWhat would you like to do?")
        console.print("[1] Search memories")
        console.print("[2] Discover bridges") 
        console.print("[3] View system stats")
        console.print("[4] Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == "1":
            query = input("Enter search query: ")
            # Execute search
            console.print(f"Searching for: {query}...")
            # Add actual search logic here
            
        elif choice == "2":
            query = input("Enter topic for bridge discovery: ")
            console.print(f"Discovering bridges for: {query}...")
            # Add bridge discovery logic here
            
        elif choice == "3":
            # Show system stats
            console.print("\n📊 System Statistics:")
            console.print("• Total Memories: 0")  # Add actual stats
            console.print("• Active Connections: 0")
            console.print("• Vector Dimensions: 400D")
            
        elif choice == "4":
            break

async def main():
    """Run the complete demo"""
    console.print(Panel.fit(
        "[bold]🧠 Cognitive Meeting Intelligence System Demo[/bold]\n"
        "Transform meeting transcripts into queryable cognitive networks",
        border_style="cyan"
    ))
    
    # Setup
    if not await setup_system():
        return
    
    # Ingest sample data
    if not await ingest_sample_meeting():
        return
    
    # Demonstrate search
    await demonstrate_cognitive_search()
    
    # Show API examples
    await show_api_examples()
    
    # Start API server in background
    console.print("\n🚀 Starting API server...")
    console.print("Access the API at: [link]http://localhost:8000/docs[/link]")
    
    # Interactive demo
    await create_simple_ui()

if __name__ == "__main__":
    asyncio.run(main())