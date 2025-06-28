# 🚀 Quick Start - Make It Visible!

This guide helps you quickly see the Cognitive Meeting Intelligence System in action.

## 1. One-Command Start

```bash
# Make the script executable
chmod +x start_system.sh

# Start everything
./start_system.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install dependencies  
- ✅ Start Qdrant vector database
- ✅ Initialize databases
- ✅ Start API server

## 2. Run the Interactive Demo

```bash
python quick_demo.py
```

This shows:
- 📝 Sample meeting ingestion
- 🧠 Cognitive search in action
- 🌉 Bridge discovery
- 📊 System statistics

## 3. Access the API

Open in your browser:
- **Interactive API Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 4. Simple Test Queries

Try these in the API docs:

### Search for Decisions
```json
POST /api/v2/cognitive/search
{
  "query": "What was decided about caching?",
  "use_activation": true
}
```

### Discover Connections
```json
POST /api/v2/bridges/discover
{
  "query": "technical debt",
  "max_bridges": 5
}
```

## 5. Visual Dashboard (Optional)

For a simple web UI, run:

```bash
python -m http.server 8080 --directory ui/
```

Then open: http://localhost:8080

## 6. See What's Happening

### Check Logs
```bash
# API logs
tail -f logs/api.log

# See what's in the database
sqlite3 data/memories.db "SELECT * FROM memories LIMIT 5;"

# Check vector store
curl http://localhost:6333/collections
```

### Monitor Performance
```bash
# Watch memory extraction
python -m src.scripts.monitor_performance
```

## 7. Load Your Own Meeting

Create a file `my_meeting.txt`:
```
[00:00] Alice: Let's discuss the new feature requirements
[00:30] Bob: I think we need better error handling
[01:00] Alice: Agreed. Let's prioritize that for next sprint
```

Then ingest it:
```python
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.models.entities import Meeting

meeting = Meeting(
    id="my-meeting-001",
    title="Feature Planning",
    participants=["Alice", "Bob"],
    transcript_path="my_meeting.txt"
)

pipeline = IngestionPipeline()
memories = await pipeline.process_meeting(meeting)
print(f"Extracted {len(memories)} memories!")
```

## 8. Quick Troubleshooting

### If Docker isn't running:
```bash
# Start Docker Desktop (Windows/Mac)
# Or on Linux:
sudo systemctl start docker
```

### If ports are in use:
```bash
# Change ports in docker-compose.yml
# Qdrant: 6333 → 6334
# API: 8000 → 8001
```

### If you see import errors:
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## 🎯 What You Should See

1. **API Documentation** - Interactive Swagger UI at http://localhost:8000/docs
2. **Sample Memories** - Extracted from the demo meeting
3. **Search Results** - Cognitive search finding relevant memories
4. **Bridge Connections** - Discovering non-obvious relationships
5. **Real-time Logs** - Showing system activity

## 💡 Next Steps

1. Try different search queries
2. Load your own meeting transcripts
3. Explore the cognitive algorithms
4. Check memory decay over time
5. Test bridge discovery with various topics

---

**Need help?** The system logs everything to `logs/` directory. Check there first!