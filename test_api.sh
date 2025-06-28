#!/bin/bash
# Test script to verify API is working

echo "🧪 Testing Cognitive Meeting Intelligence API"
echo "==========================================="
echo ""

# Wait for API to be ready
echo "⏳ Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Test 1: Health Check
echo ""
echo "1️⃣ Health Check:"
curl -s http://localhost:8000/health | python3 -m json.tool

# Test 2: Ingest Sample Meeting
echo ""
echo "2️⃣ Ingesting Sample Meeting:"
curl -s -X POST http://localhost:8000/api/v2/memories/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test-project-001",
    "title": "API Performance Meeting",
    "transcript": "Sarah: We have a critical issue with our API response times. They have degraded by 40% this week.\n\nJohn: That is concerning. We should implement caching as a temporary fix.\n\nSarah: Good idea. Let us make that our top priority for tomorrow.\n\nMike: I have a question - how long can we sustain the current performance?\n\nSarah: Based on our metrics, we have about 2 weeks before it becomes critical.",
    "participants": ["Sarah", "John", "Mike"]
  }' | python3 -m json.tool

# Test 3: Search Memories
echo ""
echo "3️⃣ Searching Memories:"
curl -s -X POST http://localhost:8000/api/v2/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test-project-001",
    "query": "performance issues",
    "limit": 5
  }' | python3 -m json.tool

echo ""
echo "✅ Tests complete! Check http://localhost:8000/docs for interactive API documentation."