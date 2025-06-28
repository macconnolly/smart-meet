#!/bin/bash
# Simple script to start the API with minimal setup

echo "🚀 Starting Cognitive Meeting Intelligence API..."
echo "================================================"

# Check if Qdrant is running
if ! docker ps | grep -q qdrant; then
    echo "📦 Starting Qdrant vector database..."
    docker run -d --name qdrant -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
    sleep 5
else
    echo "✅ Qdrant already running"
fi

# Run API in Docker
echo "🐳 Starting API in Docker (this may take a minute)..."
docker run -it --rm \
  --name cognitive-api \
  -p 8000:8000 \
  -v $(pwd):/app \
  -w /app \
  --network host \
  python:3.11-slim \
  bash -c "
    echo '📦 Installing dependencies...'
    pip install --quiet --no-cache-dir \
      fastapi==0.104.1 \
      uvicorn==0.24.0 \
      pydantic==2.6.4 \
      pydantic-settings==2.1.0 \
      aiosqlite==0.19.0 \
      numpy==1.26.4 \
      onnxruntime==1.17.3 \
      qdrant-client==1.8.2 \
      vaderSentiment==3.3.2 \
      python-multipart \
      httpx
    
    echo '✅ Dependencies installed'
    echo '🚀 Starting API server...'
    echo ''
    echo '📌 API will be available at:'
    echo '   - http://localhost:8000'
    echo '   - http://localhost:8000/docs (Interactive docs)'
    echo ''
    echo 'Press Ctrl+C to stop'
    echo ''
    
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  "