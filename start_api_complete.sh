#!/bin/bash
# Complete API startup script with all dependencies

echo "🚀 Starting Cognitive Meeting Intelligence API"
echo "============================================"
echo ""

# The complete list of dependencies from requirements.txt
docker run --rm -p 8000:8000 \
  -v $(pwd):/app \
  -w /app \
  --add-host=host.docker.internal:host-gateway \
  --name cognitive-api \
  python:3.11-slim \
  bash -c "
    echo '📦 Installing all required dependencies...'
    pip install --quiet \
      fastapi==0.104.1 \
      uvicorn[standard]==0.24.0 \
      pydantic==2.6.4 \
      pydantic-settings==2.1.0 \
      aiohttp==3.9.1 \
      aiosqlite==0.19.0 \
      sqlalchemy==2.0.25 \
      qdrant-client==1.8.2 \
      numpy==1.26.4 \
      onnxruntime==1.17.3 \
      transformers==4.38.2 \
      torch==2.2.2 \
      vaderSentiment==3.3.2 \
      pytest==7.4.3 \
      pytest-asyncio==0.21.1 \
      pytest-cov==4.1.0 \
      black==23.11.0 \
      flake8==6.1.0 \
      mypy==1.7.1 \
      python-dotenv==1.0.0 \
      click==8.1.7 \
      rich==13.7.0 \
      loguru==0.7.2 \
      python-multipart \
      httpx
    
    echo '✅ All dependencies installed!'
    echo ''
    echo '🌐 API starting at: http://localhost:8000'
    echo '📚 API docs at: http://localhost:8000/docs'
    echo '🔍 Health check: http://localhost:8000/health'
    echo ''
    echo 'Press Ctrl+C to stop'
    echo ''
    
    QDRANT_HOST=host.docker.internal python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info
  "