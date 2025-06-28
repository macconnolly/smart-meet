#!/usr/bin/env python3
"""
Simple API runner that starts the API with minimal dependencies
"""
import subprocess
import sys
import time

print("🚀 Starting Cognitive Meeting Intelligence API")
print("=" * 60)

# Check if Qdrant is running
print("\n📦 Checking Qdrant status...")
result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
if "qdrant" in result.stdout:
    print("✅ Qdrant is running")
else:
    print("🔄 Starting Qdrant...")
    subprocess.run([
        "docker", "run", "-d", "--name", "qdrant", 
        "-p", "6333:6333", 
        "-v", f"{subprocess.os.getcwd()}/data/qdrant:/qdrant/storage",
        "qdrant/qdrant"
    ])
    time.sleep(5)

# Run the API
print("\n🐳 Starting API in Docker...")
print("This will take a minute to install dependencies...\n")

cmd = [
    "docker", "run", "--rm",
    "-p", "8000:8000",
    "-v", f"{subprocess.os.getcwd()}:/app",
    "-w", "/app",
    "--name", "cognitive-api-simple",
    "python:3.11-slim",
    "bash", "-c",
    """
    echo '📦 Installing core dependencies...'
    pip install --quiet \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        pydantic==2.6.4 \
        aiosqlite==0.19.0 \
        numpy==1.26.4 \
        qdrant-client==1.8.2 || exit 1
    
    echo '📦 Installing ML dependencies...'
    pip install --quiet \
        onnxruntime==1.17.3 \
        vaderSentiment==3.3.2 || exit 1
    
    echo '✅ Dependencies installed!'
    echo ''
    echo '🚀 Starting API server at http://localhost:8000'
    echo '📚 API docs at http://localhost:8000/docs'
    echo ''
    echo 'Press Ctrl+C to stop'
    echo ''
    
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info
    """
]

try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\n🛑 Stopping API...")
    subprocess.run(["docker", "stop", "cognitive-api-simple"], capture_output=True)
    print("✅ API stopped")