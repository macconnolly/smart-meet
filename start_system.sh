#!/bin/bash
# Quick start script for Cognitive Meeting Intelligence System

echo "🚀 Starting Cognitive Meeting Intelligence System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 5

# Initialize databases
echo "Initializing databases..."
python scripts/init_db.py
python scripts/init_qdrant.py

# Start the API server
echo "Starting API server..."
echo "🌐 API will be available at: http://localhost:8000/docs"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000