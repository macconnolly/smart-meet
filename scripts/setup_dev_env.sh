#!/bin/bash
# Development Environment Setup Script for Cognitive Meeting Intelligence System

set -e  # Exit on error

echo "🚀 Setting up Cognitive Meeting Intelligence Development Environment"
echo "=================================================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ ! "$PYTHON_VERSION" == *"$REQUIRED_VERSION"* ]]; then
    echo "❌ Error: Python $REQUIRED_VERSION is required, but found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version check passed: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Configure pip for corporate proxy (if certificate exists)
if [ -f "combined-ca-bundle.crt" ]; then
    echo "🔒 Configuring SSL certificates for corporate proxy..."
    pip config set global.cert $(pwd)/combined-ca-bundle.crt
fi

# Install dependencies
echo "📦 Installing production dependencies..."
pip install -r requirements.txt

echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    
    # Start Qdrant
    echo "🐋 Starting Qdrant vector database..."
    docker-compose up -d
    
    # Wait for Qdrant to be ready
    echo "⏳ Waiting for Qdrant to be ready..."
    sleep 5
    
    # Check Qdrant health
    if curl -s http://localhost:6333/readiness > /dev/null; then
        echo "✅ Qdrant is running"
    else
        echo "⚠️  Warning: Qdrant may not be ready yet"
    fi
else
    echo "⚠️  Warning: Docker not found. Please install Docker to run Qdrant."
fi

# Initialize databases
echo "🗄️  Initializing SQLite database..."
python scripts/init_db.py

if command -v docker &> /dev/null; then
    echo "🗄️  Initializing Qdrant collections..."
    python scripts/init_qdrant.py
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data logs

# Run initial checks
echo "🔍 Running code quality checks..."
echo "  - Black (formatting)..."
black --check src/ tests/ || true

echo "  - Flake8 (linting)..."
flake8 src/ tests/ || true

echo "  - MyPy (type checking)..."
mypy src/ || true

# Display success message
echo ""
echo "✨ Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Activate the virtual environment (if not already active):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the API server:"
echo "     make run"
echo "     # or"
echo "     uvicorn src.api.cognitive_api:app --reload"
echo ""
echo "  3. Access the API documentation:"
echo "     http://localhost:8000/docs"
echo ""
echo "  4. Run tests:"
echo "     make test"
echo ""
echo "  5. Before committing code:"
echo "     make check"
echo ""
echo "📚 For more information, see:"
echo "  - docs/DEVELOPMENT_SETUP.md"
echo "  - docs/QUICK_REFERENCE.md"
echo ""
echo "Happy coding! 🎉"