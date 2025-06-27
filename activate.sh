#!/bin/bash
# Enhanced activation script for Cognitive Meeting Intelligence

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Cognitive Meeting Intelligence Development Environment${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.cfg" ] || [ ! -d "src" ]; then
    echo -e "${YELLOW}Warning: Not in project root directory!${NC}"
    echo "Please run this from the project root: /mnt/c/Users/EL436GA/dev/meet"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Virtual environment not found! Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
fi

# Source aliases
if [ -f ".venv_aliases" ]; then
    source .venv_aliases
    echo -e "${GREEN}✓ Project aliases loaded${NC}"
fi

# Check critical services
echo ""
echo -e "${BLUE}Checking services...${NC}"

# Check if Qdrant is running
if curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is running${NC}"
else
    echo -e "${YELLOW}⚠ Qdrant is not running. Run: docker-compose up -d qdrant${NC}"
fi

# Check if database exists
if [ -f "data/cognitive.db" ]; then
    echo -e "${GREEN}✓ Database exists${NC}"
else
    echo -e "${YELLOW}⚠ Database not found. Run: setup-db${NC}"
fi

# Check if model exists
if [ -f "models/all-MiniLM-L6-v2/model.onnx" ]; then
    echo -e "${GREEN}✓ ONNX model found${NC}"
else
    echo -e "${YELLOW}⚠ ONNX model not found. Run: setup-model${NC}"
fi

# Display useful info
echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo "  setup-all    - Run all setup scripts"
echo "  run-api      - Start development API server"
echo "  test         - Run tests"
echo "  quality      - Run code quality checks"
echo "  db           - Open SQLite database"
echo ""
echo -e "${BLUE}Current Status:${NC}"
echo -e "  Python: $(python --version)"
echo -e "  Directory: $(pwd)"
echo -e "  Git branch: $(git branch --show-current)"
echo ""

# Set useful environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROJECT_ROOT="$(pwd)"

echo -e "${GREEN}Ready for development! 🚀${NC}"