#!/bin/bash
# WSL Setup Script for Cognitive Meeting Intelligence
# This script provides a smooth setup experience for WSL users

set -e  # Exit on error

echo "======================================"
echo "Cognitive Meeting Intelligence"
echo "WSL Setup Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
        
        # Check if it's 3.11+
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            print_success "Python version is 3.11+"
        else
            print_error "Python version is less than 3.11"
            echo "Please install Python 3.11+:"
            echo "  sudo apt update"
            echo "  sudo apt install python3.11 python3.11-venv python3.11-dev"
            return 1
        fi
    else
        print_error "Python3 not found"
        return 1
    fi
}

# Setup with system packages (WSL optimized)
setup_system() {
    print_info "Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install Python and development packages
    sudo apt install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        build-essential \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        sqlite3 \
        libsqlite3-dev
    
    print_success "System packages installed"
}

# Setup Python virtual environment
setup_venv() {
    print_info "Setting up Python virtual environment..."
    
    # Remove old venv if exists
    if [ -d "venv" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new venv
    python3.11 -m venv venv
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
    
    # Install requirements
    print_info "Installing Python requirements..."
    pip install -r requirements.txt
    
    # Download NLTK data
    print_info "Downloading NLTK data..."
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
    
    print_success "Virtual environment setup complete"
}

# Setup with Docker in WSL
setup_docker() {
    print_info "Setting up Docker in WSL..."
    
    # Check if Docker is installed
    if command -v docker &> /dev/null; then
        print_success "Docker found"
    else
        print_error "Docker not found"
        echo ""
        echo "To install Docker in WSL2:"
        echo "1. Install Docker Desktop for Windows"
        echo "2. Enable WSL2 integration in Docker Desktop settings"
        echo "3. Restart WSL"
        echo ""
        echo "Or install Docker directly in WSL:"
        echo "  curl -fsSL https://get.docker.com | sudo sh"
        echo "  sudo usermod -aG docker $USER"
        echo "  newgrp docker"
        return 1
    fi
    
    # Start services with Docker Compose
    print_info "Starting Docker containers..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 5
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker services are running"
        echo ""
        echo "Services available at:"
        echo "  - API: http://localhost:8000"
        echo "  - API Docs: http://localhost:8000/docs"
        echo "  - Qdrant: http://localhost:6333"
    else
        print_error "Docker services failed to start"
        return 1
    fi
}

# Setup Poetry (WSL optimized)
setup_poetry() {
    print_info "Setting up Poetry..."
    
    # Install Poetry
    if command -v poetry &> /dev/null; then
        print_success "Poetry already installed"
    else
        print_info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Install dependencies with Poetry
    print_info "Installing dependencies with Poetry..."
    poetry install
    
    print_success "Poetry setup complete"
    echo ""
    echo "To use Poetry:"
    echo "  poetry shell  # Activate environment"
    echo "  poetry run pytest  # Run tests"
}

# Initialize databases
init_databases() {
    print_info "Initializing databases..."
    
    # Create data directories
    mkdir -p data models
    
    # Initialize SQLite
    if [ -f "scripts/init_db.py" ]; then
        python scripts/init_db.py
        print_success "SQLite database initialized"
    fi
    
    # Check if Qdrant is running
    if curl -s http://localhost:6333/health > /dev/null; then
        print_success "Qdrant is accessible"
        
        # Initialize Qdrant collections
        if [ -f "scripts/init_qdrant.py" ]; then
            python scripts/init_qdrant.py
            print_success "Qdrant collections initialized"
        fi
    else
        print_info "Qdrant not running. Start it with: docker-compose up -d"
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    
    # Check if we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "venv" ]; then
            source venv/bin/activate
        elif command -v poetry &> /dev/null; then
            poetry run pytest -v
            return
        fi
    fi
    
    # Run pytest
    if command -v pytest &> /dev/null; then
        pytest -v
    else
        print_error "pytest not found. Installing..."
        pip install pytest pytest-asyncio pytest-cov
        pytest -v
    fi
}

# Main menu
main_menu() {
    echo ""
    echo "Choose setup option:"
    echo "1. Full WSL setup (system packages + venv)"
    echo "2. Docker setup (recommended for isolation)"
    echo "3. Poetry setup (modern dependency management)"
    echo "4. Just create virtual environment"
    echo "5. Initialize databases"
    echo "6. Run tests"
    echo "7. Quick health check"
    echo "0. Exit"
    echo ""
    
    read -p "Enter your choice (0-7): " choice
    
    case $choice in
        1)
            check_python || setup_system
            setup_venv
            init_databases
            run_tests
            ;;
        2)
            setup_docker
            ;;
        3)
            check_python || setup_system
            setup_poetry
            ;;
        4)
            check_python && setup_venv
            ;;
        5)
            init_databases
            ;;
        6)
            run_tests
            ;;
        7)
            quick_health_check
            ;;
        0)
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            main_menu
            ;;
    esac
}

# Quick health check
quick_health_check() {
    print_info "Running health check..."
    
    # Check Python
    check_python
    
    # Check virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
    else
        print_info "No virtual environment active"
    fi
    
    # Check key Python packages
    python3 -c "
import sys
print(f'Python: {sys.version}')

packages = ['fastapi', 'qdrant_client', 'sqlalchemy', 'numpy', 'sentence_transformers']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
"
    
    # Check services
    echo ""
    print_info "Checking services..."
    
    # Check Qdrant
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        print_success "Qdrant is running"
    else
        print_info "Qdrant is not running"
    fi
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is running"
    else
        print_info "API is not running"
    fi
}

# Show WSL tips
show_wsl_tips() {
    echo ""
    echo "=== WSL Tips for This Project ==="
    echo ""
    echo "1. File Performance:"
    echo "   Store the project in WSL filesystem (~/dev/meet)"
    echo "   NOT in /mnt/c/ for better performance"
    echo ""
    echo "2. Docker Integration:"
    echo "   Use Docker Desktop with WSL2 backend enabled"
    echo "   Or install Docker directly in WSL"
    echo ""
    echo "3. VS Code Integration:"
    echo "   code .  # Opens VS Code with WSL remote"
    echo ""
    echo "4. Port Access:"
    echo "   Services are accessible from Windows at localhost:port"
    echo ""
    echo "5. Activate virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
}

# Run main menu
clear
main_menu
show_wsl_tips
