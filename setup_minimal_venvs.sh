#!/bin/bash
# Minimal virtual environment setup for worktrees

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_color() {
    echo -e "${1}${2}${NC}"
}

# Quick setup for each worktree
setup_minimal() {
    local dir=$1
    local req_file=$2
    local name=$(basename "$dir")
    
    print_color "$BLUE" "\n🔧 Setting up $name..."
    
    cd "$dir"
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_color "$GREEN" "✓ Created venv"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
    
    if [[ -f "$req_file" ]]; then
        print_color "$GREEN" "Installing from $req_file..."
        pip install -r "$req_file"
    fi
    
    deactivate
    print_color "$GREEN" "✓ $name ready!"
}

# Main execution
print_color "$BLUE" "🐍 Minimal Virtual Environment Setup\n"

# Setup each worktree with its specific requirements
setup_minimal "worktree-tests" "requirements-tests.txt"
setup_minimal "worktree-day1" "requirements-day1.txt"
setup_minimal "worktree-day2" "requirements-day2.txt"
setup_minimal "worktree-day3" "requirements-day3.txt"

print_color "$GREEN" "\n✨ All minimal environments ready!"
print_color "$YELLOW" "\n💡 Note: Each worktree has only the packages it needs:"
print_color "$BLUE" "   - worktree-tests: Testing tools only"
print_color "$BLUE" "   - worktree-day1: Models & database"
print_color "$BLUE" "   - worktree-day2: Embeddings & ONNX"
print_color "$BLUE" "   - worktree-day3: Dimensions & analysis"
print_color "$YELLOW" "\n⚡ To activate: cd worktree-XXX && source venv/bin/activate"