#!/bin/bash
# Setup virtual environments for all worktrees

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Main directory
MAIN_DIR="/mnt/c/Users/EL436GA/dev/meet"

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to setup venv for a worktree
setup_venv() {
    local worktree_dir=$1
    local worktree_name=$(basename "$worktree_dir")
    
    print_color "$BLUE" "\n🔧 Setting up venv for $worktree_name..."
    
    cd "$worktree_dir"
    
    # Check if venv already exists
    if [[ -d "venv" ]]; then
        print_color "$YELLOW" "⚠️  Virtual environment already exists in $worktree_name"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_color "$YELLOW" "Skipping $worktree_name"
            return
        fi
        rm -rf venv
    fi
    
    # Create virtual environment
    print_color "$GREEN" "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    print_color "$GREEN" "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements based on worktree
    case "$worktree_name" in
        worktree-tests)
            if [[ -f "requirements-tests.txt" ]]; then
                print_color "$GREEN" "Installing test requirements..."
                pip install -r requirements-tests.txt
            fi
            ;;
        worktree-day1)
            if [[ -f "requirements-day1.txt" ]]; then
                print_color "$GREEN" "Installing Day 1 requirements..."
                pip install -r requirements-day1.txt
            fi
            ;;
        worktree-day2)
            if [[ -f "requirements-day2.txt" ]]; then
                print_color "$GREEN" "Installing Day 2 requirements..."
                pip install -r requirements-day2.txt
            fi
            ;;
        worktree-day3)
            if [[ -f "requirements-day3.txt" ]]; then
                print_color "$GREEN" "Installing Day 3 requirements..."
                pip install -r requirements-day3.txt
            fi
            ;;
        *)
            # Main directory or unknown - use standard requirements
            if [[ -f "requirements.txt" ]]; then
                print_color "$GREEN" "Installing requirements..."
                pip install -r requirements.txt
            fi
            if [[ -f "requirements-dev.txt" ]]; then
                print_color "$GREEN" "Installing dev requirements..."
                pip install -r requirements-dev.txt
            fi
            ;;
    esac
    
    # Install project in editable mode
    if [[ -f "setup.py" ]] || [[ -f "pyproject.toml" ]]; then
        print_color "$GREEN" "Installing project in editable mode..."
        pip install -e .
    fi
    
    # Create activation shortcut
    cat > activate_venv.sh << 'EOF'
#!/bin/bash
# Quick activation script for this worktree
source venv/bin/activate
echo "✅ Virtual environment activated for $(basename $(pwd))"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
EOF
    chmod +x activate_venv.sh
    
    # Deactivate
    deactivate
    
    print_color "$GREEN" "✅ Virtual environment setup complete for $worktree_name"
    print_color "$BLUE" "   To activate: source venv/bin/activate"
    print_color "$BLUE" "   Or use: ./activate_venv.sh"
}

# Main execution
print_color "$BLUE" "🐍 Setting up virtual environments for all worktrees\n"

# Setup main directory venv first
if [[ ! -d "$MAIN_DIR/venv" ]]; then
    print_color "$YELLOW" "⚠️  Main directory doesn't have a venv. Setting it up first..."
    setup_venv "$MAIN_DIR"
fi

# Get list of worktrees
worktrees=$(git worktree list --porcelain | grep "worktree" | cut -d' ' -f2 | grep -v "^$MAIN_DIR$")

# Setup venv for each worktree
while IFS= read -r worktree; do
    if [[ -d "$worktree" ]]; then
        setup_venv "$worktree"
    fi
done <<< "$worktrees"

print_color "$GREEN" "\n✨ All virtual environments setup complete!"
print_color "$BLUE" "\n📝 Quick Reference:"
print_color "$BLUE" "   Main venv: cd $MAIN_DIR && source venv/bin/activate"

for worktree in $MAIN_DIR/worktree-*; do
    if [[ -d "$worktree" ]]; then
        name=$(basename "$worktree")
        print_color "$BLUE" "   $name: cd $worktree && source venv/bin/activate"
    fi
done

print_color "$YELLOW" "\n💡 Tip: Each worktree now has its own isolated Python environment!"
print_color "$YELLOW" "   Remember to activate the venv when working in each worktree."