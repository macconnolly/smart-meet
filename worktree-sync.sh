#!/bin/bash
# Worktree Coordination Script for Cognitive Meeting Intelligence Project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the main project directory
MAIN_DIR="/mnt/c/Users/EL436GA/dev/meet"

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show current status of all worktrees
status() {
    print_color "$BLUE" "\n📊 Worktree Status:"
    echo "===================="
    
    git worktree list | while read -r line; do
        dir=$(echo "$line" | awk '{print $1}')
        branch=$(echo "$line" | grep -o '\[.*\]' | tr -d '[]')
        
        cd "$dir"
        commit=$(git rev-parse --short HEAD)
        
        # Check for uncommitted changes
        if [[ -n $(git status --porcelain) ]]; then
            status="${RED}⚠ Uncommitted changes${NC}"
        else
            status="${GREEN}✓ Clean${NC}"
        fi
        
        # Get last commit message
        last_commit=$(git log -1 --pretty=format:"%s" 2>/dev/null || echo "No commits")
        
        echo -e "\n📁 ${YELLOW}$(basename "$dir")${NC}"
        echo -e "   Branch: ${BLUE}$branch${NC}"
        echo -e "   Commit: $commit"
        echo -e "   Status: $status"
        echo -e "   Last: $last_commit"
    done
    
    cd "$MAIN_DIR"
}

# Function to sync a specific worktree with main
sync_with_main() {
    local worktree=$1
    
    if [[ -z "$worktree" ]]; then
        print_color "$RED" "Usage: $0 sync <worktree-name>"
        echo "Available worktrees:"
        echo "  - worktree-tests (all tests)"
        echo "  - worktree-day1 (Day 1: models, database)"
        echo "  - worktree-day2 (Day 2: embeddings)"
        echo "  - worktree-day3 (Day 3: dimensions, vector mgmt)"
        return 1
    fi
    
    if [[ ! -d "$worktree" ]]; then
        print_color "$RED" "Error: Worktree '$worktree' not found"
        return 1
    fi
    
    cd "$worktree"
    print_color "$YELLOW" "\n🔄 Syncing $worktree with main..."
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        print_color "$RED" "Error: Uncommitted changes in $worktree"
        git status --short
        return 1
    fi
    
    # Merge latest from main
    git merge main --no-edit
    print_color "$GREEN" "✓ $worktree synced with main"
    
    cd "$MAIN_DIR"
}

# Function to share changes from one worktree to another
share() {
    local from_tree=$1
    local to_tree=$2
    
    if [[ -z "$from_tree" || -z "$to_tree" ]]; then
        print_color "$RED" "Usage: $0 share <from-worktree> <to-worktree>"
        return 1
    fi
    
    if [[ ! -d "$from_tree" ]] || [[ ! -d "$to_tree" ]]; then
        print_color "$RED" "Error: Both worktrees must exist"
        return 1
    fi
    
    # Get the branch name from the source worktree
    cd "$from_tree"
    from_branch=$(git branch --show-current)
    
    print_color "$YELLOW" "\n🔄 Sharing changes from $from_tree to $to_tree..."
    
    # Go to destination worktree and fetch changes
    cd "$MAIN_DIR/$to_tree"
    git fetch "../$from_tree" "$from_branch:refs/remotes/local/$from_branch"
    
    print_color "$GREEN" "✓ Fetched changes from $from_tree"
    print_color "$BLUE" "To merge these changes, run:"
    echo "  cd $to_tree"
    echo "  git merge local/$from_branch"
    
    cd "$MAIN_DIR"
}

# Function to create a new task branch in a worktree
new_task() {
    local worktree=$1
    local task_type=$2
    local day=$3
    local task_num=$4
    local description=$5
    
    if [[ -z "$worktree" || -z "$task_type" || -z "$day" || -z "$task_num" || -z "$description" ]]; then
        print_color "$RED" "Usage: $0 task <worktree> <type> <day> <task-num> <description>"
        echo "Example: $0 task worktree-tests test 1 001 model-tests"
        echo "Types: test, impl, fix, feat"
        return 1
    fi
    
    if [[ ! -d "$worktree" ]]; then
        print_color "$RED" "Error: Worktree '$worktree' not found"
        return 1
    fi
    
    cd "$worktree"
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        print_color "$RED" "Error: Uncommitted changes in $worktree"
        git status --short
        return 1
    fi
    
    # Create branch name
    branch_name="${task_type}/D${day}-${task_num}-${description}"
    
    print_color "$YELLOW" "\n🌿 Creating branch: $branch_name"
    git checkout -b "$branch_name"
    
    print_color "$GREEN" "✓ Created and switched to $branch_name"
    
    cd "$MAIN_DIR"
}

# Function to run tests in a worktree
test() {
    local worktree=$1
    local test_path=$2
    
    if [[ -z "$worktree" ]]; then
        worktree="."
    fi
    
    cd "$worktree"
    
    print_color "$YELLOW" "\n🧪 Running tests in $worktree..."
    
    # Activate virtual environment if it exists
    if [[ -f "../venv/bin/activate" ]]; then
        source ../venv/bin/activate
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    # Run tests
    if [[ -n "$test_path" ]]; then
        pytest "$test_path" -v
    else
        pytest -v
    fi
    
    cd "$MAIN_DIR"
}

# Function to show what's different between worktrees
diff_trees() {
    local tree1=$1
    local tree2=$2
    
    if [[ -z "$tree1" || -z "$tree2" ]]; then
        print_color "$RED" "Usage: $0 diff <worktree1> <worktree2>"
        return 1
    fi
    
    cd "$tree1"
    branch1=$(git branch --show-current)
    commit1=$(git rev-parse HEAD)
    
    cd "$MAIN_DIR/$tree2"
    branch2=$(git branch --show-current)
    commit2=$(git rev-parse HEAD)
    
    print_color "$BLUE" "\n📊 Comparing $tree1 vs $tree2"
    echo "================================"
    
    # Show commits in tree1 not in tree2
    print_color "$YELLOW" "\nCommits in $tree1 not in $tree2:"
    git log --oneline "$commit2..$commit1" 2>/dev/null || echo "  None"
    
    # Show commits in tree2 not in tree1
    print_color "$YELLOW" "\nCommits in $tree2 not in $tree1:"
    git log --oneline "$commit1..$commit2" 2>/dev/null || echo "  None"
    
    cd "$MAIN_DIR"
}

# Function to show coordination workflow
workflow() {
    print_color "$BLUE" "\n🔄 TDD Workflow Guide"
    echo "===================="
    echo
    print_color "$YELLOW" "1️⃣  Test Worktree (ALL tests):"
    echo "   cd worktree-tests"
    echo "   ./worktree-sync.sh task worktree-tests test 1 001 model-tests"
    echo "   # Write failing tests for Day 1"
    echo "   git add tests/unit/test_models.py"
    echo "   git commit -m \"test: Add unit tests for Memory model [TEST-D1-001]\""
    echo
    print_color "$YELLOW" "2️⃣  Share to Day 1 Implementation:"
    echo "   ./worktree-sync.sh share worktree-tests worktree-day1"
    echo
    print_color "$YELLOW" "3️⃣  Day 1 Implementation:"
    echo "   cd worktree-day1"
    echo "   git merge local/test/D1-001-model-tests"
    echo "   # Fix code to pass tests"
    echo "   ./worktree-sync.sh test . tests/unit/test_models.py"
    echo "   git add src/models/entities.py"
    echo "   git commit -m \"feat: Implement Memory model [IMPL-D1-001]\""
    echo
    print_color "$YELLOW" "4️⃣  Day 2 Implementation:"
    echo "   cd worktree-day2"
    echo "   # Work on embeddings after Day 1 is complete"
    echo "   # ONNX encoder, vector manager, etc."
    echo
    print_color "$YELLOW" "5️⃣  Update Main:"
    echo "   cd $MAIN_DIR"
    echo "   git merge feature/test-implementation"
    echo "   git merge feature/day1-implementation"
    echo "   # Update TASK_COMPLETION_CHECKLIST.md"
    echo
}

# Function to check which tasks need tests
check_tests() {
    print_color "$BLUE" "\n🔍 Checking Test Coverage"
    echo "========================"
    
    # Check for existing test files
    print_color "$YELLOW" "\nExisting test files:"
    find tests -name "test_*.py" -type f | sort
    
    print_color "$YELLOW" "\n📋 Priority test files needed:"
    echo "1. tests/unit/test_models.py - for src/models/entities.py"
    echo "2. tests/unit/test_connection.py - for src/storage/sqlite/connection.py"
    echo "3. tests/unit/test_memory_repository.py - for src/storage/sqlite/repositories/memory_repository.py"
    echo "4. tests/unit/test_onnx_encoder.py - for src/embedding/onnx_encoder.py"
    echo "5. tests/unit/test_vector_manager.py - for src/embedding/vector_manager.py"
    echo "6. tests/unit/test_temporal_extractor.py - for src/extraction/dimensions/temporal_extractor.py"
    echo "7. tests/unit/test_emotional_extractor.py - for src/extraction/dimensions/emotional_extractor.py"
    echo "8. tests/integration/test_ingestion_pipeline.py - for src/pipeline/ingestion_pipeline.py"
}

# Main command handler
case "$1" in
    status)
        status
        ;;
    sync)
        sync_with_main "$2"
        ;;
    share)
        share "$2" "$3"
        ;;
    task)
        new_task "$2" "$3" "$4" "$5" "$6"
        ;;
    test)
        test "$2" "$3"
        ;;
    diff)
        diff_trees "$2" "$3"
        ;;
    workflow)
        workflow
        ;;
    check)
        check_tests
        ;;
    *)
        print_color "$BLUE" "🛠️  Worktree Coordination Script"
        echo "================================"
        echo
        echo "Usage: $0 <command> [arguments]"
        echo
        echo "Commands:"
        echo "  status              Show status of all worktrees"
        echo "  sync <worktree>     Sync a worktree with main"
        echo "  share <from> <to>   Share changes between worktrees"
        echo "  task <wt> <type> <day> <num> <desc>  Create task branch"
        echo "  test [worktree] [path]  Run tests in a worktree"
        echo "  diff <wt1> <wt2>    Show differences between worktrees"
        echo "  workflow            Show TDD workflow guide"
        echo "  check               Check test coverage status"
        echo
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 sync worktree-tests"
        echo "  $0 share worktree-tests worktree-day1-enhance"
        echo "  $0 task worktree-tests test 1 001 model-tests"
        echo "  $0 test worktree-tests tests/unit/test_models.py"
        ;;
esac