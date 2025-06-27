#!/usr/bin/env python3
"""
Setup git hooks for the Cognitive Meeting Intelligence project.

This script installs pre-commit hooks to ensure:
- Proper commit message format with task IDs
- Code formatting with Black
- Linting with flake8
- Type checking with mypy
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_pre_commit():
    """Install and configure pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")

    # Check if pre-commit is installed
    try:
        subprocess.run(["pre-commit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing pre-commit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)

    # Create .pre-commit-config.yaml if it doesn't exist
    config_path = Path(".pre-commit-config.yaml")
    if not config_path.exists():
        print("Creating .pre-commit-config.yaml...")
        config_content = """# Pre-commit hooks for code quality and git workflow
repos:
  # Black - Python code formatter
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.12
        args: [--line-length=100]
        
  # Flake8 - Python linter
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203]
        
  # isort - Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
        
  # Check for large files
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-json
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-merge-conflict
      
  # Custom hook for commit message validation
  - repo: local
    hooks:
      - id: check-commit-msg
        name: Check commit message format
        entry: python scripts/check_commit_msg.py
        language: python
        stages: [commit-msg]
        pass_filenames: false
        always_run: true
"""
        config_path.write_text(config_content)

    # Install pre-commit hooks
    print("Installing git hooks...")
    subprocess.run(["pre-commit", "install"], check=True)
    subprocess.run(["pre-commit", "install", "--hook-type", "commit-msg"], check=True)

    print("✅ Pre-commit hooks installed successfully!")


def create_commit_msg_checker():
    """Create script to validate commit messages."""
    checker_path = Path("scripts/check_commit_msg.py")

    print("Creating commit message checker...")

    checker_content = '''#!/usr/bin/env python3
"""Check commit message follows project conventions."""

import sys
import re


def check_commit_message(message):
    """Validate commit message format."""
    lines = message.strip().split('\\n')
    
    if not lines:
        print("❌ Empty commit message")
        return False
    
    # Check first line format: <type>: <description> [<task-id>]
    first_line = lines[0]
    
    # Valid commit types
    valid_types = ['feat', 'fix', 'refactor', 'test', 'docs', 'chore', 'perf']
    
    # Pattern for first line
    pattern = r'^(' + '|'.join(valid_types) + r'):\\s+.+\\s+\\[IMPL-D\\d+-\\d{3}\\]$'
    
    if not re.match(pattern, first_line):
        print(f"❌ Invalid commit message format: {first_line}")
        print(f"✅ Expected: <type>: <description> [IMPL-D<day>-<number>]")
        print(f"✅ Example: feat: Add ONNX encoder class [IMPL-D2-001]")
        print(f"✅ Valid types: {', '.join(valid_types)}")
        return False
    
    # Check for Refs in body
    body = '\\n'.join(lines[1:])
    if 'Refs: #' not in body:
        print("❌ Missing 'Refs: #' in commit body")
        print("✅ Add: Refs: #D2-001 (matching your task ID)")
        return False
    
    return True


def main():
    """Main entry point."""
    # Read commit message from file (provided by git)
    commit_msg_file = sys.argv[1] if len(sys.argv) > 1 else '.git/COMMIT_EDITMSG'
    
    try:
        with open(commit_msg_file, 'r') as f:
            message = f.read()
    except FileNotFoundError:
        print(f"❌ Could not read commit message from {commit_msg_file}")
        sys.exit(1)
    
    # Check message
    if not check_commit_message(message):
        print("\\n💡 Tip: Use task branches like: git checkout -b impl/D2-001-description")
        sys.exit(1)
    
    print("✅ Commit message format is valid!")


if __name__ == "__main__":
    main()
'''

    checker_path.write_text(checker_content)
    checker_path.chmod(0o755)  # Make executable

    print("✅ Commit message checker created!")


def main():
    """Main setup function."""
    print("🚀 Setting up Git workflow enforcement...\n")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Create commit message checker
    create_commit_msg_checker()

    # Setup pre-commit
    setup_pre_commit()

    print("\n✨ Git workflow enforcement setup complete!")
    print("\nNext steps:")
    print("1. Run: pre-commit run --all-files  # Test on existing files")
    print("2. Make commits with proper format: feat: Description [IMPL-D2-001]")
    print("3. Include 'Refs: #D2-001' in commit body")
    print("\nHooks will run automatically on each commit!")


if __name__ == "__main__":
    main()
