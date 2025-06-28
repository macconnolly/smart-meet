#!/usr/bin/env python3
"""Fix import paths in the project to use src. prefix."""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single Python file."""
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track if we made changes
    original_content = content
    
    # Patterns to fix
    import_patterns = [
        # from X import Y patterns
        (r'from (models|storage|cognitive|embedding|extraction|pipeline|utils|core)(\.[a-zA-Z0-9_.]+)? import', 
         r'from src.\1\2 import'),
        # import X patterns
        (r'^import (models|storage|cognitive|embedding|extraction|pipeline|utils|core)(\.[a-zA-Z0-9_.]+)?',
         r'import src.\1\2'),
    ]
    
    for pattern, replacement in import_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
        return True
    return False

def main():
    """Fix all Python files in src directory."""
    src_dir = Path('src')
    fixed_count = 0
    
    for py_file in src_dir.rglob('*.py'):
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files.")

if __name__ == "__main__":
    main()