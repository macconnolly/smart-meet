#!/usr/bin/env python3
"""
Fix API Import Issues - Automated Script

This script automatically fixes the known import path issues in the API layer.
These are simple relative vs absolute import corrections.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single file."""
    print(f"🔧 Fixing imports in {file_path.name}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Common import fixes needed
    import_fixes = [
        # Models imports
        (r'from models\.', 'from src.models.'),
        (r'from \.\.models\.', 'from src.models.'),
        
        # Extraction imports
        (r'from extraction\.', 'from src.extraction.'),
        (r'from \.\.extraction\.', 'from src.extraction.'),
        
        # Storage imports
        (r'from storage\.', 'from src.storage.'),
        (r'from \.\.storage\.', 'from src.storage.'),
        
        # Cognitive imports
        (r'from cognitive\.', 'from src.cognitive.'),
        (r'from \.\.cognitive\.', 'from src.cognitive.'),
        
        # Core imports
        (r'from core\.', 'from src.core.'),
        (r'from \.\.core\.', 'from src.core.'),
        
        # API imports
        (r'from api\.', 'from src.api.'),
        (r'from \.api\.', 'from src.api.'),
        
        # Embedding imports
        (r'from embedding\.', 'from src.embedding.'),
        (r'from \.\.embedding\.', 'from src.embedding.'),
        
        # Pipeline imports
        (r'from pipeline\.', 'from src.pipeline.'),
        (r'from \.\.pipeline\.', 'from src.pipeline.'),
    ]
    
    for pattern, replacement in import_fixes:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            # Find what changed
            lines_changed = []
            for i, (old, new) in enumerate(zip(content.splitlines(), new_content.splitlines())):
                if old != new:
                    lines_changed.append((i+1, old.strip(), new.strip()))
            
            for line_num, old_line, new_line in lines_changed:
                changes.append(f"  Line {line_num}: {old_line} → {new_line}")
            
            content = new_content
    
    # Only write if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {len(changes)} import(s) in {file_path.name}:")
        for change in changes[:5]:  # Show first 5 changes
            print(change)
        if len(changes) > 5:
            print(f"  ... and {len(changes) - 5} more")
    else:
        print(f"✅ No changes needed in {file_path.name}")

def fix_all_api_imports():
    """Fix imports in all API files."""
    print("🚀 Fixing API Import Issues")
    print("=" * 60)
    
    # Find all Python files in the API directory
    api_dir = Path("src/api")
    if not api_dir.exists():
        print("❌ API directory not found at src/api")
        return False
    
    # Fix main API file
    main_file = api_dir / "main.py"
    if main_file.exists():
        fix_imports_in_file(main_file)
    
    # Fix dependencies
    deps_file = api_dir / "dependencies.py"
    if deps_file.exists():
        fix_imports_in_file(deps_file)
    
    # Fix all router files
    routers_dir = api_dir / "routers"
    if routers_dir.exists():
        for router_file in routers_dir.glob("*.py"):
            if router_file.name != "__init__.py":
                fix_imports_in_file(router_file)
    
    # Also check other directories that might have import issues
    other_dirs = ["src/cognitive", "src/extraction", "src/storage", "src/pipeline"]
    
    print("\n🔍 Checking other directories for import issues...")
    for dir_path in other_dirs:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            for py_file in dir_obj.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    # Check if file has problematic imports
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    if any(pattern in content for pattern in ['from models.', 'from extraction.', 'from storage.', 'from cognitive.', 'from core.', 'from pipeline.', 'from embedding.']):
                        fix_imports_in_file(py_file)
    
    print("\n✅ Import fixing complete!")
    return True

def verify_imports():
    """Try to import key modules to verify fixes worked."""
    print("\n🧪 Verifying imports...")
    
    test_imports = [
        "src.api.main",
        "src.models.entities",
        "src.extraction.memory_extractor",
        "src.cognitive.activation.basic_activation_engine",
        "src.storage.sqlite.connection",
    ]
    
    success_count = 0
    for module_name in test_imports:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: {str(e)[:50]}...")
    
    print(f"\n📊 Import verification: {success_count}/{len(test_imports)} successful")
    return success_count == len(test_imports)

def main():
    """Run the import fixer."""
    print("🔧 API Import Fixer for Cognitive Meeting Intelligence")
    print("This will automatically fix known import issues")
    print("-" * 60)
    
    # Create backup reminder
    print("\n⚠️  IMPORTANT: This will modify files in place.")
    print("Make sure you have committed your changes or have a backup!")
    
    response = input("\nProceed with fixing imports? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    print()
    
    # Fix imports
    if fix_all_api_imports():
        # Verify the fixes
        if verify_imports():
            print("\n🎉 All imports fixed successfully!")
            print("\nYou can now start the API with:")
            print("  uvicorn src.api.main:app --reload")
        else:
            print("\n⚠️  Some imports still have issues.")
            print("Check the error messages above for details.")
    else:
        print("\n❌ Import fixing failed")

if __name__ == "__main__":
    main()
