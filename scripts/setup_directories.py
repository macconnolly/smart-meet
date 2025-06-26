"""Setup directory structure for Cognitive Meeting Intelligence project"""
import os
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

# Define directory structure
directories = [
    "src/core",
    "src/models",
    "src/extraction",
    "src/extraction/dimensions",
    "src/embedding",
    "src/cognitive",
    "src/cognitive/activation",
    "src/cognitive/bridges",
    "src/cognitive/consolidation",
    "src/storage",
    "src/storage/sqlite",
    "src/storage/sqlite/repositories",
    "src/storage/qdrant",
    "src/api",
    "src/api/routers",
    "src/cli",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "scripts",
    "models",
    "models/all-MiniLM-L6-v2",
    "data",
    "data/qdrant",
    "data/transcripts",
    "config",
    "docs"
]

# Create directories
for dir_path in directories:
    full_path = project_root / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

# Create __init__.py files in Python packages
python_dirs = [d for d in directories if d.startswith(("src", "tests"))]
for dir_path in python_dirs:
    init_file = project_root / dir_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created: {dir_path}/__init__.py")

print("\nDirectory structure created successfully!")
