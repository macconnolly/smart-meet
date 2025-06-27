#!/usr/bin/env python3
"""
Check project state before development session.

Reference: structured_development_workflow memory
Ensures project is in good state before starting work.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def check_git_status():
    """Ensure clean git state."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if result.stdout:
            print("❌ Uncommitted changes found:")
            print(result.stdout)
            return False
        print("✅ Git working directory clean")
        return True
    except subprocess.CalledProcessError:
        print("❌ Git not initialized or error checking status")
        return False


def check_tests():
    """Run quick smoke tests."""
    test_path = Path("tests/unit/")
    if not test_path.exists():
        print("⚠️  No unit tests found (expected for initial setup)")
        return True

    try:
        result = subprocess.run(["pytest", "tests/unit/", "-x", "-q"], capture_output=True)
        if result.returncode != 0:
            print("❌ Unit tests failing")
            return False
        print("✅ Unit tests passing")
        return True
    except FileNotFoundError:
        print("⚠️  pytest not installed (run: pip install pytest)")
        return True  # Warning, not error


def check_dependencies():
    """Verify all dependencies installed."""
    missing = []
    dependencies = {
        "fastapi": "FastAPI",
        "qdrant_client": "Qdrant Client",
        "onnxruntime": "ONNX Runtime",
        "numpy": "NumPy",
        "pydantic": "Pydantic",
    }

    for module, name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False

    print("✅ Core dependencies installed")
    return True


def check_services():
    """Check if required services are running."""
    # Check Qdrant
    try:
        import requests

        response = requests.get("http://localhost:6333/", timeout=2)
        if response.status_code in [200, 404]:  # 404 is ok, means service is up
            print("✅ Qdrant is running")
            return True
    except:
        print("⚠️  Qdrant not running (run: docker-compose up -d)")
        return True  # Warning, not error for initial setup


def check_project_structure():
    """Verify essential directories exist."""
    required_dirs = [
        "src/models",
        "src/storage/sqlite",
        "src/storage/qdrant",
        "src/embedding",
        "src/extraction/dimensions",
        "src/pipeline",
        "src/api",
        "tests/unit",
        "scripts",
        "docs/progress",
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)

    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False

    print("✅ Project structure intact")
    return True


def check_essential_files():
    """Verify critical files exist."""
    essential_files = [
        "IMPLEMENTATION_GUIDE.md",
        "src/models/entities.py",
        "requirements.txt",
        "Makefile",
        ".gitignore",
    ]

    missing = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print(f"❌ Missing essential files: {', '.join(missing)}")
        return False

    print("✅ Essential files present")
    return True


def check_progress_tracking():
    """Check if progress documentation is up to date."""
    progress_dir = Path("docs/progress")
    if not progress_dir.exists():
        print("❌ Progress directory missing")
        return False

    progress_files = list(progress_dir.glob("*.md"))
    if not progress_files:
        print("⚠️  No progress documents yet (create one at session end)")
        return True

    latest = max(progress_files, key=lambda p: p.stat().st_mtime)
    print(f"✅ Latest progress doc: {latest.name}")
    return True


def main():
    """Run all checks."""
    print("🔍 Checking project state...\n")

    checks = [
        ("Git Status", check_git_status),
        ("Project Structure", check_project_structure),
        ("Essential Files", check_essential_files),
        ("Dependencies", check_dependencies),
        ("Tests", check_tests),
        ("Services", check_services),
        ("Progress Tracking", check_progress_tracking),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            results.append(False)
        print()  # Blank line between checks

    # Summary
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})!")
        print("\n🚀 Project ready for development!")
        sys.exit(0)
    else:
        print(f"⚠️  Some checks failed ({passed}/{total} passed)")
        print("\n💡 Fix issues before starting development")
        print("   Non-critical warnings (⚠️) can be ignored for initial setup")
        sys.exit(1 if passed < total - 2 else 0)  # Allow 2 warnings


if __name__ == "__main__":
    main()
