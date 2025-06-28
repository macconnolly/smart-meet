#!/usr/bin/env python
"""
Simple Test Runner for Cognitive Meeting Intelligence
Works with any Python environment setup
"""

import subprocess
import sys
import os
from pathlib import Path

def find_python():
    """Find the appropriate Python executable"""
    # Check common virtual environment locations
    venv_paths = [
        "venv/Scripts/python.exe",  # Windows venv
        "venv/bin/python",          # Unix venv
        ".venv/Scripts/python.exe", # Windows .venv
        ".venv/bin/python",         # Unix .venv
    ]
    
    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            return venv_path
    
    # Check if we're in a conda environment
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return sys.executable
    
    # Check if we're in a poetry environment
    if os.environ.get("POETRY_ACTIVE"):
        return sys.executable
    
    # Default to system Python
    return sys.executable

def run_minimal_test():
    """Run a minimal test without any dependencies"""
    print("\n🧪 Running minimal environment test...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Test basic imports
    test_imports = [
        ("Standard library", ["os", "sys", "json", "sqlite3"]),
        ("Core dependencies", ["fastapi", "pydantic", "uvicorn"]),
        ("ML dependencies", ["numpy", "sklearn", "sentence_transformers"]),
        ("Storage", ["sqlalchemy", "qdrant_client"]),
    ]
    
    for category, modules in test_imports:
        print(f"\n{category}:")
        for module in modules:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError as e:
                print(f"  ❌ {module} - {e}")

def run_pytest(args):
    """Run pytest with given arguments"""
    python = find_python()
    cmd = [python, "-m", "pytest"] + args
    
    print(f"\n🏃 Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    """Main test runner"""
    print("🧪 Cognitive Meeting Intelligence - Test Runner")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        pytest_available = True
    except ImportError:
        pytest_available = False
        print("\n⚠️  pytest not installed in current environment")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        run_minimal_test()
        return
    
    if not pytest_available:
        print("\nWould you like to:")
        print("1. Run minimal environment test")
        print("2. Install pytest and dependencies")
        print("3. Use Docker for testing")
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == "1":
            run_minimal_test()
        elif choice == "2":
            python = find_python()
            print(f"\nInstalling pytest using {python}...")
            subprocess.run([python, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-cov"])
            print("\nNow you can run the tests again!")
        elif choice == "3":
            print("\nTo run tests with Docker:")
            print("  docker-compose exec api pytest")
            print("\nOr for specific tests:")
            print("  docker-compose exec api pytest tests/test_memory_repo.py -v")
        return
    
    # Parse test arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not args:
        print("\nAvailable test options:")
        print("  python run_tests.py                    # Run all tests")
        print("  python run_tests.py --minimal          # Run minimal env test")
        print("  python run_tests.py tests/unit         # Run unit tests only")
        print("  python run_tests.py -v                 # Verbose output")
        print("  python run_tests.py --cov=src          # With coverage")
        print("  python run_tests.py -k test_memory     # Run tests matching pattern")
        print("\nRunning all tests...")
        args = ["-v"]
    
    exit_code = run_pytest(args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
