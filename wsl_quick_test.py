#!/usr/bin/env python3
"""
WSL Quick Test Runner
A simple way to get testing without complex setup
"""

import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run a command and return success status"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}")
    if check and result.returncode != 0:
        return False
    return True

def main():
    print("🚀 WSL Quick Test Setup")
    print("=" * 40)
    
    # Check if in WSL
    if not os.path.exists("/proc/version") or "microsoft" not in open("/proc/version").read().lower():
        print("⚠️  Not running in WSL. This script is optimized for WSL.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Option 1: Try Docker first (no Python issues!)
    print("\n1️⃣ Checking Docker option...")
    if run_cmd("docker --version", check=False):
        print("✅ Docker found! This is the easiest way.")
        use_docker = input("\nUse Docker for testing? (y/n): ")
        if use_docker.lower() == 'y':
            print("\nStarting Docker services...")
            if run_cmd("docker-compose up -d"):
                print("\n✅ Docker services started!")
                print("\nTo run tests:")
                print("  docker-compose exec api pytest")
                print("  docker-compose exec api pytest -v tests/test_memory_repo.py")
                print("\nTo view logs:")
                print("  docker-compose logs -f")
                return
    
    # Option 2: Simple venv setup
    print("\n2️⃣ Setting up simple Python environment...")
    
    # Check Python
    python_cmd = "python3.11" if run_cmd("python3.11 --version", check=False) else "python3"
    
    # Create minimal venv
    if not os.path.exists("venv"):
        print(f"\nCreating virtual environment with {python_cmd}...")
        run_cmd(f"{python_cmd} -m venv venv")
    
    # Create activation script
    activate_script = """#!/bin/bash
# Quick activation script
source venv/bin/activate
echo "✅ Virtual environment activated"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
"""
    
    with open("activate.sh", "w") as f:
        f.write(activate_script)
    os.chmod("activate.sh", 0o755)
    
    # Install minimal test requirements
    print("\nInstalling minimal test requirements...")
    test_requirements = """pytest>=7.4.4
pytest-asyncio>=0.23.3
pytest-cov>=4.1.0
"""
    
    with open("test-requirements.txt", "w") as f:
        f.write(test_requirements)
    
    run_cmd("venv/bin/pip install --upgrade pip")
    run_cmd("venv/bin/pip install -r test-requirements.txt")
    
    # Create a simple test
    print("\nCreating simple test to verify setup...")
    simple_test = '''"""Simple test to verify environment"""

def test_python_version():
    """Test Python version is 3.8+"""
    import sys
    assert sys.version_info >= (3, 8)

def test_imports():
    """Test basic imports work"""
    import os
    import json
    import sqlite3
    assert True

def test_project_structure():
    """Test project structure exists"""
    import os
    assert os.path.exists("src")
    assert os.path.exists("tests")
'''
    
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_simple.py", "w") as f:
        f.write(simple_test)
    
    # Run the simple test
    print("\n3️⃣ Running simple test...")
    if run_cmd("venv/bin/pytest tests/test_simple.py -v"):
        print("\n✅ Basic testing works!")
        
        # Try installing full requirements
        install_full = input("\nInstall full project requirements? (y/n): ")
        if install_full.lower() == 'y':
            print("\nInstalling full requirements (this may take a few minutes)...")
            if run_cmd("venv/bin/pip install -r requirements.txt"):
                print("\n✅ Full requirements installed!")
                print("\nYou can now run all tests:")
                print("  source venv/bin/activate")
                print("  pytest")
            else:
                print("\n⚠️  Some requirements failed. You can still run basic tests.")
    
    print("\n" + "=" * 40)
    print("Quick Commands:")
    print("  source venv/bin/activate  # or ./activate.sh")
    print("  pytest tests/test_simple.py  # Run simple test")
    print("  pytest -v  # Run all tests (if requirements installed)")
    print("\nAlternatively, use Make:")
    print("  make setup  # Full setup")
    print("  make test   # Run tests")
    print("  make docker-up  # Use Docker instead")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry using Docker instead: docker-compose up -d")
