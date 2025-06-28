# Python Environment Troubleshooting Guide

## Common Issues and Solutions

### 1. **"Module not found" errors**

**Problem**: ImportError when trying to run tests or the application

**Solutions**:
- Make sure you're in the activated virtual environment
- Check if the module is installed: `pip list | grep module_name`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- For ONNX issues on Windows: `pip install onnxruntime-gpu` or `pip install onnxruntime --force-reinstall`

### 2. **Virtual Environment Activation Issues (Windows)**

**Problem**: Scripts don't run or "not recognized as internal or external command"

**Solutions**:
```powershell
# PowerShell execution policy issue
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Use full path
.\venv\Scripts\activate

# Or use cmd instead of PowerShell
cmd
venv\Scripts\activate.bat
```

### 3. **Conflicting Python Versions**

**Problem**: Wrong Python version or multiple Python installations

**Solutions**:
```bash
# Check Python version
python --version

# Use specific Python version
py -3.11 -m venv venv

# Or specify full path
C:\Python311\python.exe -m venv venv
```

### 4. **Dependency Conflicts**

**Problem**: Version conflicts between packages

**Solutions**:
```bash
# Create fresh environment
deactivate
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with constraints
pip install -r requirements.txt --constraint requirements.txt
```

### 5. **Docker Alternative (Skip All Python Issues!)**

If you're still having issues, Docker is your best friend:

```bash
# One-time setup
docker-compose up -d

# Run any command in container
docker-compose exec api python --version
docker-compose exec api pytest
docker-compose exec api python scripts/init_db.py
```

### 6. **Quick Health Check Script**

Save this as `check_health.py`:

```python
#!/usr/bin/env python
import sys
import importlib

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

print("Python Environment Health Check")
print("=" * 40)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

modules = [
    "fastapi",
    "uvicorn", 
    "pydantic",
    "sqlalchemy",
    "qdrant_client",
    "onnxruntime",
    "sentence_transformers",
    "numpy",
    "sklearn",
    "nltk",
]

all_ok = True
for module in modules:
    ok, msg = check_import(module)
    status = "✅" if ok else "❌"
    print(f"{status} {module:<20} {'' if ok else msg}")
    all_ok = all_ok and ok

print()
if all_ok:
    print("✅ All dependencies are installed!")
else:
    print("❌ Some dependencies are missing. Run: pip install -r requirements.txt")
```

### 7. **Platform-Specific Issues**

**Windows + ONNX Runtime**:
```bash
# If ONNX fails, try:
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime==1.16.3 --no-cache-dir
```

**MacOS + scikit-learn**:
```bash
# If scikit-learn fails on M1/M2:
pip install --no-binary :all: scikit-learn
```

**Linux + SQLite**:
```bash
# If SQLite issues:
sudo apt-get install sqlite3 libsqlite3-dev
```

### 8. **Emergency Reset**

If nothing works, nuclear option:

```bash
# 1. Delete everything
rm -rf venv .venv __pycache__ 
rm -rf ~/.cache/pip  # Clear pip cache

# 2. Use Docker
docker-compose down -v  # Remove volumes too
docker-compose up --build  # Rebuild everything

# 3. Or use system packages (not recommended)
pip install --user -r requirements.txt
```

### 9. **Test Without Full Environment**

Create `minimal_test.py`:

```python
"""Minimal test that works without dependencies"""
import os
import sys
import json

print("Basic Python Test")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

# Test basic file operations
if os.path.exists("src"):
    print("✅ Source directory found")
else:
    print("❌ Source directory not found")

# Test JSON (always available)
data = {"status": "working"}
json_str = json.dumps(data)
print(f"✅ JSON working: {json_str}")

# Test SQLite (built-in)
try:
    import sqlite3
    conn = sqlite3.connect(":memory:")
    print("✅ SQLite working")
    conn.close()
except Exception as e:
    print(f"❌ SQLite error: {e}")
```

### 10. **Get Help**

If you're still stuck:

1. **Check versions**: 
   ```bash
   python --version  # Should be 3.11+
   pip --version
   ```

2. **Clean install**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt -v  # Verbose output
   ```

3. **Use Docker** - Seriously, it just works!

4. **Create issue with**:
   - Your OS and Python version
   - Full error message
   - Output of `pip freeze`
