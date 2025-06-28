#!/usr/bin/env python3
"""Test basic setup and imports."""

import sys
print(f"Python: {sys.version}")
print(f"Python Path: {sys.path[0]}")

# Test basic imports
try:
    import fastapi
    print(f"✓ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI: {e}")

try:
    import pydantic
    print(f"✓ Pydantic {pydantic.__version__}")
except ImportError as e:
    print(f"✗ Pydantic: {e}")

try:
    import uvicorn
    print(f"✓ Uvicorn {uvicorn.__version__}")
except ImportError as e:
    print(f"✗ Uvicorn: {e}")

# Test local imports
print("\nTesting local imports...")

try:
    from src.models.entities import Memory
    print("✓ Models import successful")
except ImportError as e:
    print(f"✗ Models: {e}")

# Simple FastAPI app test
print("\nCreating simple FastAPI app...")
try:
    from fastapi import FastAPI
    app = FastAPI(title="Test App")
    
    @app.get("/")
    def read_root():
        return {"status": "ok"}
    
    print("✓ Simple FastAPI app created successfully")
except Exception as e:
    print(f"✗ FastAPI app creation failed: {e}")

print("\nSetup test complete!")