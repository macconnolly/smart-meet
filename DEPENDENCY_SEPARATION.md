# Dependency Separation Strategy

## Overview
This document outlines the strategy for separating the monolithic application into distinct API and ML components.

## Current State
- Single monolithic application with mixed dependencies
- API and ML components tightly coupled
- All dependencies installed together

## Target State
- Separate packages: `cognitive-meeting-api` and `cognitive-meeting-ml`
- Shared core package for common models
- Clean interfaces between components
- Independent deployment capabilities

## Dependency Groups

### 1. API Package Dependencies
**File**: `pyproject-api.toml`
- FastAPI, Uvicorn, Pydantic
- Database clients (SQLite, Qdrant)
- Basic numpy for vector operations
- HTTP client (aiohttp)

### 2. ML Package Dependencies  
**File**: `pyproject-ml.toml`
- ONNX Runtime, Transformers
- Scientific computing (numpy, scikit-learn)
- Text analysis (NLTK, VADER)
- Storage backends
- Optional: PyTorch for model conversion

### 3. Shared Core (Future)
- Common data models
- Shared utilities
- Interface definitions

## Migration Path

### Phase 1: Current (Completed)
- Created unified Poetry configuration
- Separated dependencies into logical groups
- Created template configurations for future split

### Phase 2: Interface Definition
1. Define clear API contracts between components
2. Create abstraction layers for cross-component communication
3. Implement dependency injection

### Phase 3: Package Extraction
1. Extract shared models to `cognitive-meeting-core`
2. Create separate packages with their own pyproject.toml
3. Update imports and module paths

### Phase 4: Independent Services
1. Create separate Docker images
2. Implement service communication (gRPC/REST)
3. Deploy as microservices

## Benefits
- Independent scaling of API and ML components
- Reduced deployment size for API-only instances
- Easier to maintain and test
- Clear separation of concerns
- Ability to use different Python versions if needed

## Usage

### For Monolithic Development (Current)
```bash
poetry install
poetry install --with dev,docs,security
poetry install --with ml-convert  # If converting models
```

### For API-Only Development (Future)
```bash
cd api/
poetry install
```

### For ML-Only Development (Future)
```bash
cd ml/
poetry install
poetry install --with ml-convert
```

## Dependency Installation Commands

### Install all dependencies
```bash
poetry install --all-extras
```

### Install only production dependencies
```bash
poetry install --only main
```

### Install with specific groups
```bash
poetry install --with dev,security
```

### Export requirements (for compatibility)
```bash
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --only main --output requirements-runtime.txt
```