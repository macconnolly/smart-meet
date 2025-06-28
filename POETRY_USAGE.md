# Poetry Usage Guide

## Installation Complete! ✅

Poetry has successfully installed all dependencies. Here's how to use it:

## Running Commands

### Option 1: Using `poetry run` (Recommended)
```bash
# Run any command in the virtual environment
poetry run python your_script.py
poetry run pytest
poetry run uvicorn src.api.main:app --reload
```

### Option 2: Activating the Virtual Environment
```bash
# For bash/zsh
source .venv/bin/activate

# For fish
source .venv/bin/activate.fish

# For Windows
.venv\Scripts\activate

# To deactivate
deactivate
```

### Option 3: Using Poetry's env activate (new in Poetry 2.0)
```bash
# Get activation instructions
poetry env activate

# Or directly get the path
poetry env info --path
```

## Common Commands

### Start the API
```bash
poetry run uvicorn src.api.main:app --reload
```

### Run Tests
```bash
poetry run pytest
poetry run pytest --cov=src
```

### Code Quality
```bash
poetry run black src tests
poetry run flake8 src tests
poetry run mypy src
```

### Install Pre-commit Hooks
```bash
poetry run pre-commit install
```

### Add New Dependencies
```bash
# Add to main dependencies
poetry add package-name

# Add to dev dependencies
poetry add --group dev package-name

# Add to specific group
poetry add --group docs mkdocs
```

### Update Dependencies
```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package-name
```

### Export Requirements
```bash
# Export all dependencies
poetry export -f requirements.txt --output requirements.txt

# Export only production dependencies
poetry export -f requirements.txt --only main --output requirements-prod.txt
```

## Environment Information
- Python Version: 3.12.3
- Virtual Environment: `.venv/`
- All dependencies installed with `--all-extras`

## Troubleshooting

If you get "command not found" errors:
1. Make sure you're using `poetry run` prefix
2. Or activate the virtual environment first
3. Check that Poetry is in your PATH: `which poetry`

If you need to recreate the environment:
```bash
poetry env remove python
poetry install --all-extras
```