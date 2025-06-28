# Setting Up Poetry for Cognitive Meeting Intelligence

## Installation
```bash
# Install Poetry (Windows PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Or using pip
pip install poetry
```

## Initialize Poetry in the Project
```bash
# Convert existing requirements to Poetry
poetry init
poetry add $(cat requirements.txt)

# Or create a proper pyproject.toml
```

## Poetry pyproject.toml Configuration
```toml
[tool.poetry]
name = "cognitive-meeting-intelligence"
version = "0.1.0"
description = "AI-powered meeting intelligence system"
authors = ["Your Name <email@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
qdrant-client = "^1.7.0"
sqlalchemy = "^2.0.25"
alembic = "^1.13.1"
onnxruntime = "^1.16.3"
sentence-transformers = "^2.2.2"
numpy = "^1.26.3"
scikit-learn = "^1.4.0"
nltk = "^3.8.1"
vaderSentiment = "^3.3.2"
python-dateutil = "^2.8.2"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
python-multipart = "^0.0.6"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.4"
pytest-asyncio = "^0.23.3"
pytest-cov = "^4.1.0"
black = "^23.12.1"
flake8 = "^7.0.0"
mypy = "^1.8.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## Usage
```bash
# Install dependencies
poetry install

# Run commands in the virtual environment
poetry run pytest
poetry run uvicorn src.api.simple_api:app --reload

# Activate shell
poetry shell

# Add new dependencies
poetry add requests
poetry add --group dev pytest-mock
```
