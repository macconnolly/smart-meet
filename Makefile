# Makefile for Cognitive Meeting Intelligence
# Reference: IMPLEMENTATION_GUIDE.md - Day 6-7: API & Integration

.PHONY: help setup install test run clean docker-up docker-down format lint check install-dev pre-commit coverage docs profile clean-db

# Default target
help:
	@echo "Cognitive Meeting Intelligence - Development Commands"
	@echo "===================================================="
	@echo "Setup & Installation:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make install      - Install dependencies only"
	@echo "  make download-model - Download ONNX model"
	@echo ""
	@echo "Database & Storage:"
	@echo "  make init-db      - Initialize SQLite database"
	@echo "  make init-qdrant  - Initialize Qdrant collections"
	@echo "  make docker-up    - Start Qdrant with Docker"
	@echo "  make docker-down  - Stop Docker services"
	@echo ""
	@echo "Development:"
	@echo "  make run          - Run API server (development)"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run linting checks"
	@echo "  make typecheck    - Run type checking with mypy"
	@echo ""
	@echo "Quality & Testing:"
	@echo "  make check        - Run all quality checks"
	@echo "  make pre-commit   - Install and run pre-commit hooks"
	@echo "  make coverage     - Generate coverage report"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean temporary files"
	@echo "  make clean-db     - Clean database files"
	@echo "  make logs         - Show Docker logs"
	@echo "  make docs         - Build documentation"
	@echo "  make profile      - Profile the application"
	@echo "  make docker       - Build Docker image"
	@echo "  make reset        - Full reset and rebuild"
	@echo "  make install-dev  - Install development dependencies"

# Setup virtual environment and install dependencies
setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete\! Activate with: source venv/bin/activate"

# Install dependencies
install:
	pip install -r requirements.txt

# Download ONNX model
download-model:
	python scripts/download_model.py

# Initialize SQLite database
init-db:
	python scripts/init_db.py

# Initialize Qdrant collections
init-qdrant:
	python scripts/init_qdrant.py

# Start Docker services
docker-up:
	docker-compose up -d
	@echo "Waiting for Qdrant to be ready..."
	@sleep 5
	@echo "Qdrant is running at http://localhost:6333"

# Stop Docker services
docker-down:
	docker-compose down

# Run API server (development)
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Run unit tests
test-unit:
	pytest tests/unit/ -v

# Run integration tests
test-integration:
	pytest tests/integration/ -v

# Format code
format:
	black src/ tests/ scripts/ --line-length 100

# Run linting
lint:
	flake8 src/ tests/ scripts/ --max-line-length 100

# Run type checking
typecheck:
	mypy src/

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Show Docker logs
logs:
	docker-compose logs -f

# Quick start for development
quickstart: docker-up init-db init-qdrant download-model
	@echo "System ready\! Run 'make run' to start the API server."

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run all quality checks
check: format lint typecheck test

# Install and run pre-commit hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Generate coverage report
coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Build documentation
docs:
	mkdocs build
	@echo "Documentation built in site/"

# Profile the application
profile:
	python -m cProfile -o profile.stats src/api/cognitive_api.py
	@echo "Profile saved to profile.stats. Analyze with: python -m pstats profile.stats"

# Clean database files
clean-db:
	rm -f data/memories.db
	docker-compose down -v
	@echo "Database files cleaned"

# Build Docker image
docker: 
	docker build -t cognitive-meeting-intelligence .
	@echo "Docker image built: cognitive-meeting-intelligence"

# Full reset (clean everything and rebuild)
reset: clean clean-db
	rm -rf venv/
	make setup
	make docker-up
	make init-db
	make init-qdrant
	@echo "Full reset complete!"

# TODO Day 6-7: Add production deployment targets
# deploy:
#     Production deployment steps

# TODO Day 6-7: Add database migration targets
# migrate:
#     Database migration steps
