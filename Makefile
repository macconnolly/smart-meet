# Makefile for Cognitive Meeting Intelligence
# Optimized for WSL/Linux development

.PHONY: help setup test run clean docker-up docker-down format lint check

# Default shell for WSL
SHELL := /bin/bash
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-wsl: ## Check if running in WSL
	@if grep -q microsoft /proc/version; then \
		echo "$(GREEN)✓ Running in WSL$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Not running in WSL$(NC)"; \
	fi

setup: check-wsl ## Full setup: venv, dependencies, and databases
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Upgrading pip..."
	@$(PIP) install --upgrade pip wheel setuptools
	@echo "Installing dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Downloading NLTK data..."
	@$(VENV)/bin/python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "Activate with: source $(VENV)/bin/activate"

install: ## Install/update dependencies
	@$(PIP) install -r requirements.txt

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@$(PYTEST) -v

test-cov: ## Run tests with coverage
	@$(PYTEST) --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

test-fast: ## Run tests in parallel
	@$(PIP) install pytest-xdist > /dev/null 2>&1
	@$(PYTEST) -n auto

test-unit: ## Run unit tests only
	@$(PYTEST) tests/unit -v

test-integration: ## Run integration tests only
	@$(PYTEST) tests/integration -v

run: ## Run the API server
	@echo "$(YELLOW)Starting API server...$(NC)"
	@$(UVICORN) src.api.simple_api:app --reload --host 0.0.0.0 --port 8000

run-cognitive: ## Run the cognitive API server
	@$(UVICORN) src.api.cognitive_api:app --reload --host 0.0.0.0 --port 8000

docker-up: ## Start Docker services
	@echo "$(YELLOW)Starting Docker services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Qdrant: http://localhost:6333"

docker-down: ## Stop Docker services
	@docker-compose down

docker-test: ## Run tests in Docker
	@docker-compose exec api pytest -v

docker-logs: ## Show Docker logs
	@docker-compose logs -f

init-db: ## Initialize databases
	@echo "$(YELLOW)Initializing databases...$(NC)"
	@mkdir -p data models
	@if [ -f "scripts/init_db.py" ]; then \
		$(VENV)/bin/python scripts/init_db.py; \
		echo "$(GREEN)✓ SQLite initialized$(NC)"; \
	fi
	@if curl -s http://localhost:6333/health > /dev/null 2>&1; then \
		if [ -f "scripts/init_qdrant.py" ]; then \
			$(VENV)/bin/python scripts/init_qdrant.py; \
			echo "$(GREEN)✓ Qdrant initialized$(NC)"; \
		fi \
	else \
		echo "$(YELLOW)⚠ Qdrant not running. Start with: make docker-up$(NC)"; \
	fi

format: ## Format code with Black
	@echo "$(YELLOW)Formatting code...$(NC)"
	@$(VENV)/bin/black src/ tests/ --line-length 100
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Lint code with flake8
	@echo "$(YELLOW)Linting code...$(NC)"
	@$(VENV)/bin/flake8 src/ tests/ --max-line-length 100

type-check: ## Type check with mypy
	@echo "$(YELLOW)Type checking...$(NC)"
	@$(VENV)/bin/mypy src/

check: format lint type-check ## Run all code quality checks

clean: ## Clean up cache and temporary files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .pytest_cache/ .mypy_cache/ 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean ## Clean everything including venv
	@rm -rf $(VENV)
	@echo "$(GREEN)✓ All cleaned$(NC)"

health: ## Run health check
	@$(VENV)/bin/python -c "import sys; print(f'Python: {sys.version}')"
	@echo ""
	@echo "Checking packages:"
	@for pkg in fastapi qdrant_client sqlalchemy pytest; do \
		if $(VENV)/bin/python -c "import $$pkg" 2>/dev/null; then \
			echo "$(GREEN)✓ $$pkg$(NC)"; \
		else \
			echo "$(RED)✗ $$pkg$(NC)"; \
		fi \
	done
	@echo ""
	@echo "Checking services:"
	@if curl -s http://localhost:8000/health > /dev/null 2>&1; then \
		echo "$(GREEN)✓ API running$(NC)"; \
	else \
		echo "$(YELLOW)⚠ API not running$(NC)"; \
	fi
	@if curl -s http://localhost:6333/health > /dev/null 2>&1; then \
		echo "$(GREEN)✓ Qdrant running$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Qdrant not running$(NC)"; \
	fi

quick-start: setup docker-up init-db ## Complete quick start
	@echo ""
	@echo "$(GREEN)✓ Everything is set up!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  source $(VENV)/bin/activate"
	@echo "  make test"
	@echo "  make run"

# Development workflow shortcuts
dev: ## Start development (docker + api + logs)
	@make docker-up
	@sleep 2
	@make init-db
	@make run

test-watch: ## Watch for changes and run tests
	@$(PIP) install pytest-watch > /dev/null 2>&1
	@$(VENV)/bin/ptw tests/ -- -v

# WSL-specific commands
wsl-perf: ## Check if project is in WSL filesystem
	@if pwd | grep -q "^/mnt/"; then \
		echo "$(RED)⚠ WARNING: Project is in Windows filesystem (slow)$(NC)"; \
		echo "Move to WSL filesystem for better performance:"; \
		echo "  cp -r . ~/dev/meet"; \
	else \
		echo "$(GREEN)✓ Project is in WSL filesystem (fast)$(NC)"; \
	fi
