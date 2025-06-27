#!/usr/bin/env python3
"""
Complete setup script for Cognitive Meeting Intelligence project.
Runs all initialization steps in the correct order.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def create_directory_structure():
    """Create all necessary directories."""
    print("\n📁 Creating directory structure...")

    directories = [
        # Source directories
        "src/core",
        "src/models",
        "src/extraction/dimensions",
        "src/embedding",
        "src/cognitive/activation",
        "src/cognitive/bridges",
        "src/cognitive/consolidation",
        "src/storage/sqlite/repositories",
        "src/storage/qdrant",
        "src/pipeline",
        "src/api/routers",

        # Test directories
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/fixtures",

        # Other directories
        "scripts",
        "config",
        "models/all-MiniLM-L6-v2",
        "data/qdrant",
        "data/transcripts",
        "docs/architecture",
        "docs/api",
        "docs/development",
        "docs/cognitive",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created")


def create_init_files():
    """Create __init__.py files in all Python packages."""
    print("\n📄 Creating __init__.py files...")

    for root, dirs, files in os.walk("src"):
        if "__pycache__" not in root:
            init_file = Path(root) / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    for root, dirs, files in os.walk("tests"):
        if "__pycache__" not in root:
            init_file = Path(root) / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    print("✅ __init__.py files created")


def create_env_file():
    """Create .env file from template."""
    print("\n🔐 Creating .env file...")

    env_content = """# Environment Configuration
ENVIRONMENT=development
DATABASE_URL=sqlite:///./data/memories.db
QDRANT_HOST=localhost
QDRANT_PORT=6333
ONNX_MODEL_PATH=models/all-MiniLM-L6-v2
LOG_LEVEL=INFO
CACHE_SIZE=10000
CACHE_TTL=3600
"""

    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("⚠️  .env file already exists, skipping")

    # Also create .env.example
    env_example = Path(".env.example")
    if not env_example.exists():
        env_example.write_text(env_content)


def create_config_files():
    """Create configuration files."""
    print("\n⚙️  Creating configuration files...")

    # Default config
    default_config = """# Default Configuration
app:
  name: "Cognitive Meeting Intelligence"
  version: "1.0.0"
  debug: true

qdrant:
  host: localhost
  port: 6333
  l0_collection: cognitive_concepts
  l1_collection: cognitive_contexts
  l2_collection: cognitive_episodes

  # HNSW parameters
  l0_hnsw:
    m: 32
    ef_construct: 400
  l1_hnsw:
    m: 24
    ef_construct: 300
  l2_hnsw:
    m: 16
    ef_construct: 200

extraction:
  min_memory_length: 10
  max_memory_length: 500
  batch_size: 100

dimensions:
  temporal:
    urgency_keywords: ["urgent", "asap", "immediately", "critical", "now"]
    deadline_patterns: ["by", "before", "until", "deadline"]
  emotional:
    use_vader: true

cognitive:
  activation:
    threshold: 0.7
    max_activations: 50
    decay_factor: 0.8
    max_depth: 3
  bridges:
    novelty_weight: 0.6
    connection_weight: 0.4
    max_bridges: 5
  consolidation:
    min_cluster_size: 5
    similarity_threshold: 0.8

performance:
  embedding_cache_size: 10000
  query_cache_size: 1000
  connection_pool_size: 10
"""

    config_file = Path("config/default.yaml")
    config_file.parent.mkdir(exist_ok=True)
    if not config_file.exists():
        config_file.write_text(default_config)
        print("✅ config/default.yaml created")
    else:
        print("⚠️  config/default.yaml already exists, skipping")

    # Logging config
    logging_config = """version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: default
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false

  uvicorn:
    level: INFO
    handlers: [console]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
"""

    logging_file = Path("config/logging.yaml")
    if not logging_file.exists():
        logging_file.write_text(logging_config)
        print("✅ config/logging.yaml created")


def create_makefile():
    """Create Makefile with common commands."""
    print("\n🛠️  Creating Makefile...")

    makefile_content = """# Makefile for Cognitive Meeting Intelligence

.PHONY: help setup install test run quality clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make setup      - Complete project setup"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make run        - Run the API server"
	@echo "  make quality    - Run code quality checks"
	@echo "  make clean      - Clean cache and temp files"
	@echo "  make docker-up  - Start Docker services"
	@echo "  make docker-down - Stop Docker services"

setup: install docker-up
	python scripts/download_model.py
	python scripts/init_db.py
	python scripts/init_qdrant.py
	python -c "import nltk; nltk.download('vader_lexicon')"
	@echo "✅ Setup complete! Run 'make run' to start the API."

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term

run:
	uvicorn src.api.main:app --reload --port 8000

quality:
	black src/ tests/ --line-length 100
	flake8 src/ tests/ --max-line-length 100
	mypy src/

format:
	black src/ tests/ --line-length 100
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
"""

    makefile = Path("Makefile")
    if not makefile.exists():
        makefile.write_text(makefile_content)
        print("✅ Makefile created")
    else:
        print("⚠️  Makefile already exists, skipping")


def create_docker_compose():
    """Create docker-compose.yml file."""
    print("\n🐳 Creating docker-compose.yml...")

    docker_compose_content = """version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: cognitive-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Future: Add Redis for caching
  # redis:
  #   image: redis:7-alpine
  #   container_name: cognitive-redis
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - ./data/redis:/data
  #   restart: unless-stopped

networks:
  default:
    name: cognitive-network
"""

    docker_file = Path("docker-compose.yml")
    if not docker_file.exists():
        docker_file.write_text(docker_compose_content)
        print("✅ docker-compose.yml created")
    else:
        print("⚠️  docker-compose.yml already exists, skipping")


def create_requirements_dev():
    """Create requirements-dev.txt file."""
    print("\n📦 Creating requirements-dev.txt...")

    requirements_dev = """# Development dependencies
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
pytest-benchmark==4.0.0
black==23.12.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2
ipython==8.19.0
jupyter==1.0.0
pre-commit==3.6.0
"""

    req_file = Path("requirements-dev.txt")
    if not req_file.exists():
        req_file.write_text(requirements_dev)
        print("✅ requirements-dev.txt created")


def create_pytest_ini():
    """Create pytest.ini configuration."""
    print("\n🧪 Creating pytest.ini...")

    pytest_content = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance benchmarks
    slow: Slow tests
addopts = -v --tb=short
"""

    pytest_file = Path("pytest.ini")
    if not pytest_file.exists():
        pytest_file.write_text(pytest_content)
        print("✅ pytest.ini created")


def create_setup_cfg():
    """Create setup.cfg for tool configurations."""
    print("\n⚙️  Creating setup.cfg...")

    setup_cfg_content = """[flake8]
max-line-length = 100
exclude = .git,__pycache__,docs/,build/,dist/,.venv/,venv/
ignore = E203,W503,E501

[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
ignore_missing_imports = True
exclude = venv/|.venv/|tests/

[isort]
profile = black
line_length = 100
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
ensure_newline_before_comments = True
"""

    setup_file = Path("setup.cfg")
    if not setup_file.exists():
        setup_file.write_text(setup_cfg_content)
        print("✅ setup.cfg created")


def check_prerequisites():
    """Check if required tools are installed."""
    print("\n🔍 Checking prerequisites...")

    requirements = {
        "python": "Python 3.11+",
        "pip": "pip",
        "docker": "Docker",
        "docker-compose": "Docker Compose"
    }

    all_good = True
    for cmd, name in requirements.items():
        try:
            if cmd == "docker-compose":
                # Try both docker-compose and docker compose
                try:
                    subprocess.run(["docker-compose", "--version"],
                                 capture_output=True, check=True)
                except:
                    subprocess.run(["docker", "compose", "version"],
                                 capture_output=True, check=True)
            else:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
            print(f"✅ {name} is installed")
        except:
            print(f"❌ {name} is not installed or not in PATH")
            all_good = False

    if not all_good:
        print("\n⚠️  Please install missing prerequisites before continuing")
        return False

    return True


def main():
    """Run complete setup process."""
    print("🚀 Cognitive Meeting Intelligence - Complete Setup")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Create directory structure
    create_directory_structure()

    # Create configuration files
    create_init_files()
    create_env_file()
    create_config_files()
    create_makefile()
    create_docker_compose()
    create_requirements_dev()
    create_pytest_ini()
    create_setup_cfg()

    # Create placeholder scripts if they don't exist
    scripts_to_create = [
        ("scripts/download_model.py", "# TODO: Implement ONNX model download
print('Model download not yet implemented')"),
        # init_db.py and init_qdrant.py are already implemented
    ]

    for script_path, content in scripts_to_create:
        script = Path(script_path)
        if not script.exists():
            script.parent.mkdir(exist_ok=True)
            script.write_text(content)
            print(f"✅ Created placeholder: {script_path}")

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Create and activate a virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("   pip install -r requirements-dev.txt")
    print("
3. Start Docker services (Qdrant vector database):")
    print("   docker-compose up -d")
    print("\n4. Initialize the project:")
    print("   make setup")
    print("\n5. Start development:")
    print("   make run")
    print("\n📚 See IMPLEMENTATION_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
