# Multi-stage Dockerfile for Cognitive Meeting Intelligence System

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install production dependencies
RUN poetry install --only main --no-root

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r cognitive && useradd -r -g cognitive cognitive

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=cognitive:cognitive . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R cognitive:cognitive /app/data /app/logs /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src \
    LOG_LEVEL=INFO

# Switch to non-root user
USER cognitive

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development environment (optional)
FROM runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for development
COPY --from=builder /opt/poetry /opt/poetry
RUN ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Install all dependencies including dev
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --with dev,docs,security --no-root

USER cognitive

# Use reload for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Testing environment
FROM development as testing

USER root

# Install test-specific tools
RUN poetry install --with dev,security

USER cognitive

# Set test environment
ENV TESTING=1

# Run tests by default
CMD ["pytest", "-v", "--cov=src", "--cov-report=term-missing"]