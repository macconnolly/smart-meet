"""
Core configuration for the Cognitive Meeting Intelligence system.

This module manages all configuration settings using Pydantic BaseSettings
for environment variable management and validation.
"""

from typing import List, Optional, Union
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Settings are loaded from environment variables or .env file.
    """

    # Application settings
    app_name: str = "Cognitive Meeting Intelligence"
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_prefix: str = Field("/api/v2", env="API_PREFIX")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"], env="CORS_ORIGINS"
    )

    # Database settings
    database_url: str = Field("sqlite:///data/cognitive.db", env="DATABASE_URL")
    database_pool_size: int = Field(5, env="DATABASE_POOL_SIZE")

    # Qdrant settings
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_prefer_grpc: bool = Field(False, env="QDRANT_PREFER_GRPC")
    qdrant_timeout: int = Field(30, env="QDRANT_TIMEOUT")

    # Model settings
    model_path: str = Field("models/embeddings/model.onnx", env="MODEL_PATH")
    tokenizer_path: str = Field("models/embeddings/tokenizer", env="TOKENIZER_PATH")
    model_cache_size: int = Field(10000, env="MODEL_CACHE_SIZE")

    # Pipeline settings
    pipeline_batch_size: int = Field(50, env="PIPELINE_BATCH_SIZE")
    pipeline_parallel: bool = Field(True, env="PIPELINE_PARALLEL")
    min_memory_length: int = Field(10, env="MIN_MEMORY_LENGTH")
    max_memory_length: int = Field(1000, env="MAX_MEMORY_LENGTH")

    # Cognitive settings
    activation_threshold: float = Field(0.7, env="ACTIVATION_THRESHOLD")
    max_activations: int = Field(50, env="MAX_ACTIVATIONS")
    activation_decay_factor: float = Field(0.8, env="ACTIVATION_DECAY_FACTOR")
    activation_max_depth: int = Field(5, env="ACTIVATION_MAX_DEPTH")

    # Security settings
    secret_key: str = Field("your-secret-key-here-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Feature flags
    enable_activation_spreading: bool = Field(True, env="ENABLE_ACTIVATION_SPREADING")
    enable_bridge_discovery: bool = Field(True, env="ENABLE_BRIDGE_DISCOVERY")
    enable_memory_consolidation: bool = Field(
        False, env="ENABLE_MEMORY_CONSOLIDATION"  # Not implemented in Phase 1
    )

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment is one of allowed values."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development-specific settings."""

    debug: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production-specific settings."""

    debug: bool = False
    log_level: str = "INFO"

    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Ensure secret key is changed in production."""
        if v == "your-secret-key-here-change-in-production":
            raise ValueError("Secret key must be changed in production")
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns settings based on ENVIRONMENT variable.
    """
    env = os.getenv("ENVIRONMENT", "development")

    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    else:
        return Settings()


# Utility functions for common settings access
def get_database_url() -> str:
    """Get database URL."""
    return get_settings().database_url


def get_qdrant_config() -> dict:
    """Get Qdrant configuration."""
    settings = get_settings()
    return {
        "host": settings.qdrant_host,
        "port": settings.qdrant_port,
        "api_key": settings.qdrant_api_key,
        "prefer_grpc": settings.qdrant_prefer_grpc,
        "timeout": settings.qdrant_timeout,
    }


def get_model_config() -> dict:
    """Get model configuration."""
    settings = get_settings()
    return {
        "model_path": settings.model_path,
        "tokenizer_path": settings.tokenizer_path,
        "cache_size": settings.model_cache_size,
    }


def get_pipeline_config() -> dict:
    """Get pipeline configuration."""
    settings = get_settings()
    return {
        "batch_size": settings.pipeline_batch_size,
        "parallel": settings.pipeline_parallel,
        "min_memory_length": settings.min_memory_length,
        "max_memory_length": settings.max_memory_length,
    }


def get_activation_config() -> dict:
    """Get activation spreading configuration."""
    settings = get_settings()
    return {
        "threshold": settings.activation_threshold,
        "max_activations": settings.max_activations,
        "decay_factor": settings.activation_decay_factor,
        "max_depth": settings.activation_max_depth,
    }


# Environment variable documentation
ENV_VAR_DOCS = """
Environment Variables:

Core:
- ENVIRONMENT: development|staging|production (default: development)
- DEBUG: Enable debug mode (default: False)
- LOG_LEVEL: Logging level (default: INFO)

API:
- API_HOST: API host (default: 0.0.0.0)
- API_PORT: API port (default: 8000)
- CORS_ORIGINS: Comma-separated CORS origins

Database:
- DATABASE_URL: SQLite database URL (default: sqlite:///data/cognitive.db)
- DATABASE_POOL_SIZE: Connection pool size (default: 5)

Qdrant:
- QDRANT_HOST: Qdrant host (default: localhost)
- QDRANT_PORT: Qdrant port (default: 6333)
- QDRANT_API_KEY: Optional API key for Qdrant

Models:
- MODEL_PATH: Path to ONNX model (default: models/embeddings/model.onnx)
- TOKENIZER_PATH: Path to tokenizer (default: models/embeddings/tokenizer)
- MODEL_CACHE_SIZE: Embedding cache size (default: 10000)

Pipeline:
- PIPELINE_BATCH_SIZE: Batch size for processing (default: 50)
- PIPELINE_PARALLEL: Enable parallel processing (default: True)
- MIN_MEMORY_LENGTH: Minimum memory length (default: 10)
- MAX_MEMORY_LENGTH: Maximum memory length (default: 1000)

Cognitive:
- ACTIVATION_THRESHOLD: Minimum activation to propagate (default: 0.7)
- MAX_ACTIVATIONS: Maximum memories to activate (default: 50)
- ACTIVATION_DECAY_FACTOR: Decay per level (default: 0.8)
- ACTIVATION_MAX_DEPTH: Maximum search depth (default: 5)

Security:
- SECRET_KEY: Application secret key (MUST change in production)
- ACCESS_TOKEN_EXPIRE_MINUTES: JWT token expiration (default: 30)

Features:
- ENABLE_ACTIVATION_SPREADING: Enable activation spreading (default: True)
- ENABLE_BRIDGE_DISCOVERY: Enable bridge discovery (default: True)
- ENABLE_MEMORY_CONSOLIDATION: Enable consolidation (default: False)
"""


if __name__ == "__main__":
    # Print current configuration
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Database: {settings.database_url}")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"API: {settings.api_host}:{settings.api_port}")
