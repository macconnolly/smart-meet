"""
FastAPI main application for the Cognitive Meeting Intelligence API.

This module sets up the FastAPI application with all routes, middleware,
and configuration for the cognitive meeting intelligence system.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routers import memories
from ..core.config import Settings, get_settings
from ..storage.sqlite.connection import DatabaseConnection
from ..storage.qdrant.vector_store import QdrantVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .dependencies import set_db_connection, set_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    """
    # Startup
    logger.info("Starting Cognitive Meeting Intelligence API...")

    settings = get_settings()

    # Initialize database connection
    db_connection = DatabaseConnection(db_path=settings.database_url.replace("sqlite:///", ""))
    set_db_connection(db_connection)

    # Initialize database schema
    await db_connection.execute_schema()
    logger.info("Database initialized")

    # Initialize vector store
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host, port=settings.qdrant_port, api_key=settings.qdrant_api_key
    )
    set_vector_store(vector_store)
    logger.info("Vector store initialized")

    # Warm up models
    from ..embedding.onnx_encoder import get_encoder

    encoder = get_encoder()
    encoder.warmup()
    logger.info("Models warmed up")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down API...")

    # Close connections
    await db_connection.close()
    await vector_store.close()

    # Clean up dimension analyzer
    from ..extraction.dimensions.dimension_analyzer import get_dimension_analyzer

    analyzer = get_dimension_analyzer()
    analyzer.close()

    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title="Cognitive Meeting Intelligence API",
        description=(
            "Advanced meeting intelligence system with cognitive memory processing, "
            "activation spreading, and intelligent search capabilities."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(memories.router, prefix="/api/v2", tags=["memories"])
    app.include_router(cognitive.router, prefix="/api/v2", tags=["cognitive"])

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Cognitive Meeting Intelligence API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "cognitive_search": "/api/v2/cognitive/query",
                "ingest": "/api/v2/memories/ingest",
                "search": "/api/v2/memories/search",
            },
        }

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """
        Health check endpoint for monitoring.

        Returns:
            Health status and component states
        """
        health_status = {"status": "healthy", "components": {}}

        # Check database
        try:
            await db_connection.execute_query("SELECT 1")
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check vector store
        try:
            stats = await vector_store.get_collection_stats(2)  # Check L2
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "vectors": stats.get("vectors_count", 0),
            }
        except Exception as e:
            health_status["components"]["vector_store"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check encoder
        try:
            from ..embedding.onnx_encoder import get_encoder

            encoder = get_encoder()
            perf_stats = encoder.get_performance_stats()
            health_status["components"]["encoder"] = {
                "status": "healthy",
                "avg_time_ms": perf_stats.get("avg_encoding_time_ms", 0),
            }
        except Exception as e:
            health_status["components"]["encoder"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        status_code = (
            status.HTTP_200_OK
            if health_status["status"] == "healthy"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return JSONResponse(status_code=status_code, content=health_status)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed messages."""
        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "field": " -> ".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                }
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error", "errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with consistent format."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        # Don't expose internal errors in production
        settings = get_settings()
        if settings.environment == "production":
            detail = "An internal error occurred"
        else:
            detail = str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": detail, "status_code": 500},
        )

    # Middleware for request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests for monitoring."""
        import time

        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.0f}ms"
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)

        return response

    return app


# Create app instance
app = create_app()


# Export for uvicorn
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
