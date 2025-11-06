"""
KSAMDS API - Main Application
FastAPI backend for Knowledge, Skills, Abilities, Functions, and Tasks explorer
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time

from config import settings
from database import init_db, close_db, test_connection, test_schema
from routers import filters, search, relationships

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ========================================
# Lifespan Context Manager
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Handles database connection initialization and cleanup
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    try:
        # Initialize database
        init_db()
        logger.info("✅ Database initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

    logger.info(f"🌐 Server running at: http://localhost:8000")
    logger.info(f"📚 API Documentation: http://localhost:8000/docs")
    logger.info(f"🔍 Alternative docs: http://localhost:8000/redoc")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("=" * 60)
    logger.info("🛑 Shutting down application...")
    close_db()
    logger.info("✅ Database connections closed")
    logger.info("=" * 60)


# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## KSAMDS Occupation Explorer API
    
    Backend API for exploring occupational data including:
    - **Knowledge** - Areas of expertise and understanding
    - **Skills** - Practical abilities and techniques
    - **Abilities** - Innate and developed capabilities
    - **Functions** - Job-specific activities and responsibilities
    - **Tasks** - Specific work activities and duties
    
    ### Features
    
    - 🔍 **Dynamic Filtering** - Filter by multiple dimensions (type, level, basis, etc.)
    - 🎯 **Occupation Search** - Find entities by occupation title
    - 📊 **Entity Counts** - See how many entities match each filter
    - 🔄 **Pagination** - Efficient data retrieval with limit/offset
    - ✅ **Validation** - Request validation with helpful error messages
    
    ### Endpoints
    
    - `GET /api/filters/all` - Load all available filters
    - `POST /api/search` - Search and filter entities
    - `GET /api/search/occupations` - List all occupations
    - `GET /health` - Service health check
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ========================================
# CORS Configuration
# ========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logger.info(f"✅ CORS enabled for origins: {settings.CORS_ORIGINS}")


# ========================================
# Request Logging Middleware
# ========================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests and their processing time
    """
    start_time = time.time()

    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log response
    logger.info(
        f"📤 {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    # Add custom header with processing time
    response.headers["X-Process-Time"] = str(process_time)

    return response


# ========================================
# Exception Handlers
# ========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for Pydantic validation errors
    Returns user-friendly error messages
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(f"❌ Validation error on {request.url.path}: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request parameters",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unexpected errors
    """
    logger.error(
        f"❌ Unhandled exception on {request.url.path}: {exc}",
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.DEBUG else None
        }
    )


# ========================================
# Include Routers
# ========================================

app.include_router(
    filters.router,
    prefix=settings.API_PREFIX,
    tags=["filters"]
)

app.include_router(
    search.router,
    prefix=settings.API_PREFIX,
    tags=["search"]
)

app.include_router(
    relationships.router,  # ADD THIS
    prefix=settings.API_PREFIX,
    tags=["relationships"]
)

logger.info("✅ Routers registered: /api/filters, /api/search")


# ========================================
# Root Endpoints
# ========================================

@app.get(
    "/",
    tags=["root"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
def root():
    """
    GET /

    Root endpoint providing API information and links
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "filters": f"{settings.API_PREFIX}/filters/all",
            "search": f"{settings.API_PREFIX}/search",
            "occupations": f"{settings.API_PREFIX}/search/occupations",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    tags=["health"],
    summary="Service Health Check",
    description="Check overall service health including database connectivity"
)
def health_check():
    """
    GET /health

    Complete health check for the API service
    Tests database connectivity and returns service status
    """
    try:
        # Test database connection
        db_healthy = test_connection() and test_schema()

        if db_healthy:
            return {
                "status": "healthy",
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "database": "connected",
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "service": settings.APP_NAME,
                    "version": settings.APP_VERSION,
                    "database": "disconnected",
                    "timestamp": time.time()
                }
            )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "database": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@app.get(
    "/version",
    tags=["root"],
    summary="API Version",
    description="Get API version information"
)
def version():
    """
    GET /version

    Returns current API version
    """
    return {
        "version": settings.APP_VERSION,
        "name": settings.APP_NAME
    }


# ========================================
# Run Application
# ========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
