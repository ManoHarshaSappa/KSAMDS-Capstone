"""
Database connection and session management for KSAMDS API
Handles PostgreSQL connection pool and provides database dependencies
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Database Engine Configuration
# ========================================

engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Additional connections when pool is exhausted
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=settings.DEBUG,  # Log all SQL statements in debug mode
    connect_args={
        "options": f"-c search_path={settings.DB_SCHEMA},public"
    }
)

# ========================================
# Session Factory
# ========================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ========================================
# Database Dependencies for FastAPI
# ========================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session
    Automatically handles session lifecycle and cleanup

    Usage in routes:
        @router.get("/endpoint")
        def my_route(db: Session = Depends(get_db)):
            # Use db here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions
    Use this for non-FastAPI contexts (scripts, services)

    Usage:
        with get_db_context() as db:
            result = db.execute(query)
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


# ========================================
# Database Connection Testing
# ========================================

def test_connection() -> bool:
    """
    Test database connectivity
    Returns True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                logger.info("✅ Database connection successful")
                return True
            return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def test_schema() -> bool:
    """
    Verify that the ksamds schema exists and is accessible
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.schemata 
                    WHERE schema_name = :schema
                )
            """), {"schema": settings.DB_SCHEMA})
            exists = result.scalar()

            if exists:
                logger.info(f"✅ Schema '{settings.DB_SCHEMA}' found")
                return True
            else:
                logger.error(f"❌ Schema '{settings.DB_SCHEMA}' not found")
                return False
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")
        return False


# ========================================
# Query Execution Helpers
# ========================================

def execute_query(db: Session, query: str, params: dict = None):
    """
    Execute a raw SQL query with optional parameters

    Args:
        db: SQLAlchemy session
        query: SQL query string
        params: Dictionary of query parameters

    Returns:
        Result proxy object
    """
    try:
        if params:
            return db.execute(text(query), params)
        else:
            return db.execute(text(query))
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise


def fetch_all(db: Session, query: str, params: dict = None) -> list[dict]:
    """
    Execute query and return all results as list of dictionaries

    Args:
        db: SQLAlchemy session
        query: SQL query string
        params: Dictionary of query parameters

    Returns:
        List of dictionaries (one per row)
    """
    result = execute_query(db, query, params)
    columns = result.keys()
    return [dict(zip(columns, row)) for row in result.fetchall()]


def fetch_one(db: Session, query: str, params: dict = None) -> dict | None:
    """
    Execute query and return first result as dictionary

    Args:
        db: SQLAlchemy session
        query: SQL query string
        params: Dictionary of query parameters

    Returns:
        Dictionary representing the row, or None if no results
    """
    result = execute_query(db, query, params)
    row = result.fetchone()
    if row:
        columns = result.keys()
        return dict(zip(columns, row))
    return None


# ========================================
# Database Initialization
# ========================================

def init_db():
    """
    Initialize database connection and verify connectivity
    Call this on application startup
    """
    logger.info("Initializing database connection...")

    if not test_connection():
        raise Exception("Failed to connect to database")

    if not test_schema():
        raise Exception(f"Schema '{settings.DB_SCHEMA}' not accessible")

    logger.info("Database initialized successfully")


def close_db():
    """
    Close all database connections
    Call this on application shutdown
    """
    logger.info("Closing database connections...")
    engine.dispose()
    logger.info("Database connections closed")


# ========================================
# Connection Pool Stats (for monitoring)
# ========================================

def get_pool_stats() -> dict:
    """
    Get current connection pool statistics
    Useful for monitoring and debugging
    """
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow()
    }
