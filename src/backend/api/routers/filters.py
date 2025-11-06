"""
Filters Router - API endpoints for loading dimensional filters
Provides filter options for the frontend dropdown menus
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from database import get_db
from models.schemas import FiltersResponse, ErrorResponse
from services.filter_service import get_all_filters

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(
    prefix="/filters",
    tags=["filters"]
)


# ========================================
# Endpoints
# ========================================

@router.get(
    "/all",
    response_model=FiltersResponse,
    summary="Get all available filters",
    description="""
    Returns all available dimensional filters for all entity types.
    This endpoint loads filter options dynamically from the database,
    including counts of how many entities have each dimension value.
    
    Used by the frontend to populate filter dropdown menus on page load.
    
    Returns filter dimensions for:
    - Knowledge (type, level, basis)
    - Skills (type, level, basis)
    - Abilities (type, level, basis)
    - Functions (environment, physicality, cognitive)
    - Tasks (environment, type, mode)
    """,
    responses={
        200: {
            "description": "Successfully retrieved filters",
            "model": FiltersResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
def get_filters(db: Session = Depends(get_db)):
    """
    GET /api/filters/all

    Load all dimensional filters with entity counts.
    No parameters required.

    Returns:
        FiltersResponse: Complete filter structure for all entity types

    Example Response:
        {
            "knowledge": {
                "dimensions": [
                    {
                        "dimension": "type",
                        "label": "Type",
                        "options": [
                            {"value": "Technical", "label": "Technical", "count": 45},
                            {"value": "Theoretical", "label": "Theoretical", "count": 32}
                        ]
                    },
                    {
                        "dimension": "level",
                        "label": "Level",
                        "options": [...]
                    }
                ]
            },
            "skills": {...},
            "abilities": {...},
            "functions": {...},
            "tasks": {...}
        }
    """
    try:
        logger.info("📥 GET /api/filters/all - Loading all filters")

        # Load filters from database
        filters = get_all_filters(db)

        logger.info("✅ GET /api/filters/all - Filters loaded successfully")

        return filters

    except Exception as e:
        logger.error(f"❌ GET /api/filters/all - Error: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "Failed to load filters from database",
                "details": str(e) if logger.level == logging.DEBUG else None
            }
        )


@router.get(
    "/entity/{entity_type}",
    response_model=dict,
    summary="Get filters for specific entity type",
    description="""
    Returns dimensional filters for a specific entity type only.
    Useful when you only need filters for one entity (knowledge, skills, etc).
    """,
    responses={
        200: {
            "description": "Successfully retrieved entity filters"
        },
        400: {
            "description": "Invalid entity type",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
def get_entity_filters(
    entity_type: str,
    db: Session = Depends(get_db)
):
    """
    GET /api/filters/entity/{entity_type}

    Load filters for a specific entity type.

    Args:
        entity_type: One of 'knowledge', 'skills', 'abilities', 'functions', 'tasks'

    Returns:
        EntityFilters: Dimensions for the specified entity type

    Example:
        GET /api/filters/entity/knowledge

        Returns:
        {
            "dimensions": [
                {
                    "dimension": "type",
                    "label": "Type",
                    "options": [...]
                }
            ]
        }
    """
    # Validate entity type
    valid_types = ['knowledge', 'skills', 'abilities', 'functions', 'tasks']
    if entity_type not in valid_types:
        logger.warning(f"❌ Invalid entity_type requested: {entity_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": f"Invalid entity_type. Must be one of: {', '.join(valid_types)}",
                "details": [
                    {
                        "field": "entity_type",
                        "message": f"Invalid value: {entity_type}",
                        "type": "value_error"
                    }
                ]
            }
        )

    try:
        logger.info(
            f"📥 GET /api/filters/entity/{entity_type} - Loading filters")

        # Load all filters and extract the requested entity type
        all_filters = get_all_filters(db)
        entity_filters = getattr(all_filters, entity_type)

        logger.info(
            f"✅ GET /api/filters/entity/{entity_type} - Filters loaded")

        return {"dimensions": entity_filters.dimensions}

    except Exception as e:
        logger.error(
            f"❌ GET /api/filters/entity/{entity_type} - Error: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to load filters for {entity_type}",
                "details": str(e) if logger.level == logging.DEBUG else None
            }
        )


@router.get(
    "/health",
    summary="Filter service health check",
    description="Check if the filter service is operational"
)
def filters_health_check(db: Session = Depends(get_db)):
    """
    GET /api/filters/health

    Simple health check for the filters service.
    Verifies database connectivity and filter loading capability.

    Returns:
        dict: Health status information
    """
    try:
        # Try to load a simple query to verify DB connection
        from database import fetch_one
        result = fetch_one(db, "SELECT 1 as test")

        if result and result["test"] == 1:
            return {
                "status": "healthy",
                "service": "filters",
                "database": "connected"
            }
        else:
            raise Exception("Database query returned unexpected result")

    except Exception as e:
        logger.error(f"❌ Filter service health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "filters",
                "database": "disconnected",
                "error": str(e)
            }
        )
