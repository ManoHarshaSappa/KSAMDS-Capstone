"""
Search Router - API endpoints for searching and filtering entities
Handles occupation search and dimensional filtering for K, S, A, F, T entities
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Union
import logging

from database import get_db
from models.schemas import (
    SearchRequest,
    SearchResponse,
    MultiEntitySearchResponse,
    ErrorResponse
)
from services.search_service import (
    search_single_entity_type,
    search_all_entity_types,
    find_occupation_by_title,
    count_entities_by_occupation
)
from services.filter_service import validate_filter_values, get_filter_summary

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(
    prefix="/search",
    tags=["search"]
)


# ========================================
# Main Search Endpoint
# ========================================

@router.post(
    "",
    response_model=Union[SearchResponse, MultiEntitySearchResponse],
    summary="Search entities with filters",
    description="""
    Search for Knowledge, Skills, Abilities, Functions, and Tasks entities
    based on occupation and dimensional filters.
    
    **Search Modes:**
    
    1. **Single Entity Type** - Specify `entity_type` to search one type:
       - Returns SearchResponse with results for that entity type only
       
    2. **All Entity Types** - Omit `entity_type` to search all:
       - Returns MultiEntitySearchResponse with results for all 5 types
    
    **Filtering:**
    
    - Filters are applied as dimension -> list of values
    - Multiple values in same dimension = OR logic
    - Multiple dimensions = AND logic
    - Example: `{"type": ["Technical", "Practical"], "level": ["Advanced"]}`
      means: (type=Technical OR type=Practical) AND level=Advanced
    
    **Occupation Search:**
    
    - Query uses fuzzy matching on occupation titles
    - If no query provided, returns entities from all occupations
    - Example queries: "Data Scientist", "Software Engineer", "Nurse"
    """,
    responses={
        200: {
            "description": "Successfully retrieved search results"
        },
        400: {
            "description": "Invalid request parameters",
            "model": ErrorResponse
        },
        404: {
            "description": "No occupation found matching query",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
def search_entities(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    POST /api/search

    Search entities with optional filters.

    Args:
        request: SearchRequest body containing:
            - query: Occupation search term (optional)
            - entity_type: Specific entity type or None for all (optional)
            - filters: Dimensional filters (optional)
            - limit: Max results per entity type (default: 50)
            - offset: Pagination offset (default: 0)

    Returns:
        SearchResponse or MultiEntitySearchResponse depending on entity_type

    Example Request (Single Entity Type):
        {
            "query": "Data Scientist",
            "entity_type": "knowledge",
            "filters": {
                "type": ["Technical"],
                "level": ["Advanced", "Expert"]
            },
            "limit": 50,
            "offset": 0
        }

    Example Request (All Entity Types):
        {
            "query": "Data Scientist",
            "filters": {
                "type": ["Technical"],
                "level": ["Advanced"]
            }
        }
    """
    try:
        # Log incoming request
        logger.info(
            f"📥 POST /api/search - Query: '{request.query or 'ALL'}', "
            f"Entity: {request.entity_type or 'ALL'}, "
            f"Filters: {get_filter_summary(request.filters)}"
        )

        # If query provided but no occupation found, return 404
        if request.query:
            occupation = find_occupation_by_title(db, request.query)
            if not occupation:
                logger.warning(
                    f"❌ No occupation found for query: {request.query}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "NotFound",
                        "message": f"No occupation found matching '{request.query}'",
                        "query": request.query
                    }
                )

        # Single entity type search
        if request.entity_type:
            # Validate filters for this entity type
            validated_filters = validate_filter_values(
                db,
                request.entity_type,
                request.filters
            )

            result = search_single_entity_type(
                db=db,
                query=request.query,
                entity_type=request.entity_type,
                filters=validated_filters,
                limit=request.limit,
                offset=request.offset
            )

            logger.info(
                f"✅ POST /api/search - Found {result.total} {request.entity_type} "
                f"({len(result.results)} returned)"
            )

            return result

        # All entity types search
        else:
            # Validate filters for each entity type
            validated_filters_by_type = {}
            for entity_type in ["knowledge", "skills", "abilities", "functions", "tasks"]:
                entity_filters = request.filters.copy() if request.filters else {}
                validated = validate_filter_values(
                    db, entity_type, entity_filters)
                if validated:
                    validated_filters_by_type[entity_type] = validated

            result = search_all_entity_types(
                db=db,
                query=request.query,
                filters=validated_filters_by_type,
                limit=request.limit,
                offset=request.offset
            )

            # Log summary
            totals = {
                "knowledge": result.knowledge.total,
                "skills": result.skills.total,
                "abilities": result.abilities.total,
                "functions": result.functions.total,
                "tasks": result.tasks.total
            }
            logger.info(
                f"✅ POST /api/search - Found entities: "
                f"K={totals['knowledge']}, S={totals['skills']}, "
                f"A={totals['abilities']}, F={totals['functions']}, "
                f"T={totals['tasks']}"
            )

            return result

    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise

    except Exception as e:
        logger.error(f"❌ POST /api/search - Error: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "An error occurred while processing the search",
                "details": str(e) if logger.level == logging.DEBUG else None
            }
        )


# ========================================
# Convenience Endpoints
# ========================================

@router.get(
    "/occupation/{occupation_title}",
    summary="Get occupation details",
    description="Find an occupation by title and return its entity counts"
)
def get_occupation_info(
    occupation_title: str,
    db: Session = Depends(get_db)
):
    """
    GET /api/search/occupation/{occupation_title}

    Get information about a specific occupation including entity counts.

    Args:
        occupation_title: Occupation title to search for

    Returns:
        dict: Occupation info with entity counts

    Example:
        GET /api/search/occupation/Data Scientist

        Returns:
        {
            "id": "uuid",
            "title": "Data Scientist",
            "entity_counts": {
                "knowledge": 45,
                "skills": 32,
                "abilities": 28,
                "functions": 15,
                "tasks": 40
            }
        }
    """
    try:
        logger.info(f"📥 GET /api/search/occupation/{occupation_title}")

        # Find occupation
        occupation = find_occupation_by_title(db, occupation_title)

        if not occupation:
            logger.warning(f"❌ Occupation not found: {occupation_title}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "NotFound",
                    "message": f"Occupation '{occupation_title}' not found"
                }
            )

        # Get entity counts
        counts = count_entities_by_occupation(db, occupation["id"])

        logger.info(f"✅ GET /api/search/occupation/{occupation_title} - Found")

        return {
            "id": occupation["id"],
            "title": occupation["title"],
            "entity_counts": counts
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"❌ GET /api/search/occupation - Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve occupation information"
            }
        )


@router.get(
    "/occupations",
    summary="List all occupations",
    description="Get a list of all available occupations"
)
def list_occupations(
    limit: int = Query(100, ge=1, le=500,
                       description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    search: Optional[str] = Query(
        None, description="Filter occupations by title"),
    db: Session = Depends(get_db)
):
    """
    GET /api/search/occupations

    List all occupations with optional search filter.

    Args:
        limit: Maximum results to return (default: 100, max: 500)
        offset: Number of results to skip for pagination
        search: Optional search term to filter occupation titles

    Returns:
        dict: List of occupations with pagination info

    Example:
        GET /api/search/occupations?limit=20&search=Engineer

        Returns:
        {
            "occupations": [
                {"id": "uuid", "title": "Software Engineer"},
                {"id": "uuid", "title": "Mechanical Engineer"}
            ],
            "total": 47,
            "limit": 20,
            "offset": 0
        }
    """
    try:
        from database import fetch_all, fetch_one

        logger.info(
            f"📥 GET /api/search/occupations - "
            f"Search: '{search or 'ALL'}', Limit: {limit}, Offset: {offset}"
        )

        # Build query based on search parameter
        if search:
            count_query = """
                SELECT COUNT(*) as total
                FROM occupation
                WHERE title ILIKE :search
            """
            data_query = """
                SELECT id, title
                FROM occupation
                WHERE title ILIKE :search
                ORDER BY title
                LIMIT :limit OFFSET :offset
            """
            params = {"search": f"%{search}%",
                      "limit": limit, "offset": offset}
        else:
            count_query = """
                SELECT COUNT(*) as total
                FROM occupation
            """
            data_query = """
                SELECT id, title
                FROM occupation
                ORDER BY title
                LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}

        # Get total count
        count_result = fetch_one(db, count_query, params if search else None)
        total = count_result["total"] if count_result else 0

        # Get occupations
        occupations = fetch_all(db, data_query, params)

        logger.info(
            f"✅ GET /api/search/occupations - "
            f"Found {len(occupations)} of {total} occupations"
        )

        return {
            "occupations": occupations,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(occupations)) < total
        }

    except Exception as e:
        logger.error(
            f"❌ GET /api/search/occupations - Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve occupations list"
            }
        )


@router.get(
    "/health",
    summary="Search service health check",
    description="Check if the search service is operational"
)
def search_health_check(db: Session = Depends(get_db)):
    """
    GET /api/search/health

    Simple health check for the search service.
    Verifies database connectivity and search capability.

    Returns:
        dict: Health status information
    """
    try:
        # Test database connection with a simple query
        from database import fetch_one
        result = fetch_one(db, "SELECT COUNT(*) as count FROM occupation")

        if result:
            return {
                "status": "healthy",
                "service": "search",
                "database": "connected",
                "occupation_count": result["count"]
            }
        else:
            raise Exception("Database query failed")

    except Exception as e:
        logger.error(f"❌ Search service health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "search",
                "database": "disconnected",
                "error": str(e)
            }
        )

# ========================================
# Autocomplete Suggestions Endpoint
# ========================================


@router.get(
    "/suggest",
    summary="Get occupation suggestions",
    description="Autocomplete suggestions for occupation titles"
)
def get_occupation_suggestions(
    q: str = Query(..., min_length=1, max_length=100,
                   description="Search query"),
    limit: int = Query(
        10, ge=1, le=20, description="Maximum suggestions to return"),
    db: Session = Depends(get_db)
):
    """
    GET /api/search/suggest?q={query}

    Get autocomplete suggestions for occupation titles.

    Args:
        q: Search query string
        limit: Maximum number of suggestions (default: 10, max: 20)

    Returns:
        List of occupation suggestions

    Example:
        GET /api/search/suggest?q=data

        Returns:
        {
            "suggestions": [
                {"id": "uuid", "title": "Data Scientist"},
                {"id": "uuid", "title": "Data Engineer"},
                {"id": "uuid", "title": "Database Administrator"}
            ],
            "query": "data"
        }
    """
    try:
        from database import fetch_all

        logger.info(f"📥 GET /api/search/suggest?q={q}")

        # Search for occupations matching the query
        query = """
            SELECT id, title
            FROM occupation
            WHERE title ILIKE :search_term
            ORDER BY 
                CASE 
                    WHEN title ILIKE :starts THEN 1
                    WHEN title ILIKE :contains THEN 2
                    ELSE 3
                END,
                title
            LIMIT :limit
        """

        params = {
            "search_term": f"%{q}%",
            "starts": f"{q}%",
            "contains": f"%{q}%",
            "limit": limit
        }

        suggestions = fetch_all(db, query, params)

        logger.info(f"✅ Found {len(suggestions)} suggestions for '{q}'")

        return {
            "suggestions": suggestions,
            "query": q,
            "count": len(suggestions)
        }

    except Exception as e:
        logger.error(f"❌ GET /api/search/suggest - Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve suggestions"
            }
        )
