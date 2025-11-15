"""
Search Service - Handles occupation search and entity filtering
Searches occupations and returns filtered K, S, A, F, T entities
"""

from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple
import logging

from models.schemas import (
    SearchResponse,
    MultiEntitySearchResponse,
    EntityItem,
    OccupationInfo,
    DIMENSION_CONFIG
)
from database import fetch_all, fetch_one

logger = logging.getLogger(__name__)


# ========================================
# Entity Configuration
# ========================================

ENTITY_CONFIG = {
    "knowledge": {
        "table": "knowledge",
        "occupation_junction": "occupation_knowledge",
        "entity_fk": "knowledge_id",
        "dimensions": {
            "type": {"table": "type_dim", "junction": "knowledge_type", "fk": "type_id"},
            "level": {"table": "level_dim", "junction": "knowledge_level", "fk": "level_id"},
            "basis": {"table": "basis_dim", "junction": "knowledge_basis", "fk": "basis_id"}
        }
    },
    "skills": {
        "table": "skill",
        "occupation_junction": "occupation_skill",
        "entity_fk": "skill_id",
        "dimensions": {
            "type": {"table": "type_dim", "junction": "skill_type", "fk": "type_id"},
            "level": {"table": "level_dim", "junction": "skill_level", "fk": "level_id"},
            "basis": {"table": "basis_dim", "junction": "skill_basis", "fk": "basis_id"}
        }
    },
    "abilities": {
        "table": "ability",
        "occupation_junction": "occupation_ability",
        "entity_fk": "ability_id",
        "dimensions": {
            "type": {"table": "type_dim", "junction": "ability_type", "fk": "type_id"},
            "level": {"table": "level_dim", "junction": "ability_level", "fk": "level_id"},
            "basis": {"table": "basis_dim", "junction": "ability_basis", "fk": "basis_id"}
        }
    },
    "functions": {
        "table": "function",
        "occupation_junction": "occupation_function",
        "entity_fk": "function_id",
        "dimensions": {
            "environment": {"table": "environment_dim", "junction": "function_env", "fk": "environment_id"},
            "physicality": {"table": "physicality_dim", "junction": "function_physicality", "fk": "physicality_id"},
            "cognitive": {"table": "cognitive_dim", "junction": "function_cognitive", "fk": "cognitive_id"}
        }
    },
    "tasks": {
        "table": "task",
        "occupation_junction": "occupation_task",
        "entity_fk": "task_id",
        "dimensions": {
            "environment": {"table": "environment_dim", "junction": "task_env", "fk": "environment_id"},
            "type": {"table": "type_dim", "junction": "task_type", "fk": "type_id"},
            "mode": {"table": "mode_dim", "junction": "task_mode", "fk": "mode_id"}
        }
    }
}


# ========================================
# Occupation Search Functions
# ========================================

def find_occupation_by_title(db: Session, search_term: str) -> Optional[Dict]:
    """
    Find occupation by title using fuzzy matching

    Args:
        db: Database session
        search_term: Occupation search term

    Returns:
        Occupation dict with id, title, and description, or None if not found
    """
    query = """
        SELECT id, title, description
        FROM occupation
        WHERE title ILIKE :search_term
        ORDER BY 
            CASE 
                WHEN title ILIKE :exact THEN 1
                WHEN title ILIKE :starts THEN 2
                ELSE 3
            END,
            title
        LIMIT 1
    """

    params = {
        "search_term": f"%{search_term}%",
        "exact": search_term,
        "starts": f"{search_term}%"
    }

    result = fetch_one(db, query, params)

    if result:
        logger.info(f"Found occupation: {result['title']}")
    else:
        logger.warning(f"No occupation found for: {search_term}")

    return result


def get_all_occupations(db: Session, limit: int = 100) -> List[Dict]:
    """
    Get all occupations (for when no search term provided)

    Args:
        db: Database session
        limit: Maximum number of occupations to return

    Returns:
        List of occupation dicts
    """
    query = """
        SELECT id, title, description
        FROM occupation
        ORDER BY title
        LIMIT :limit
    """

    return fetch_all(db, query, {"limit": limit})


# ========================================
# Filter Query Building
# ========================================

def build_dimension_filter_clause(
    entity_type: str,
    filters: Dict[str, List[str]],
    entity_alias: str = "e",
    occupation_junction_alias: str = "oj"
) -> Tuple[str, Dict]:
    """
    Build WHERE clause for dimensional filters

    All dimensions are stored directly in the occupation junction tables:
    - occupation_knowledge: type_id, level_id, basis_id
    - occupation_skill: type_id, level_id, basis_id
    - occupation_ability: type_id, level_id, basis_id
    - occupation_function: environment_id, physicality_id, cognitive_id
    - occupation_task: type_id, environment_id, mode_id

    Args:
        entity_type: Type of entity being filtered
        filters: Dictionary of dimension -> list of values
        entity_alias: SQL alias for entity table
        occupation_junction_alias: SQL alias for occupation junction table

    Returns:
        Tuple of (WHERE clause string, parameters dict)
    """
    if not filters:
        return "", {}

    where_clauses = []
    params = {}
    param_counter = 0

    # Mapping of dimension names to their table names and foreign key columns in occupation junction
    dimension_mappings = {
        "knowledge": {
            "type": {"table": "type_dim", "fk": "type_id"},
            "level": {"table": "level_dim", "fk": "level_id"},
            "basis": {"table": "basis_dim", "fk": "basis_id"}
        },
        "skills": {
            "type": {"table": "type_dim", "fk": "type_id"},
            "level": {"table": "level_dim", "fk": "level_id"},
            "basis": {"table": "basis_dim", "fk": "basis_id"}
        },
        "abilities": {
            "type": {"table": "type_dim", "fk": "type_id"},
            "level": {"table": "level_dim", "fk": "level_id"},
            "basis": {"table": "basis_dim", "fk": "basis_id"}
        },
        "functions": {
            "environment": {"table": "environment_dim", "fk": "environment_id"},
            "physicality": {"table": "physicality_dim", "fk": "physicality_id"},
            "cognitive": {"table": "cognitive_dim", "fk": "cognitive_id"}
        },
        "tasks": {
            "type": {"table": "type_dim", "fk": "type_id"},
            "environment": {"table": "environment_dim", "fk": "environment_id"},
            "mode": {"table": "mode_dim", "fk": "mode_id"}
        }
    }

    # Get dimension mapping for this entity type
    entity_dimensions = dimension_mappings.get(entity_type, {})

    for dim_name, values in filters.items():
        # Check if dimension is applicable to this entity type
        if dim_name not in entity_dimensions:
            logger.warning(
                f"Dimension '{dim_name}' not applicable to '{entity_type}'")
            continue

        dim_info = entity_dimensions[dim_name]
        dim_table = dim_info["table"]
        dim_fk = dim_info["fk"]

        # Create unique parameter names for this dimension's values
        value_params = []
        for value in values:
            param_name = f"dim_value_{param_counter}"
            params[param_name] = value
            value_params.append(f":{param_name}")
            param_counter += 1

        # Build filter clause using the dimension column in occupation junction table
        # The dimension values are directly in the occupation_* junction table
        # Note: Also check that the FK is NOT NULL to avoid matching records without dimensions
        subquery = f"""
            {occupation_junction_alias}.{dim_fk} IS NOT NULL
            AND EXISTS (
                SELECT 1
                FROM {dim_table} d_{dim_name}
                WHERE {occupation_junction_alias}.{dim_fk} = d_{dim_name}.id
                AND d_{dim_name}.name IN ({','.join(value_params)})
            )
        """

        where_clauses.append(subquery)

    if where_clauses:
        return " AND " + " AND ".join(where_clauses), params

    return "", {}


# ========================================
# Entity Search Functions
# ========================================

def search_entities(
    db: Session,
    entity_type: str,
    occupation_ids: List[str],
    filters: Optional[Dict[str, List[str]]] = None,
    limit: int = 50,
    offset: int = 0,
    occupation_info: Optional[Dict] = None
) -> SearchResponse:
    """
    Search entities of a specific type linked to occupation(s)
    Apply dimensional filters if provided

    Args:
        db: Database session
        entity_type: One of 'knowledge', 'skills', 'abilities', 'functions', 'tasks'
        occupation_ids: List of occupation IDs to search
        filters: Optional dimensional filters
        limit: Maximum results to return
        offset: Number of results to skip
        occupation_info: Optional occupation information dict

    Returns:
        SearchResponse with filtered entities and occupation info
    """
    if not occupation_ids:
        return SearchResponse(
            results=[],
            total=0,
            entity_type=entity_type,
            limit=limit,
            offset=offset,
            has_more=False,
            occupation=None
        )

    config = ENTITY_CONFIG[entity_type]
    entity_table = config["table"]
    occupation_junction = config["occupation_junction"]
    entity_fk = config["entity_fk"]

    # Build dimension filter clause
    filter_clause, filter_params = build_dimension_filter_clause(
        entity_type,
        filters or {},
        "e",
        "oj"
    )

    # Debug logging
    if filters:
        logger.info(f"Applied filters: {filters}")
        logger.info(f"Filter clause: {filter_clause}")
        logger.info(f"Filter params: {filter_params}")

    # Build occupation ID parameters
    occ_params = {}
    occ_id_placeholders = []
    for idx, occ_id in enumerate(occupation_ids):
        param_name = f"occ_id_{idx}"
        occ_params[param_name] = occ_id
        occ_id_placeholders.append(f":{param_name}")

    # Combine all parameters
    all_params = {**occ_params, **filter_params,
                  "limit": limit, "offset": offset}

    # Count query
    count_query = f"""
        SELECT COUNT(DISTINCT e.id) as total
        FROM {entity_table} e
        JOIN {occupation_junction} oj ON e.id = oj.{entity_fk}
        WHERE oj.occupation_id IN ({','.join(occ_id_placeholders)})
        {filter_clause}
    """

    # Data query
    data_query = f"""
        SELECT DISTINCT e.id, e.name
        FROM {entity_table} e
        JOIN {occupation_junction} oj ON e.id = oj.{entity_fk}
        WHERE oj.occupation_id IN ({','.join(occ_id_placeholders)})
        {filter_clause}
        ORDER BY e.name
        LIMIT :limit OFFSET :offset
    """

    try:
        # Get total count
        logger.debug(f"Count query: {count_query}")
        logger.debug(f"Query params: {all_params}")
        count_result = fetch_one(db, count_query, all_params)
        total = count_result["total"] if count_result else 0

        # Get results
        logger.debug(f"Data query: {data_query}")
        results = fetch_all(db, data_query, all_params)

        # Convert to EntityItem objects
        entities = [
            EntityItem(id=row["id"], name=row["name"])
            for row in results
        ]

        has_more = (offset + len(entities)) < total

        # Build OccupationInfo if provided
        occ_info = None
        if occupation_info:
            occ_info = OccupationInfo(
                id=occupation_info["id"],
                title=occupation_info["title"],
                description=occupation_info.get("description")
            )

        logger.info(
            f"Found {len(entities)} {entity_type} entities "
            f"(total: {total}, offset: {offset})"
        )

        return SearchResponse(
            results=entities,
            total=total,
            entity_type=entity_type,
            limit=limit,
            offset=offset,
            has_more=has_more,
            occupation=occ_info
        )

    except Exception as e:
        logger.error(f"Error searching {entity_type}: {e}")
        raise


# ========================================
# Main Search Functions
# ========================================

def search_single_entity_type(
    db: Session,
    query: Optional[str],
    entity_type: str,
    filters: Optional[Dict[str, List[str]]] = None,
    limit: int = 50,
    offset: int = 0
) -> SearchResponse:
    """
    Search for a single entity type with optional filters

    Args:
        db: Database session
        query: Occupation search term (optional)
        entity_type: Entity type to search
        filters: Dimensional filters
        limit: Max results
        offset: Result offset

    Returns:
        SearchResponse for the specified entity type with occupation info
    """
    # Find occupation(s)
    occupation = None
    if query:
        occupation = find_occupation_by_title(db, query)
        occupation_ids = [occupation["id"]] if occupation else []
    else:
        # If no query, get all occupations
        occupations = get_all_occupations(db, limit=100)
        occupation_ids = [occ["id"] for occ in occupations]

    # Search entities with filters
    return search_entities(
        db=db,
        entity_type=entity_type,
        occupation_ids=occupation_ids,
        filters=filters,
        limit=limit,
        offset=offset,
        occupation_info=occupation
    )


def search_all_entity_types(
    db: Session,
    query: Optional[str],
    filters: Optional[Dict[str, Dict[str, List[str]]]] = None,
    limit: int = 50,
    offset: int = 0
) -> MultiEntitySearchResponse:
    """
    Search across all entity types with optional per-type filters

    Args:
        db: Database session
        query: Occupation search term (optional)
        filters: Dictionary mapping entity_type -> dimensional filters
            Example: {
                "knowledge": {"type": ["Technical"], "level": ["Advanced"]},
                "skills": {"type": ["Hard Skills"]}
            }
        limit: Max results per entity type
        offset: Result offset

    Returns:
        MultiEntitySearchResponse with results for all entity types and occupation info
    """
    # Find occupation(s)
    occupation = None
    if query:
        occupation = find_occupation_by_title(db, query)
        occupation_ids = [occupation["id"]] if occupation else []
    else:
        # If no query, get all occupations
        occupations = get_all_occupations(db, limit=100)
        occupation_ids = [occ["id"] for occ in occupations]

    # Search each entity type
    entity_types = ["knowledge", "skills", "abilities", "functions", "tasks"]
    results = {}

    for entity_type in entity_types:
        entity_filters = filters.get(entity_type, {}) if filters else {}

        results[entity_type] = search_entities(
            db=db,
            entity_type=entity_type,
            occupation_ids=occupation_ids,
            filters=entity_filters,
            limit=limit,
            offset=offset,
            occupation_info=occupation
        )

    # Build OccupationInfo for the response
    occ_info = None
    if occupation:
        occ_info = OccupationInfo(
            id=occupation["id"],
            title=occupation["title"],
            description=occupation.get("description")
        )

    return MultiEntitySearchResponse(
        knowledge=results["knowledge"],
        skills=results["skills"],
        abilities=results["abilities"],
        functions=results["functions"],
        tasks=results["tasks"],
        query=query,
        occupation=occ_info
    )


# ========================================
# Convenience Functions
# ========================================

def get_entity_by_id(db: Session, entity_type: str, entity_id: str) -> Optional[EntityItem]:
    """
    Get a single entity by ID

    Args:
        db: Database session
        entity_type: Type of entity
        entity_id: Entity UUID

    Returns:
        EntityItem or None if not found
    """
    config = ENTITY_CONFIG[entity_type]
    table = config["table"]

    query = f"""
        SELECT id, name
        FROM {table}
        WHERE id = :entity_id
    """

    result = fetch_one(db, query, {"entity_id": entity_id})

    if result:
        return EntityItem(id=result["id"], name=result["name"])

    return None


def count_entities_by_occupation(db: Session, occupation_id: str) -> Dict[str, int]:
    """
    Count all entities linked to an occupation
    Useful for displaying occupation statistics

    Args:
        db: Database session
        occupation_id: Occupation UUID

    Returns:
        Dictionary with counts for each entity type
    """
    counts = {}

    for entity_type in ["knowledge", "skills", "abilities", "functions", "tasks"]:
        config = ENTITY_CONFIG[entity_type]
        junction_table = config["occupation_junction"]

        query = f"""
            SELECT COUNT(*) as count
            FROM {junction_table}
            WHERE occupation_id = :occupation_id
        """

        result = fetch_one(db, query, {"occupation_id": occupation_id})
        counts[entity_type] = result["count"] if result else 0

    return counts
