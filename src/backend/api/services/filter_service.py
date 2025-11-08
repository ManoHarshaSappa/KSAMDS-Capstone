"""
Filter Service - Dynamically loads dimension data from database
Builds the complete filter structure for the frontend
"""

from sqlalchemy.orm import Session
from typing import Dict, List
import logging

from models.schemas import (
    FiltersResponse,
    EntityFilters,
    DimensionGroup,
    DimensionOption,
    DIMENSION_CONFIG,
    ENTITY_SCOPE_MAP
)
from database import fetch_all

logger = logging.getLogger(__name__)


# ========================================
# Entity Configuration
# ========================================

ENTITY_CONFIG = {
    "knowledge": {
        "table": "knowledge",
        "scope": "K",
        "junction_prefix": "knowledge_"
    },
    "skills": {
        "table": "skill",
        "scope": "S",
        "junction_prefix": "skill_"
    },
    "abilities": {
        "table": "ability",
        "scope": "A",
        "junction_prefix": "ability_"
    },
    "functions": {
        "table": "function",
        "scope": "F",
        "junction_prefix": "function_"
    },
    "tasks": {
        "table": "task",
        "scope": "T",
        "junction_prefix": "task_"
    }
}


# ========================================
# Helper Functions
# ========================================

def get_applicable_dimensions(entity_type: str) -> List[str]:
    """
    Get list of dimension names that apply to a specific entity type

    Args:
        entity_type: One of 'knowledge', 'skills', 'abilities', 'functions', 'tasks'

    Returns:
        List of dimension names (e.g., ['type', 'level', 'basis'])
    """
    entity_scope = ENTITY_CONFIG[entity_type]["scope"]

    applicable = []
    for dim_name, dim_meta in DIMENSION_CONFIG.items():
        if entity_scope in dim_meta.scope:
            applicable.append(dim_name)

    return applicable


def get_junction_table_name(entity_type: str, dimension_name: str) -> str:
    """
    Construct junction table name

    Args:
        entity_type: 'knowledge', 'skills', etc.
        dimension_name: 'type', 'level', etc.

    Returns:
        Junction table name (e.g., 'knowledge_type', 'function_env')

    Special cases:
        - environment -> 'env' suffix
        - physicality/cognitive -> no prefix change
    """
    prefix = ENTITY_CONFIG[entity_type]["junction_prefix"]

    # Handle special naming conventions
    if dimension_name == "environment":
        suffix = "env"
    elif dimension_name == "physicality":
        suffix = "physicality"
    elif dimension_name == "cognitive":
        suffix = "cognitive"
    else:
        suffix = dimension_name

    return f"{prefix}{suffix}"


# ========================================
# Core Query Functions
# ========================================

def get_dimension_options(
    db: Session,
    dimension_name: str,
    entity_type: str
) -> List[DimensionOption]:
    """
    Get all options for a dimension that are actually used by entities
    Only returns dimension values that have been assigned to at least one entity

    Args:
        db: Database session
        dimension_name: Name of dimension (e.g., 'type', 'level')
        entity_type: Entity type (e.g., 'knowledge')

    Returns:
        List of DimensionOption objects
    """
    dim_meta = DIMENSION_CONFIG[dimension_name]
    entity_config = ENTITY_CONFIG[entity_type]
    entity_scope = entity_config["scope"]
    entity_table = entity_config["table"]

    # Get dimension table name
    dim_table = dim_meta.table_name

    # Get junction table name
    junction_table = get_junction_table_name(entity_type, dimension_name)

    # Determine the foreign key column name in the junction table
    if dimension_name == "environment":
        dimension_fk = "env_id"
    else:
        dimension_fk = f"{dimension_name}_id"

    # Special handling for physicality and cognitive (no scope filter)
    if dimension_name in ["physicality", "cognitive"]:
        query = f"""
            SELECT DISTINCT d.id, d.name
            FROM {dim_table} d
            INNER JOIN {junction_table} j ON d.id = j.{dimension_fk}
            ORDER BY d.name
        """
    else:
        # Query only dimension values that are actually used and match the scope
        query = f"""
            SELECT DISTINCT d.id, d.name
            FROM {dim_table} d
            INNER JOIN {junction_table} j ON d.id = j.{dimension_fk}
            WHERE d.scope = :scope
            ORDER BY d.name
        """

    try:
        if dimension_name in ["physicality", "cognitive"]:
            results = fetch_all(db, query)
        else:
            results = fetch_all(db, query, {"scope": entity_scope})

        options = []
        for row in results:
            options.append(DimensionOption(
                value=row["name"],
                label=row["name"],
                count=0  # Not calculated
            ))

        return options

    except Exception as e:
        logger.error(
            f"Error loading dimension '{dimension_name}' for '{entity_type}': {e}")
        return []


def get_entity_filters(db: Session, entity_type: str) -> EntityFilters:
    """
    Get all available filters for a specific entity type

    Args:
        db: Database session
        entity_type: One of 'knowledge', 'skills', 'abilities', 'functions', 'tasks'

    Returns:
        EntityFilters object with all applicable dimensions
    """
    applicable_dims = get_applicable_dimensions(entity_type)

    dimension_groups = []

    for dim_name in applicable_dims:
        # Get dimension metadata
        dim_meta = DIMENSION_CONFIG[dim_name]

        # Create readable label
        label = dim_name.replace("_", " ").title()
        if label == "Environment":
            label = "Environment"
        elif label == "Cognitive":
            label = "Cognitive Load"

        # Get options with counts
        options = get_dimension_options(db, dim_name, entity_type)

        dimension_groups.append(DimensionGroup(
            dimension=dim_name,
            label=label,
            options=options
        ))

    return EntityFilters(dimensions=dimension_groups)


# ========================================
# Main Service Function
# ========================================

def get_all_filters(db: Session) -> FiltersResponse:
    """
    Load all available filters for all entity types
    This is the main function called by the API endpoint

    Args:
        db: Database session

    Returns:
        FiltersResponse with complete filter structure
    """
    logger.info("Loading filters for all entity types...")

    try:
        filters_response = FiltersResponse(
            knowledge=get_entity_filters(db, "knowledge"),
            skills=get_entity_filters(db, "skills"),
            abilities=get_entity_filters(db, "abilities"),
            functions=get_entity_filters(db, "functions"),
            tasks=get_entity_filters(db, "tasks")
        )

        logger.info("✅ Filters loaded successfully")
        return filters_response

    except Exception as e:
        logger.error(f"❌ Error loading filters: {e}")
        raise


# ========================================
# Utility Functions for Validation
# ========================================

def validate_filter_values(
    db: Session,
    entity_type: str,
    filters: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Validate that filter values exist in the database
    Remove invalid values and log warnings

    Args:
        db: Database session
        entity_type: Entity type being filtered
        filters: Dictionary of dimension -> values

    Returns:
        Cleaned filters dictionary with only valid values
    """
    validated = {}

    for dim_name, values in filters.items():
        if dim_name not in DIMENSION_CONFIG:
            logger.warning(f"Unknown dimension '{dim_name}' ignored")
            continue

        # Check if dimension applies to this entity type
        entity_scope = ENTITY_CONFIG[entity_type]["scope"]
        dim_meta = DIMENSION_CONFIG[dim_name]

        if entity_scope not in dim_meta.scope:
            logger.warning(
                f"Dimension '{dim_name}' does not apply to '{entity_type}', ignored"
            )
            continue

        # Get valid values from database (only values actually in use)
        dim_table = dim_meta.table_name
        junction_table = get_junction_table_name(entity_type, dim_name)

        # Determine the foreign key column name in the junction table
        if dim_name == "environment":
            dimension_fk = "env_id"
        else:
            dimension_fk = f"{dim_name}_id"

        if dim_name in ["physicality", "cognitive"]:
            query = f"""
                SELECT DISTINCT d.name 
                FROM {dim_table} d
                INNER JOIN {junction_table} j ON d.id = j.{dimension_fk}
            """
            valid_values = [row["name"] for row in fetch_all(db, query)]
        else:
            query = f"""
                SELECT DISTINCT d.name 
                FROM {dim_table} d
                INNER JOIN {junction_table} j ON d.id = j.{dimension_fk}
                WHERE d.scope = :scope
            """
            valid_values = [
                row["name"]
                for row in fetch_all(db, query, {"scope": entity_scope})
            ]

        # Filter out invalid values
        valid_filter_values = [v for v in values if v in valid_values]

        if valid_filter_values:
            validated[dim_name] = valid_filter_values

        # Log any invalid values
        invalid = set(values) - set(valid_filter_values)
        if invalid:
            logger.warning(
                f"Invalid values for '{dim_name}': {invalid}"
            )

    return validated


def get_filter_summary(filters: Dict[str, List[str]]) -> str:
    """
    Create a human-readable summary of applied filters
    Useful for logging and debugging

    Args:
        filters: Dictionary of dimension -> values

    Returns:
        String summary (e.g., "type:Technical, level:Advanced|Expert")
    """
    if not filters:
        return "No filters"

    parts = []
    for dim, values in filters.items():
        parts.append(f"{dim}:{','.join(values)}")

    return " | ".join(parts)
