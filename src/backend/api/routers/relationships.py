"""
Relationships Router - API endpoints for viewing entity relationships
Shows how K, S, A, F, T entities are connected to each other
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, List
import logging

from database import get_db, fetch_all
from models.schemas import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/entities",
    tags=["relationships"]
)


# Relationship configuration based on database schema
RELATIONSHIP_MAP = {
    "knowledge": {
        "skills": {
            "table": "knowledge_skill",
            "from_fk": "knowledge_id",
            "to_fk": "skill_id",
            "to_table": "skill",
            "header": "Skills that use this Knowledge"
        },
        "functions": {
            "table": "knowledge_function",
            "from_fk": "knowledge_id",
            "to_fk": "function_id",
            "to_table": "function",
            "header": "Functions that require this Knowledge"
        }
    },
    "skills": {
        "knowledge": {
            "table": "knowledge_skill",
            "from_fk": "skill_id",
            "to_fk": "knowledge_id",
            "to_table": "knowledge",
            "header": "Knowledge required for this Skill"
        },
        "abilities": {
            "table": "skill_ability",
            "from_fk": "skill_id",
            "to_fk": "ability_id",
            "to_table": "ability",
            "header": "Abilities needed for this Skill"
        }
    },
    "abilities": {
        "skills": {
            "table": "skill_ability",
            "from_fk": "ability_id",
            "to_fk": "skill_id",
            "to_table": "skill",
            "header": "Skills that develop this Ability"
        },
        "tasks": {
            "table": "ability_task",
            "from_fk": "ability_id",
            "to_fk": "task_id",
            "to_table": "task",
            "header": "Tasks that require this Ability"
        }
    },
    "functions": {
        "knowledge": {
            "table": "knowledge_function",
            "from_fk": "function_id",
            "to_fk": "knowledge_id",
            "to_table": "knowledge",
            "header": "Knowledge required for this Function"
        },
        "tasks": {
            "table": "function_task",
            "from_fk": "function_id",
            "to_fk": "task_id",
            "to_table": "task",
            "header": "Tasks within this Function"
        }
    },
    "tasks": {
        "functions": {
            "table": "function_task",
            "from_fk": "task_id",
            "to_fk": "function_id",
            "to_table": "function",
            "header": "Functions this Task belongs to"
        },
        "abilities": {
            "table": "ability_task",
            "from_fk": "task_id",
            "to_fk": "ability_id",
            "to_table": "ability",
            "header": "Abilities needed for this Task"
        }
    }
}


@router.get(
    "/{entity_type}/{entity_id}/relationships",
    summary="Get entity relationships",
    description="Get all related entities for a specific entity"
)
def get_entity_relationships(
    entity_type: str,
    entity_id: str,
    db: Session = Depends(get_db)
):
    """
    GET /api/entities/{entity_type}/{entity_id}/relationships

    Returns all entities related to the specified entity.

    Args:
        entity_type: One of 'knowledge', 'skills', 'abilities', 'functions', 'tasks'
        entity_id: UUID of the entity

    Returns:
        Dictionary with related entities grouped by type

    Example:
        GET /api/entities/skills/uuid-123/relationships

        Returns:
        {
            "entity": {"id": "uuid-123", "name": "Coding", "type": "skills"},
            "relationships": {
                "knowledge": {
                    "header": "Knowledge required for this Skill",
                    "items": [
                        {"id": "uuid-456", "name": "Programming Languages"},
                        {"id": "uuid-789", "name": "Data Structures"}
                    ],
                    "count": 2
                },
                "abilities": {
                    "header": "Abilities needed for this Skill",
                    "items": [...],
                    "count": 3
                }
            }
        }
    """
    # Validate entity type
    valid_types = ['knowledge', 'skills', 'abilities', 'functions', 'tasks']
    if entity_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": f"Invalid entity_type. Must be one of: {', '.join(valid_types)}"
            }
        )

    try:
        logger.info(
            f"📥 GET /api/entities/{entity_type}/{entity_id}/relationships")

        # Get the entity itself
        entity_table_map = {
            "knowledge": "knowledge",
            "skills": "skill",
            "abilities": "ability",
            "functions": "function",
            "tasks": "task"
        }

        entity_table = entity_table_map[entity_type]

        entity_query = f"""
            SELECT id, name
            FROM {entity_table}
            WHERE id = :entity_id
        """

        entity_result = fetch_all(db, entity_query, {"entity_id": entity_id})

        if not entity_result or len(entity_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "NotFound",
                    "message": f"{entity_type.capitalize()} entity not found"
                }
            )

        entity_info = {
            "id": entity_result[0]["id"],
            "name": entity_result[0]["name"],
            "type": entity_type
        }

        # Get all relationships for this entity type
        relationships = {}

        if entity_type in RELATIONSHIP_MAP:
            for related_type, config in RELATIONSHIP_MAP[entity_type].items():
                query = f"""
                    SELECT DISTINCT e.id, e.name
                    FROM {config['to_table']} e
                    JOIN {config['table']} rel ON e.id = rel.{config['to_fk']}
                    WHERE rel.{config['from_fk']} = :entity_id
                    ORDER BY e.name
                """

                related_items = fetch_all(db, query, {"entity_id": entity_id})

                relationships[related_type] = {
                    "header": config["header"],
                    "items": related_items,
                    "count": len(related_items)
                }

        logger.info(f"✅ Found {len(relationships)} relationship types")

        return {
            "entity": entity_info,
            "relationships": relationships
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Error fetching relationships: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "Failed to fetch entity relationships"
            }
        )
