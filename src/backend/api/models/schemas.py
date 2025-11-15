"""
Pydantic models for request validation and response serialization
Defines the contract between frontend and backend API
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from uuid import UUID


# ========================================
# Filter Response Models
# ========================================

class DimensionOption(BaseModel):
    """
    Single option within a dimension
    Example: {"value": "Technical", "label": "Technical", "count": 45}
    """
    value: str = Field(..., description="Internal value for filtering")
    label: str = Field(..., description="Display label for frontend")
    count: int = Field(
        0, description="Number of entities with this dimension value")


class DimensionGroup(BaseModel):
    """
    Group of related options within a dimension
    Example: Type dimension with options [Technical, Theoretical, Practical]
    """
    dimension: str = Field(...,
                           description="Dimension identifier (e.g., 'type', 'level')")
    label: str = Field(..., description="Human-readable dimension name")
    options: List[DimensionOption] = Field(default_factory=list)


class EntityFilters(BaseModel):
    """
    All available dimensions for a specific entity type
    """
    dimensions: List[DimensionGroup] = Field(default_factory=list)


class FiltersResponse(BaseModel):
    """
    Complete filter structure for all entity types
    Response for GET /api/filters/all
    """
    knowledge: EntityFilters
    skills: EntityFilters
    abilities: EntityFilters
    functions: EntityFilters
    tasks: EntityFilters

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge": {
                    "dimensions": [
                        {
                            "dimension": "type",
                            "label": "Type",
                            "options": [
                                {"value": "Technical",
                                    "label": "Technical", "count": 45},
                                {"value": "Theoretical",
                                    "label": "Theoretical", "count": 32}
                            ]
                        }
                    ]
                }
            }
        }


# ========================================
# Search Request Models
# ========================================

class SearchRequest(BaseModel):
    """
    Request body for POST /api/search
    """
    query: Optional[str] = Field(
        None,
        description="Occupation search term (e.g., 'Data Scientist')",
        min_length=1,
        max_length=200
    )

    entity_type: Optional[str] = Field(
        None,
        description="Specific entity type to search (knowledge, skills, abilities, functions, tasks)"
    )

    filters: Optional[Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="Dimensional filters as key-value pairs"
    )

    limit: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum number of results to return"
    )

    offset: int = Field(
        0,
        ge=0,
        description="Number of results to skip (for pagination)"
    )

    @field_validator('entity_type')
    @classmethod
    def validate_entity_type(cls, v):
        """Validate that entity_type is one of the allowed values"""
        if v is not None:
            allowed = ['knowledge', 'skills',
                       'abilities', 'functions', 'tasks']
            if v not in allowed:
                raise ValueError(
                    f"entity_type must be one of: {', '.join(allowed)}")
        return v

    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v):
        """Ensure filters is a dictionary with list values"""
        if v is not None:
            for key, value in v.items():
                if not isinstance(value, list):
                    raise ValueError(
                        f"Filter '{key}' must be a list of values")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Data Scientist",
                "entity_type": "knowledge",
                "filters": {
                    "type": ["Technical"],
                    "level": ["Advanced", "Expert"]
                },
                "limit": 50,
                "offset": 0
            }
        }


# ========================================
# Search Response Models
# ========================================

class EntityItem(BaseModel):
    """
    Individual entity result (Knowledge, Skill, Ability, Function, or Task)
    Only includes ID and name as per requirements
    """
    id: UUID = Field(..., description="Unique identifier")
    name: str = Field(..., description="Entity name/title")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Statistical Analysis"
            }
        }


class OccupationInfo(BaseModel):
    """
    Occupation information
    """
    id: UUID = Field(..., description="Occupation UUID")
    title: str = Field(..., description="Occupation title")
    description: Optional[str] = Field(
        None, description="Occupation description")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Data Scientist",
                "description": "Develop and implement algorithms..."
            }
        }


class SearchResponse(BaseModel):
    """
    Response for POST /api/search
    Returns filtered entities for the requested type
    """
    results: List[EntityItem] = Field(default_factory=list)
    total: int = Field(0, description="Total number of matching entities")
    entity_type: str = Field(..., description="Type of entities returned")
    limit: int = Field(..., description="Requested limit")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(
        False, description="Whether more results are available")
    occupation: Optional[OccupationInfo] = Field(
        None, description="Occupation information")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {"id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Statistical Analysis"},
                    {"id": "550e8400-e29b-41d4-a716-446655440001",
                        "name": "Machine Learning"}
                ],
                "total": 47,
                "entity_type": "knowledge",
                "limit": 50,
                "offset": 0,
                "has_more": False
            }
        }


class MultiEntitySearchResponse(BaseModel):
    """
    Response when searching across all entity types
    Used when entity_type is not specified in request
    """
    knowledge: SearchResponse
    skills: SearchResponse
    abilities: SearchResponse
    functions: SearchResponse
    tasks: SearchResponse
    query: Optional[str] = Field(None, description="Original search query")
    occupation: Optional[OccupationInfo] = Field(
        None, description="Occupation information")

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge": {
                    "results": [{"id": "uuid", "name": "Statistical Analysis"}],
                    "total": 12,
                    "entity_type": "knowledge",
                    "limit": 50,
                    "offset": 0,
                    "has_more": False
                },
                "skills": {
                    "results": [],
                    "total": 0,
                    "entity_type": "skills",
                    "limit": 50,
                    "offset": 0,
                    "has_more": False
                },
                "query": "Data Scientist"
            }
        }


# ========================================
# Health Check Models
# ========================================

class HealthResponse(BaseModel):
    """
    Response for health check endpoint
    """
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database connection status")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "database": "connected",
                "version": "1.0.0"
            }
        }


# ========================================
# Error Response Models
# ========================================

class ErrorDetail(BaseModel):
    """
    Detailed error information
    """
    field: Optional[str] = Field(
        None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    type: Optional[str] = Field(None, description="Error type")


class ErrorResponse(BaseModel):
    """
    Standardized error response
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "entity_type",
                        "message": "Must be one of: knowledge, skills, abilities, functions, tasks",
                        "type": "value_error"
                    }
                ]
            }
        }


# ========================================
# Dimension Metadata Models (for internal use)
# ========================================

class DimensionMetadata(BaseModel):
    """
    Internal model for dimension table structure
    Maps dimension types to their scopes and available entities
    """
    dimension_name: str
    table_name: str
    scope: List[str]  # e.g., ['K', 'S', 'A']
    applies_to: List[str]  # e.g., ['knowledge', 'skills', 'abilities']


# Dimension configuration mapping
DIMENSION_CONFIG: Dict[str, DimensionMetadata] = {
    "type": DimensionMetadata(
        dimension_name="type",
        table_name="type_dim",
        scope=["K", "S", "A", "T"],
        applies_to=["knowledge", "skills", "abilities", "tasks"]
    ),
    "level": DimensionMetadata(
        dimension_name="level",
        table_name="level_dim",
        scope=["K", "S", "A"],
        applies_to=["knowledge", "skills", "abilities"]
    ),
    "basis": DimensionMetadata(
        dimension_name="basis",
        table_name="basis_dim",
        scope=["K", "S", "A"],
        applies_to=["knowledge", "skills", "abilities"]
    ),
    "environment": DimensionMetadata(
        dimension_name="environment",
        table_name="environment_dim",
        scope=["F", "T"],
        applies_to=["functions", "tasks"]
    ),
    "mode": DimensionMetadata(
        dimension_name="mode",
        table_name="mode_dim",
        scope=[],  # No scope column in mode_dim table
        applies_to=["tasks"]
    ),
    "physicality": DimensionMetadata(
        dimension_name="physicality",
        table_name="physicality_dim",
        scope=[],  # No scope column in physicality_dim table
        applies_to=["functions"]
    ),
    "cognitive": DimensionMetadata(
        dimension_name="cognitive",
        table_name="cognitive_dim",
        scope=[],  # No scope column in cognitive_dim table
        applies_to=["functions"]
    )
}


# Entity type to scope code mapping
ENTITY_SCOPE_MAP = {
    "knowledge": "K",
    "skills": "S",
    "abilities": "A",
    "functions": "F",
    "tasks": "T"
}
