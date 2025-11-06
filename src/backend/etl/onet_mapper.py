"""
O*NET Data Mapper for KSAMDS Project - DETERMINISTIC VERSION WITH IM FILTERING
Updated to properly handle multiple proficiency levels per entity

This module handles the complex mapping and transformation of O*NET data
into the KSAMDS multi-dimensional structure. Uses deterministic UUIDs to
ensure idempotent pipeline execution.

IMPORTANT: Only maps Knowledge, Skills, Abilities, and Functions that have
Scale ID = 'IM' (Importance) and Data Value >= 3.0 to ensure relevance.

UPDATED: Now properly extracts level (LV) data from O*NET and maps it to
KSAMDS level dimensions, tracking both entity levels and occupation requirements.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import re
from dataclasses import dataclass, field
from fuzzywuzzy import fuzz  # type: ignore
from collections import defaultdict
import uuid
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# UUID namespace for KSAMDS (fixed for deterministic UUID generation)
KSAMDS_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def generate_deterministic_uuid(entity_type: str, name: str) -> str:
    """
    Generate deterministic UUID based on entity type and name.

    Uses UUID v5 (name-based) to ensure the same entity always gets
    the same UUID across multiple pipeline runs.

    Args:
        entity_type: Type of entity (e.g., 'knowledge', 'skill', 'occupation')
        name: Name of the entity (must be unique within entity type)

    Returns:
        str: Deterministic UUID as string
    """
    # Create a unique identifier by combining entity type and name
    unique_identifier = f"{entity_type.lower()}:{name.strip()}"

    # Generate UUID v5 using the KSAMDS namespace
    return str(uuid.uuid5(KSAMDS_NAMESPACE, unique_identifier))


@dataclass
class KSAMDSEntity:
    """Represents a mapped KSAMDS entity with its dimensions."""
    id: str
    name: str
    source_ref: str
    # Dimensions (optional)
    type_dims: List[str] = field(default_factory=list)
    # Now supports multiple levels
    level_dims: List[str] = field(default_factory=list)
    basis_dims: List[str] = field(default_factory=list)
    environment_dims: List[str] = field(default_factory=list)
    mode_dims: List[str] = field(default_factory=list)
    physicality_dims: List[str] = field(default_factory=list)
    cognitive_dims: List[str] = field(default_factory=list)


@dataclass
class OccupationRelationship:
    """Represents an occupation relationship with level and importance tracking."""
    occupation_id: str
    entity_id: str
    level: Optional[str] = None
    importance_score: Optional[float] = None


class ONetMapper:
    """Map O*NET data to KSAMDS multi-dimensional structure."""

    def __init__(self):
        """Initialize the O*NET mapper."""
        # Get project root directory (3 levels up from etl folder)
        project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.processed_dir = project_root / "data/archive/intermediate"
        self.mapped_dir = project_root / "data/archive/mapped"

        self.mapped_dir.mkdir(parents=True, exist_ok=True)

        self.onet_data: Dict[str, pd.DataFrame] = {}
        self.ksamds_entities = {
            'knowledge': {}, 'skill': {}, 'ability': {},
            'occupation': {}, 'function': {}, 'task': {},
            'education_level': {}, 'certification': {}
        }
        self.hierarchy_mappings = {}
        self.job_zone_lookup = {}
        self.duplicate_groups = defaultdict(list)
        self.similarity_threshold = 85
        self.onet_to_ksamds_ids = {}
        self.ksamds_relationships = defaultdict(list)

        # NEW: Track entity-level mappings for occupation relationships
        # Format: {(onet_soc, element_id): level_name}
        self.occupation_entity_levels = {}

    def load_onet_data(self) -> bool:
        """Load processed O*NET CSV files."""
        logger.info("=" * 70)
        logger.info("LOADING O*NET DATA")
        logger.info("=" * 70)

        required_files = ['knowledge', 'skills', 'abilities', 'occupation_data',
                          'task_statements', 'work_activities',
                          'content_model_reference', 'job_zones']

        try:
            for file_key in required_files:
                csv_path = self.processed_dir / f"{file_key}.csv"
                if csv_path.exists():
                    self.onet_data[file_key] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {file_key}: {len(self.onet_data[file_key])} records")
                else:
                    logger.error(f"Required file not found: {csv_path}")
                    return False

            optional_files = ['scales_reference',
                              'education_training_experience']
            for file_key in optional_files:
                csv_path = self.processed_dir / f"{file_key}.csv"
                if csv_path.exists():
                    self.onet_data[file_key] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded optional {file_key}: {len(self.onet_data[file_key])} records")

            return True
        except Exception as e:
            logger.error(f"Failed to load O*NET data: {e}")
            return False

    def build_hierarchy_mappings(self) -> bool:
        """Build mappings from O*NET Content Model hierarchy."""
        if 'content_model_reference' not in self.onet_data:
            logger.error("Content Model Reference not loaded")
            return False

        logger.info("Building O*NET hierarchy mappings...")

        self.hierarchy_mappings = {
            '1.A.1': 'Cognitive', '1.A.2': 'Physical', '1.A.3': 'Physical', '1.A.4': 'Sensory',
            '2.A.1': 'Technical', '2.A.2': 'Technical', '2.B.1': 'Analytical', '2.B.2': 'Social',
            '2.B.3': 'Technical', '2.B.4': 'Management', '2.B.5': 'Management',
            '2.C.1': 'Business', '2.C.2': 'Technical', '2.C.3': 'Technical', '2.C.4': 'Technical',
            '2.C.5': 'Safety', '2.C.6': 'Social', '2.C.7': 'Social', '2.C.8': 'Safety',
            '2.C.9': 'Social', '2.C.10': 'Technical',
            '4.A.1': 'Analytical', '4.A.2': 'Analytical', '4.A.3': 'Social', '4.A.4': 'Social',
        }

        if 'job_zones' in self.onet_data:
            job_zones_df = self.onet_data['job_zones']
            self.job_zone_lookup = dict(
                zip(job_zones_df['O*NET-SOC Code'], job_zones_df['Job Zone']))

        logger.info("Hierarchy mappings built successfully")
        return True

    def get_type_from_hierarchy(self, element_id: str) -> str:
        """Get type dimension from O*NET hierarchy."""
        for hierarchy_prefix, ksamds_type in self.hierarchy_mappings.items():
            if element_id.startswith(hierarchy_prefix):
                return ksamds_type
        return 'General'

    def map_level_value_to_category(self, lv_value: float, scope: str) -> str:
        """
        Map O*NET Level (LV) scale value to KSAMDS level category.

        O*NET uses a 1-7 scale for level values. This maps them to our
        predefined level categories based on scope (K, S, or A).

        Args:
            lv_value: O*NET level value (typically 0-7)
            scope: Entity scope ('K' for Knowledge, 'S' for Skills, 'A' for Abilities)

        Returns:
            str: KSAMDS level category name
        """
        if pd.isna(lv_value):
            # Default to middle level if no data
            if scope == 'K':
                return 'Intermediate'
            elif scope == 'S':
                return 'Proficient'
            else:  # 'A'
                return 'Moderate'

        # Map based on scope
        if scope == 'K':  # Knowledge: Basic, Intermediate, Advanced, Expert
            if lv_value <= 2.0:
                return 'Basic'
            elif lv_value <= 4.0:
                return 'Intermediate'
            elif lv_value <= 6.0:
                return 'Advanced'
            else:
                return 'Expert'

        elif scope == 'S':  # Skills: Novice, Proficient, Expert, Master
            if lv_value <= 2.0:
                return 'Novice'
            elif lv_value <= 4.5:
                return 'Proficient'
            elif lv_value <= 6.0:
                return 'Expert'
            else:
                return 'Master'

        else:  # 'A' - Abilities: Low, Moderate, High
            if lv_value <= 3.0:
                return 'Low'
            elif lv_value <= 5.0:
                return 'Moderate'
            else:
                return 'High'

    def get_level_from_rating(self, lv_score: float, entity_type: str) -> str:
        """
        Legacy method for backward compatibility.
        Now delegates to map_level_value_to_category.
        """
        scope_map = {'knowledge': 'K', 'skill': 'S', 'ability': 'A'}
        scope = scope_map.get(entity_type, 'K')
        return self.map_level_value_to_category(lv_score, scope)

    def get_basis_from_job_zone(self, job_zone: int) -> List[str]:
        """Map job zone to basis dimensions."""
        if job_zone <= 2:
            return ['On-the-Job Training']
        elif job_zone == 3:
            return ['On-the-Job Training', 'Vocational Training']
        elif job_zone == 4:
            return ['Academic']
        else:
            return ['Academic', 'Professional Development']

    def map_similar_entities(self, df: pd.DataFrame, name_col: str = 'Element Name') -> Dict[str, str]:
        """Group similar entities to avoid duplicates."""
        unique_names = df[name_col].unique()
        name_to_canonical = {}
        processed = set()

        for name in unique_names:
            if name in processed:
                continue

            canonical_name = name
            similar_group = [name]

            for other_name in unique_names:
                if other_name != name and other_name not in processed:
                    similarity_score = fuzz.ratio(
                        name.lower(), other_name.lower())
                    if similarity_score >= self.similarity_threshold:
                        similar_group.append(other_name)
                        processed.add(other_name)

            for similar_name in similar_group:
                name_to_canonical[similar_name] = canonical_name
            processed.add(name)

            if len(similar_group) > 1:
                self.duplicate_groups[canonical_name] = similar_group

        return name_to_canonical

    def map_occupation_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET occupation data to KSAMDS occupation entities."""
        logger.info("=" * 70)
        logger.info("MAPPING OCCUPATION ENTITIES")
        logger.info("=" * 70)

        if 'occupation_data' not in self.onet_data:
            logger.error("Occupation data not loaded")
            return {}

        occupations = {}
        df = self.onet_data['occupation_data']

        for _, row in df.iterrows():
            onet_soc = row.get('O*NET-SOC Code', '')
            title = row.get('Title', '')
            description = row.get('Description', '')

            entity_id = generate_deterministic_uuid('occupation', onet_soc)
            self.onet_to_ksamds_ids[onet_soc] = entity_id

            occupation = KSAMDSEntity(
                id=entity_id, name=title, source_ref=onet_soc)
            occupations[entity_id] = occupation

        logger.info(f"Mapped {len(occupations)} occupation entities")
        return occupations

    def map_knowledge_entities(self) -> Dict[str, KSAMDSEntity]:
        """
        Map O*NET knowledge data to KSAMDS knowledge entities.

        UPDATED: Now extracts level data from 'Data Value LV' column and
        tracks occupation-level requirements.
        """
        logger.info("=" * 70)
        logger.info("MAPPING KNOWLEDGE ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'knowledge' not in self.onet_data:
            logger.error("Knowledge data not loaded")
            return {}

        df = self.onet_data['knowledge']

        # Filter for importance (IM) >= 3.0
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]
        logger.info(
            f"Filtered to {len(df_filtered)} knowledge records (IM >= 3.0) from {len(df)} total")

        # Map similar entities to avoid duplicates
        name_to_canonical = self.map_similar_entities(df_filtered)

        knowledge_entities = {}
        level_stats = defaultdict(int)

        for _, row in df_filtered.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            onet_soc = row.get('O*NET-SOC Code', '')
            importance_score = row.get('Data Value', None)

            # Get level value from enriched data
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid(
                'knowledge', canonical_name)

            # Map level value to category
            level_category = self.map_level_value_to_category(lv_value, 'K')
            level_stats[level_category] += 1

            # Store occupation-entity level mapping
            self.occupation_entity_levels[(
                onet_soc, element_id)] = level_category

            if entity_id not in knowledge_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                job_zone = self.job_zone_lookup.get(onet_soc, 3)
                basis_dims = self.get_basis_from_job_zone(job_zone)

                knowledge = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                    basis_dims=basis_dims
                )
                knowledge_entities[entity_id] = knowledge
            else:
                # Add level if not already present
                if level_category not in knowledge_entities[entity_id].level_dims:
                    knowledge_entities[entity_id].level_dims.append(
                        level_category)

            self.onet_to_ksamds_ids[element_id] = entity_id

        logger.info(
            f"Mapped {len(knowledge_entities)} unique knowledge entities")
        logger.info("Level distribution:")
        for level, count in sorted(level_stats.items()):
            logger.info(f"  {level}: {count}")

        return knowledge_entities

    def map_skill_entities(self) -> Dict[str, KSAMDSEntity]:
        """
        Map O*NET skills data to KSAMDS skill entities.

        UPDATED: Now extracts level data from 'Data Value LV' column and
        tracks occupation-level requirements.
        """
        logger.info("=" * 70)
        logger.info("MAPPING SKILL ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'skills' not in self.onet_data:
            logger.error("Skills data not loaded")
            return {}

        df = self.onet_data['skills']

        # Filter for importance (IM) >= 3.0
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]
        logger.info(
            f"Filtered to {len(df_filtered)} skill records (IM >= 3.0) from {len(df)} total")

        name_to_canonical = self.map_similar_entities(df_filtered)

        skill_entities = {}
        level_stats = defaultdict(int)

        for _, row in df_filtered.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            onet_soc = row.get('O*NET-SOC Code', '')
            importance_score = row.get('Data Value', None)

            # Get level value from enriched data
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('skill', canonical_name)

            # Map level value to category
            level_category = self.map_level_value_to_category(lv_value, 'S')
            level_stats[level_category] += 1

            # Store occupation-entity level mapping
            self.occupation_entity_levels[(
                onet_soc, element_id)] = level_category

            if entity_id not in skill_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                job_zone = self.job_zone_lookup.get(onet_soc, 3)
                basis_dims = self.get_basis_from_job_zone(job_zone)

                skill = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                    basis_dims=basis_dims
                )
                skill_entities[entity_id] = skill
            else:
                # Add level if not already present
                if level_category not in skill_entities[entity_id].level_dims:
                    skill_entities[entity_id].level_dims.append(level_category)

            self.onet_to_ksamds_ids[element_id] = entity_id

        logger.info(f"Mapped {len(skill_entities)} unique skill entities")
        logger.info("Level distribution:")
        for level, count in sorted(level_stats.items()):
            logger.info(f"  {level}: {count}")

        return skill_entities

    def map_ability_entities(self) -> Dict[str, KSAMDSEntity]:
        """
        Map O*NET abilities data to KSAMDS ability entities.

        UPDATED: Now extracts level data from 'Data Value LV' column and
        tracks occupation-level requirements.
        """
        logger.info("=" * 70)
        logger.info("MAPPING ABILITY ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'abilities' not in self.onet_data:
            logger.error("Abilities data not loaded")
            return {}

        df = self.onet_data['abilities']

        # Filter for importance (IM) >= 3.0
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]
        logger.info(
            f"Filtered to {len(df_filtered)} ability records (IM >= 3.0) from {len(df)} total")

        name_to_canonical = self.map_similar_entities(df_filtered)

        ability_entities = {}
        level_stats = defaultdict(int)

        for _, row in df_filtered.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            onet_soc = row.get('O*NET-SOC Code', '')
            importance_score = row.get('Data Value', None)

            # Get level value from enriched data
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('ability', canonical_name)

            # Map level value to category
            level_category = self.map_level_value_to_category(lv_value, 'A')
            level_stats[level_category] += 1

            # Store occupation-entity level mapping
            self.occupation_entity_levels[(
                onet_soc, element_id)] = level_category

            if entity_id not in ability_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                job_zone = self.job_zone_lookup.get(onet_soc, 3)
                basis_dims = self.get_basis_from_job_zone(job_zone)

                ability = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                    basis_dims=basis_dims
                )
                ability_entities[entity_id] = ability
            else:
                # Add level if not already present
                if level_category not in ability_entities[entity_id].level_dims:
                    ability_entities[entity_id].level_dims.append(
                        level_category)

            self.onet_to_ksamds_ids[element_id] = entity_id

        logger.info(f"Mapped {len(ability_entities)} unique ability entities")
        logger.info("Level distribution:")
        for level, count in sorted(level_stats.items()):
            logger.info(f"  {level}: {count}")

        return ability_entities

    def map_task_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET task statements to KSAMDS task entities."""
        logger.info("=" * 70)
        logger.info("MAPPING TASK ENTITIES")
        logger.info("=" * 70)

        if 'task_statements' not in self.onet_data:
            logger.error("Task statements not loaded")
            return {}

        tasks = {}
        df = self.onet_data['task_statements']

        for _, row in df.iterrows():
            task_text = row.get('Task', '')
            task_id_onet = f"TASK_{row.get('Task ID', '')}"

            entity_id = generate_deterministic_uuid('task', task_text)
            self.onet_to_ksamds_ids[task_id_onet] = entity_id

            if entity_id not in tasks:
                task = KSAMDSEntity(
                    id=entity_id, name=task_text, source_ref=task_id_onet)
                tasks[entity_id] = task

        logger.info(f"Mapped {len(tasks)} unique task entities")
        return tasks

    def map_function_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET work activities to KSAMDS function entities."""
        logger.info("=" * 70)
        logger.info("MAPPING FUNCTION ENTITIES")
        logger.info("=" * 70)

        if 'work_activities' not in self.onet_data:
            logger.error("Work activities not loaded")
            return {}

        df = self.onet_data['work_activities']
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]

        logger.info(
            f"Filtered to {len(df_filtered)} function records (IM >= 3.0) from {len(df)} total")

        name_to_canonical = self.map_similar_entities(df_filtered)

        functions = {}
        for _, row in df_filtered.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('function', canonical_name)

            if entity_id not in functions:
                function = KSAMDSEntity(
                    id=entity_id, name=canonical_name, source_ref=element_id)
                functions[entity_id] = function

            self.onet_to_ksamds_ids[element_id] = entity_id

        logger.info(f"Mapped {len(functions)} unique function entities")
        return functions

    def map_education_levels(self) -> Dict[str, KSAMDSEntity]:
        """Create standard education level entities."""
        logger.info("=" * 70)
        logger.info("CREATING EDUCATION LEVEL ENTITIES")
        logger.info("=" * 70)

        education_levels = [
            ("High School", 1),
            ("Some College", 2),
            ("Associate's Degree", 3),
            ("Bachelor's Degree", 4),
            ("Master's Degree", 5),
            ("Doctoral Degree", 6)
        ]

        entities = {}
        for name, ordinal in education_levels:
            entity_id = generate_deterministic_uuid('education_level', name)
            edu = KSAMDSEntity(
                id=entity_id, name=name, source_ref=f"EDU_{ordinal}")
            entities[entity_id] = edu

        logger.info(f"Created {len(entities)} education level entities")
        return entities

    def save_mapped_entities(self) -> bool:
        """Save all mapped entities and relationships to CSV files."""
        logger.info("=" * 70)
        logger.info("SAVING MAPPED ENTITIES")
        logger.info("=" * 70)

        try:
            # Save core entities
            for entity_type, entities in self.ksamds_entities.items():
                if not entities:
                    continue

                rows = []
                for entity in entities.values():
                    row = {
                        'id': entity.id,
                        'name': entity.name,
                        'source_ref': entity.source_ref,
                        'type_dims': '|'.join(entity.type_dims) if entity.type_dims else '',
                        'level_dims': '|'.join(entity.level_dims) if entity.level_dims else '',
                        'basis_dims': '|'.join(entity.basis_dims) if entity.basis_dims else '',
                        'environment_dims': '|'.join(entity.environment_dims) if entity.environment_dims else '',
                        'mode_dims': '|'.join(entity.mode_dims) if entity.mode_dims else '',
                        'physicality_dims': '|'.join(entity.physicality_dims) if entity.physicality_dims else '',
                        'cognitive_dims': '|'.join(entity.cognitive_dims) if entity.cognitive_dims else ''
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)
                csv_path = self.mapped_dir / f"{entity_type}_mapped.csv"
                df.to_csv(csv_path, index=False)
                logger.info(
                    f"Saved {len(rows)} {entity_type} entities to {csv_path}")

            # Save relationships
            for rel_type, relationships in self.ksamds_relationships.items():
                if not relationships:
                    continue

                # Check if relationships include OccupationRelationship objects
                if relationships and isinstance(relationships[0], OccupationRelationship):
                    rows = []
                    for rel in relationships:
                        row = {
                            'occupation_id': rel.occupation_id,
                            'entity_id': rel.entity_id,
                            'level': rel.level if rel.level else '',
                            'importance_score': rel.importance_score if rel.importance_score else ''
                        }
                        rows.append(row)
                    df = pd.DataFrame(rows)
                else:
                    # Simple tuple relationships
                    df = pd.DataFrame(relationships, columns=[
                                      'source_id', 'target_id'])

                csv_path = self.mapped_dir / f"{rel_type}_relationships.csv"
                df.to_csv(csv_path, index=False)
                logger.info(
                    f"Saved {len(relationships)} {rel_type} relationships to {csv_path}")

            logger.info("All mapped data saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save mapped entities: {e}")
            return False

    def map_occupation_relationships(self) -> Dict[str, List]:
        """
        Map occupation relationships to Knowledge, Skills, Abilities, Tasks, and Functions.

        UPDATED: Now includes level tracking for K/S/A relationships using data
        stored in self.occupation_entity_levels.
        """
        logger.info("=" * 70)
        logger.info("MAPPING OCCUPATION RELATIONSHIPS WITH LEVELS")
        logger.info("=" * 70)

        relationships = defaultdict(list)

        # KNOWLEDGE RELATIONSHIPS (with levels)
        if 'knowledge' in self.onet_data:
            df_filtered = self.onet_data['knowledge'][(self.onet_data['knowledge']['Scale ID'] == 'IM') &
                                                      (self.onet_data['knowledge']['Data Value'] >= 3.0)]
            logger.info(
                f"Processing {len(df_filtered)} knowledge relationships (filtered from {len(self.onet_data['knowledge'])})")

            for _, row in df_filtered.iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                importance_score = row.get('Data Value', None)

                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                knowledge_ksamds_id = self.onet_to_ksamds_ids.get(element_id)

                if occupation_ksamds_id and knowledge_ksamds_id:
                    # Get the level requirement for this occupation-knowledge pair
                    level = self.occupation_entity_levels.get(
                        (onet_soc, element_id))

                    rel = OccupationRelationship(
                        occupation_id=occupation_ksamds_id,
                        entity_id=knowledge_ksamds_id,
                        level=level,
                        importance_score=importance_score
                    )

                    # Avoid duplicates
                    if not any(r.occupation_id == rel.occupation_id and
                               r.entity_id == rel.entity_id
                               for r in relationships['occupation_knowledge']):
                        relationships['occupation_knowledge'].append(rel)

        # SKILL RELATIONSHIPS (with levels)
        if 'skills' in self.onet_data:
            df_filtered = self.onet_data['skills'][(self.onet_data['skills']['Scale ID'] == 'IM') &
                                                   (self.onet_data['skills']['Data Value'] >= 3.0)]
            logger.info(
                f"Processing {len(df_filtered)} skill relationships (filtered from {len(self.onet_data['skills'])})")

            for _, row in df_filtered.iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                importance_score = row.get('Data Value', None)

                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                skill_ksamds_id = self.onet_to_ksamds_ids.get(element_id)

                if occupation_ksamds_id and skill_ksamds_id:
                    level = self.occupation_entity_levels.get(
                        (onet_soc, element_id))

                    rel = OccupationRelationship(
                        occupation_id=occupation_ksamds_id,
                        entity_id=skill_ksamds_id,
                        level=level,
                        importance_score=importance_score
                    )

                    if not any(r.occupation_id == rel.occupation_id and
                               r.entity_id == rel.entity_id
                               for r in relationships['occupation_skill']):
                        relationships['occupation_skill'].append(rel)

        # ABILITY RELATIONSHIPS (with levels)
        if 'abilities' in self.onet_data:
            df_filtered = self.onet_data['abilities'][(self.onet_data['abilities']['Scale ID'] == 'IM') &
                                                      (self.onet_data['abilities']['Data Value'] >= 3.0)]
            logger.info(
                f"Processing {len(df_filtered)} ability relationships (filtered from {len(self.onet_data['abilities'])})")

            for _, row in df_filtered.iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                importance_score = row.get('Data Value', None)

                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                ability_ksamds_id = self.onet_to_ksamds_ids.get(element_id)

                if occupation_ksamds_id and ability_ksamds_id:
                    level = self.occupation_entity_levels.get(
                        (onet_soc, element_id))

                    rel = OccupationRelationship(
                        occupation_id=occupation_ksamds_id,
                        entity_id=ability_ksamds_id,
                        level=level,
                        importance_score=importance_score
                    )

                    if not any(r.occupation_id == rel.occupation_id and
                               r.entity_id == rel.entity_id
                               for r in relationships['occupation_ability']):
                        relationships['occupation_ability'].append(rel)

        # TASK RELATIONSHIPS (no levels)
        if 'task_statements' in self.onet_data:
            logger.info(
                f"Processing {len(self.onet_data['task_statements'])} task relationships from task_statements")

            for _, row in self.onet_data['task_statements'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                task_id = f"TASK_{row.get('Task ID', '')}"
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                task_ksamds_id = self.onet_to_ksamds_ids.get(task_id)
                if occupation_ksamds_id and task_ksamds_id:
                    relationship = (occupation_ksamds_id, task_ksamds_id)
                    if relationship not in relationships['occupation_task']:
                        relationships['occupation_task'].append(relationship)

        # FUNCTION RELATIONSHIPS (no levels)
        if 'work_activities' in self.onet_data:
            df_filtered = self.onet_data['work_activities'][(self.onet_data['work_activities']['Scale ID'] == 'IM') &
                                                            (self.onet_data['work_activities']['Data Value'] >= 3.0)]
            logger.info(
                f"Processing {len(df_filtered)} function relationships (filtered from {len(self.onet_data['work_activities'])})")

            for _, row in df_filtered.iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                function_ksamds_id = self.onet_to_ksamds_ids.get(element_id)
                if occupation_ksamds_id and function_ksamds_id:
                    relationship = (occupation_ksamds_id, function_ksamds_id)
                    if relationship not in relationships['occupation_function']:
                        relationships['occupation_function'].append(
                            relationship)

        # EDUCATION RELATIONSHIPS (no levels)
        if 'job_zones' in self.onet_data:
            job_zone_to_education = {
                1: "High School", 2: "High School", 3: "Some College",
                4: "Bachelor's Degree", 5: "Master's Degree"
            }

            education_lookup = {}
            for edu_id, edu_entity in self.ksamds_entities.get('education_level', {}).items():
                education_lookup[edu_entity.name] = edu_id

            for _, row in self.onet_data['job_zones'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                job_zone = row.get('Job Zone', 1)
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                if occupation_ksamds_id:
                    education_name = job_zone_to_education.get(
                        job_zone, "High School")
                    education_ksamds_id = education_lookup.get(education_name)
                    if education_ksamds_id:
                        relationship = (occupation_ksamds_id,
                                        education_ksamds_id)
                        if relationship not in relationships['occupation_education']:
                            relationships['occupation_education'].append(
                                relationship)

        for rel_type, rel_list in relationships.items():
            logger.info(f"Mapped {len(rel_list)} {rel_type} relationships")

        return relationships

    def cleanup_intermediate_files(self):
        """
        Clean up intermediate O*NET CSV files after mapping is complete.
        """
        logger.info("Cleaning up intermediate O*NET CSV files...")

        # Clean up the intermediate directory
        if self.processed_dir.exists() and self.processed_dir.parts[-1] == 'intermediate':
            try:
                file_count = len(list(self.processed_dir.glob('*.csv')))
                shutil.rmtree(self.processed_dir)
                logger.info(
                    f"Removed {file_count} intermediate CSV files from {self.processed_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up intermediate files: {e}")

    def map_all_entities(self, cleanup_after: bool = True) -> bool:
        """
        Map all O*NET entities to KSAMDS structure.

        Args:
            cleanup_after: Whether to clean up intermediate files after mapping

        Returns:
            bool: True if successful
        """
        logger.info("=" * 70)
        logger.info("STARTING O*NET TO KSAMDS MAPPING")
        logger.info("=" * 70)

        if not self.load_onet_data():
            return False
        if not self.build_hierarchy_mappings():
            return False

        self.ksamds_entities['occupation'] = self.map_occupation_entities()
        self.ksamds_entities['knowledge'] = self.map_knowledge_entities()
        self.ksamds_entities['skill'] = self.map_skill_entities()
        self.ksamds_entities['ability'] = self.map_ability_entities()
        self.ksamds_entities['task'] = self.map_task_entities()
        self.ksamds_entities['function'] = self.map_function_entities()
        self.ksamds_entities['education_level'] = self.map_education_levels()

        self.ksamds_relationships = self.map_occupation_relationships()

        if not self.save_mapped_entities():
            return False

        # Clean up intermediate files if requested
        if cleanup_after:
            self.cleanup_intermediate_files()

        logger.info("O*NET to KSAMDS mapping completed successfully!")
        return True

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of mapping results."""
        summary = {
            'entities': {}, 'relationships': {},
            'hierarchy_mappings': len(self.hierarchy_mappings),
            'id_mappings': len(self.onet_to_ksamds_ids),
            'job_zones': len(self.job_zone_lookup),
            'occupation_level_mappings': len(self.occupation_entity_levels)
        }

        for entity_type, entities in self.ksamds_entities.items():
            summary['entities'][entity_type] = len(entities)

        for rel_type, relationships in self.ksamds_relationships.items():
            summary['relationships'][rel_type] = len(relationships)

        return summary


def main():
    """Main function to run the O*NET mapper."""
    # Initialize mapper
    mapper = ONetMapper()
    success = mapper.map_all_entities(cleanup_after=True)

    if success:
        summary = mapper.get_mapping_summary()
        logger.info("=" * 70)
        logger.info("O*NET TO KSAMDS MAPPING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Hierarchy mappings: {summary['hierarchy_mappings']}")
        logger.info(f"ID mappings: {summary['id_mappings']}")
        logger.info(
            f"Occupation-level mappings: {summary['occupation_level_mappings']}")

        logger.info("\nENTITY MAPPING RESULTS")
        logger.info("-" * 70)
        for entity_type, count in summary['entities'].items():
            if count > 0:
                logger.info(f"{entity_type.upper()}: {count} entities mapped")

        logger.info("\nRELATIONSHIP MAPPING RESULTS")
        logger.info("-" * 70)
        for rel_type, count in summary['relationships'].items():
            if count > 0:
                logger.info(
                    f"{rel_type.upper()}: {count} relationships mapped")

        logger.info("=" * 70)
        logger.info(f"Mapped data saved to: {mapper.mapped_dir}")
        logger.info("=" * 70)
    else:
        logger.error("Mapping failed. Check logs for details.")


if __name__ == "__main__":
    main()
