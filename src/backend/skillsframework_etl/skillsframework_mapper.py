"""
Skills Framework Data Mapper for KSAMDS Project

This module handles the mapping and transformation of Singapore's SkillsFuture
Skills Framework data into the KSAMDS multi-dimensional structure.
Uses deterministic UUIDs to ensure idempotent pipeline execution.

Maps occupations, knowledge, skills, abilities, functions, and tasks with their
associated dimensions (type, basis, level, physicality, cognitive_load, environment, mode).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
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
    description: Optional[str] = None
    # Dimensions (optional)
    type_dims: List[str] = field(default_factory=list)
    level_dims: List[str] = field(default_factory=list)
    basis_dims: List[str] = field(default_factory=list)
    environment_dims: List[str] = field(default_factory=list)
    mode_dims: List[str] = field(default_factory=list)
    physicality_dims: List[str] = field(default_factory=list)
    cognitive_dims: List[str] = field(default_factory=list)


@dataclass
class OccupationRelationship:
    """Represents an occupation relationship with level tracking."""
    occupation_id: str
    entity_id: str
    level: Optional[str] = None


class SkillsFrameworkMapper:
    """Map Skills Framework data to KSAMDS multi-dimensional structure."""

    def __init__(self):
        """Initialize the Skills Framework mapper."""
        # Get project root directory (3 levels up from skillsframework_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.processed_dir = project_root / "data" / \
            "skillsframework" / "archive" / "intermediate"
        self.mapped_dir = project_root / "data" / \
            "skillsframework" / "archive" / "mapped"
        self.relationships_dir = project_root / "data" / \
            "skillsframework" / "archive" / "relationships"

        self.mapped_dir.mkdir(parents=True, exist_ok=True)
        self.relationships_dir.mkdir(parents=True, exist_ok=True)

        self.sf_data: Dict[str, pd.DataFrame] = {}
        self.ksamds_entities = {
            'knowledge': {}, 'skill': {}, 'ability': {},
            'occupation': {}, 'function': {}, 'task': {}
        }
        self.sf_to_ksamds_ids = {}
        self.ksamds_relationships = defaultdict(list)

    def normalize_knowledge_level(self, level) -> str:
        """
        Normalize knowledge level to standardized categories.

        Mapping:
        - 1, 2 -> Basic
        - 3 -> Intermediate
        - 4, 5 -> Advanced
        - 6 -> Expert

        Args:
            level: Raw level value (int, float, or str)

        Returns:
            Normalized level string
        """
        if pd.isna(level):
            return 'Basic'  # Default

        level_str = str(level).strip().lower()

        # Already standardized
        if level_str in ['basic', 'intermediate', 'advanced', 'expert']:
            return level_str.capitalize()

        # Convert numeric
        try:
            level_num = float(level)
            if level_num in [1, 2]:
                return 'Basic'
            elif level_num == 3:
                return 'Intermediate'
            elif level_num in [4, 5]:
                return 'Advanced'
            elif level_num == 6:
                return 'Expert'
        except (ValueError, TypeError):
            pass

        # Default fallback
        return 'Basic'

    def normalize_skill_level(self, level) -> str:
        """
        Normalize skill level to standardized categories.

        Mapping:
        - 1, 2, Basic -> Novice
        - 3, 4, Intermediate -> Proficient
        - 5, 6, Advanced -> Expert

        Args:
            level: Raw level value (int, float, or str)

        Returns:
            Normalized level string
        """
        if pd.isna(level):
            return 'Novice'  # Default

        level_str = str(level).strip().lower()

        # Already standardized
        if level_str in ['novice', 'proficient', 'expert']:
            return level_str.capitalize()

        # Text mappings
        if level_str == 'basic':
            return 'Novice'
        elif level_str == 'intermediate':
            return 'Proficient'
        elif level_str == 'advanced':
            return 'Expert'

        # Convert numeric
        try:
            level_num = float(level)
            if level_num in [1, 2]:
                return 'Novice'
            elif level_num in [3, 4]:
                return 'Proficient'
            elif level_num in [5, 6]:
                return 'Expert'
        except (ValueError, TypeError):
            pass

        # Default fallback
        return 'Novice'

    def normalize_ability_level(self, level) -> str:
        """
        Normalize ability level to standardized categories.

        Mapping:
        - 1, 2, Basic -> Low
        - 3, 4, Intermediate -> Moderate
        - 5, 6, Advanced -> High

        Args:
            level: Raw level value (int, float, or str)

        Returns:
            Normalized level string
        """
        if pd.isna(level):
            return 'Low'  # Default

        level_str = str(level).strip().lower()

        # Already standardized
        if level_str in ['low', 'moderate', 'high']:
            return level_str.capitalize()

        # Text mappings
        if level_str == 'basic':
            return 'Low'
        elif level_str == 'intermediate':
            return 'Moderate'
        elif level_str == 'advanced':
            return 'High'

        # Convert numeric
        try:
            level_num = float(level)
            if level_num in [1, 2]:
                return 'Low'
            elif level_num in [3, 4]:
                return 'Moderate'
            elif level_num in [5, 6]:
                return 'High'
        except (ValueError, TypeError):
            pass

        # Default fallback
        return 'Low'

    def load_sf_data(self) -> bool:
        """Load processed Skills Framework CSV files."""
        logger.info("=" * 70)
        logger.info("LOADING SKILLS FRAMEWORK DATA")
        logger.info("=" * 70)

        required_files = ['occupations', 'knowledge', 'skills', 'abilities',
                          'functions', 'tasks']

        try:
            for file_key in required_files:
                csv_path = self.processed_dir / f"{file_key}.csv"
                if csv_path.exists():
                    self.sf_data[file_key] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {file_key}: {len(self.sf_data[file_key])} records")
                else:
                    logger.error(f"Required file not found: {csv_path}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Failed to load Skills Framework data: {e}")
            return False

    def map_occupation_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map occupation entities to KSAMDS structure."""
        logger.info("Mapping occupation entities...")

        if 'occupations' not in self.sf_data:
            logger.warning("Occupations data not loaded")
            return {}

        occupations = {}
        df = self.sf_data['occupations']

        for _, row in df.iterrows():
            occupation_name = str(row['occupation']).strip()

            # Get description if available
            description = None
            if 'description' in row and pd.notna(row['description']):
                description = str(row['description']).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid(
                'occupation', occupation_name)

            # Store mapping from original name to KSAMDS ID
            # Use a prefix to avoid collision with knowledge/skill/ability names
            self.sf_to_ksamds_ids[f"occupation:{occupation_name}"] = ksamds_id

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=occupation_name,
                source_ref=f"SF:{occupation_name}",
                description=description
            )

            occupations[ksamds_id] = entity

        logger.info(f"Mapped {len(occupations)} occupation entities")
        return occupations

    def map_knowledge_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map knowledge entities to KSAMDS structure."""
        logger.info("Mapping knowledge entities...")

        if 'knowledge' not in self.sf_data:
            logger.warning("Knowledge data not loaded")
            return {}

        knowledge_entities = {}
        df = self.sf_data['knowledge']

        # Group by knowledge to get unique knowledge items with their dimensions
        knowledge_groups = df.groupby('knowledge')

        for knowledge_name, group in knowledge_groups:
            knowledge_name = str(knowledge_name).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid(
                'knowledge', knowledge_name)

            # Store mapping with prefix to avoid collision with other entity types
            self.sf_to_ksamds_ids[f"knowledge:{knowledge_name}"] = ksamds_id

            # Extract dimensions from the first row (should be consistent)
            first_row = group.iloc[0]

            type_dims = []
            if 'knowledge_type' in first_row and pd.notna(first_row['knowledge_type']):
                type_dims.append(str(first_row['knowledge_type']))

            basis_dims = []
            if 'knowledge_basis' in first_row and pd.notna(first_row['knowledge_basis']):
                basis_dims.append(str(first_row['knowledge_basis']))

            # Collect all unique levels across occupations
            level_dims = []
            if 'level' in group.columns:
                levels = group['level'].dropna().unique()
                normalized_levels = set()
                for level in levels:
                    normalized_levels.add(
                        self.normalize_knowledge_level(level))
                level_dims = sorted(list(normalized_levels))

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=knowledge_name,
                source_ref=f"SF:K:{knowledge_name}",
                type_dims=type_dims,
                basis_dims=basis_dims,
                level_dims=level_dims
            )

            knowledge_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(knowledge_entities)} knowledge entities")
        return knowledge_entities

    def map_skill_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map skill entities to KSAMDS structure."""
        logger.info("Mapping skill entities...")

        if 'skills' not in self.sf_data:
            logger.warning("Skills data not loaded")
            return {}

        skill_entities = {}
        df = self.sf_data['skills']

        # Group by skill to get unique skills with their dimensions
        skill_groups = df.groupby('skill')

        for skill_name, group in skill_groups:
            skill_name = str(skill_name).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid('skill', skill_name)

            # Store mapping with prefix to avoid collision with other entity types
            self.sf_to_ksamds_ids[f"skill:{skill_name}"] = ksamds_id

            # Extract dimensions
            first_row = group.iloc[0]

            type_dims = []
            if 'skill_type' in first_row and pd.notna(first_row['skill_type']):
                type_dims.append(str(first_row['skill_type']))

            basis_dims = []
            if 'skill_basis' in first_row and pd.notna(first_row['skill_basis']):
                basis_dims.append(str(first_row['skill_basis']))

            # Collect all unique levels
            level_dims = []
            if 'level' in group.columns:
                levels = group['level'].dropna().unique()
                normalized_levels = set()
                for level in levels:
                    normalized_levels.add(self.normalize_skill_level(level))
                level_dims = sorted(list(normalized_levels))

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=skill_name,
                source_ref=f"SF:S:{skill_name}",
                type_dims=type_dims,
                basis_dims=basis_dims,
                level_dims=level_dims
            )

            skill_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(skill_entities)} skill entities")
        return skill_entities

    def map_ability_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map ability entities to KSAMDS structure."""
        logger.info("Mapping ability entities...")

        if 'abilities' not in self.sf_data:
            logger.warning("Abilities data not loaded")
            return {}

        ability_entities = {}
        df = self.sf_data['abilities']

        # Group by ability to get unique abilities with their dimensions
        ability_groups = df.groupby('ability')

        for ability_name, group in ability_groups:
            ability_name = str(ability_name).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid('ability', ability_name)

            # Store mapping with prefix to avoid collision with other entity types
            self.sf_to_ksamds_ids[f"ability:{ability_name}"] = ksamds_id

            # Extract dimensions
            first_row = group.iloc[0]

            type_dims = []
            if 'ability_type' in first_row and pd.notna(first_row['ability_type']):
                type_dims.append(str(first_row['ability_type']))

            basis_dims = []
            if 'ability_basis' in first_row and pd.notna(first_row['ability_basis']):
                basis_dims.append(str(first_row['ability_basis']))

            # Collect all unique levels
            level_dims = []
            if 'level' in group.columns:
                levels = group['level'].dropna().unique()
                normalized_levels = set()
                for level in levels:
                    normalized_levels.add(self.normalize_ability_level(level))
                level_dims = sorted(list(normalized_levels))

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=ability_name,
                source_ref=f"SF:A:{ability_name}",
                type_dims=type_dims,
                basis_dims=basis_dims,
                level_dims=level_dims
            )

            ability_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(ability_entities)} ability entities")
        return ability_entities

    def map_function_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map function entities to KSAMDS structure."""
        logger.info("Mapping function entities...")

        if 'functions' not in self.sf_data:
            logger.warning("Functions data not loaded")
            return {}

        function_entities = {}
        df = self.sf_data['functions']

        # Group by function to get unique functions with their dimensions
        function_groups = df.groupby('function')

        for function_name, group in function_groups:
            function_name = str(function_name).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid('function', function_name)

            # Store mapping with prefix to avoid collision with other entity types
            self.sf_to_ksamds_ids[f"function:{function_name}"] = ksamds_id

            # Extract dimensions
            first_row = group.iloc[0]

            physicality_dims = []
            if 'function_physicality' in first_row and pd.notna(first_row['function_physicality']):
                physicality_dims.append(str(first_row['function_physicality']))

            cognitive_dims = []
            if 'function_cognitive_load' in first_row and pd.notna(first_row['function_cognitive_load']):
                cognitive_dims.append(
                    str(first_row['function_cognitive_load']))

            environment_dims = []
            if 'function_environment' in first_row and pd.notna(first_row['function_environment']):
                environment_dims.append(str(first_row['function_environment']))

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=function_name,
                source_ref=f"SF:F:{function_name}",
                physicality_dims=physicality_dims,
                cognitive_dims=cognitive_dims,
                environment_dims=environment_dims
            )

            function_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(function_entities)} function entities")
        return function_entities

    def map_task_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map task entities to KSAMDS structure."""
        logger.info("Mapping task entities...")

        if 'tasks' not in self.sf_data:
            logger.warning("Tasks data not loaded")
            return {}

        task_entities = {}
        df = self.sf_data['tasks']

        # Group by task to get unique tasks with their dimensions
        task_groups = df.groupby('task')

        for task_name, group in task_groups:
            task_name = str(task_name).strip()

            # Generate deterministic UUID
            ksamds_id = generate_deterministic_uuid('task', task_name)

            # Store mapping with prefix to avoid collision with other entity types
            self.sf_to_ksamds_ids[f"task:{task_name}"] = ksamds_id

            # Extract dimensions
            first_row = group.iloc[0]

            type_dims = []
            if 'task_type' in first_row and pd.notna(first_row['task_type']):
                type_dims.append(str(first_row['task_type']))

            mode_dims = []
            if 'task_mode' in first_row and pd.notna(first_row['task_mode']):
                mode_dims.append(str(first_row['task_mode']))

            environment_dims = []
            if 'task_environment' in first_row and pd.notna(first_row['task_environment']):
                environment_dims.append(str(first_row['task_environment']))

            entity = KSAMDSEntity(
                id=ksamds_id,
                name=task_name,
                source_ref=f"SF:T:{task_name}",
                type_dims=type_dims,
                mode_dims=mode_dims,
                environment_dims=environment_dims
            )

            task_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(task_entities)} task entities")
        return task_entities

    def map_occupation_relationships(self) -> Dict[str, List]:
        """Map relationships between occupations and other entities."""
        logger.info("=" * 70)
        logger.info("MAPPING OCCUPATION RELATIONSHIPS")
        logger.info("=" * 70)

        relationships = defaultdict(list)

        # KNOWLEDGE RELATIONSHIPS
        if 'knowledge' in self.sf_data:
            logger.info(
                f"Processing {len(self.sf_data['knowledge'])} knowledge relationships")

            # Use set for O(1) duplicate checking
            seen_knowledge_pairs = set()

            for _, row in self.sf_data['knowledge'].iterrows():
                occupation_name = str(row['occupation']).strip()
                knowledge_name = str(row['knowledge']).strip()

                occupation_id = self.sf_to_ksamds_ids.get(
                    f"occupation:{occupation_name}")
                knowledge_id = self.sf_to_ksamds_ids.get(
                    f"knowledge:{knowledge_name}")

                if occupation_id and knowledge_id:
                    pair_key = (occupation_id, knowledge_id)
                    if pair_key not in seen_knowledge_pairs:
                        seen_knowledge_pairs.add(pair_key)

                        level = None
                        if 'level' in row and pd.notna(row['level']):
                            level = self.normalize_knowledge_level(
                                row['level'])

                        rel = OccupationRelationship(
                            occupation_id=occupation_id,
                            entity_id=knowledge_id,
                            level=level
                        )
                        relationships['occupation_knowledge'].append(rel)

        # SKILL RELATIONSHIPS
        if 'skills' in self.sf_data:
            logger.info(
                f"Processing {len(self.sf_data['skills'])} skill relationships")

            # Use set for O(1) duplicate checking
            seen_skill_pairs = set()

            for _, row in self.sf_data['skills'].iterrows():
                occupation_name = str(row['occupation']).strip()
                skill_name = str(row['skill']).strip()

                occupation_id = self.sf_to_ksamds_ids.get(
                    f"occupation:{occupation_name}")
                skill_id = self.sf_to_ksamds_ids.get(f"skill:{skill_name}")

                if occupation_id and skill_id:
                    pair_key = (occupation_id, skill_id)
                    if pair_key not in seen_skill_pairs:
                        seen_skill_pairs.add(pair_key)

                        level = None
                        if 'level' in row and pd.notna(row['level']):
                            level = self.normalize_skill_level(row['level'])

                        rel = OccupationRelationship(
                            occupation_id=occupation_id,
                            entity_id=skill_id,
                            level=level
                        )
                        relationships['occupation_skill'].append(rel)

        # ABILITY RELATIONSHIPS
        if 'abilities' in self.sf_data:
            logger.info(
                f"Processing {len(self.sf_data['abilities'])} ability relationships")

            # Use set for O(1) duplicate checking
            seen_ability_pairs = set()

            for _, row in self.sf_data['abilities'].iterrows():
                occupation_name = str(row['occupation']).strip()
                ability_name = str(row['ability']).strip()

                occupation_id = self.sf_to_ksamds_ids.get(
                    f"occupation:{occupation_name}")
                ability_id = self.sf_to_ksamds_ids.get(
                    f"ability:{ability_name}")

                if occupation_id and ability_id:
                    pair_key = (occupation_id, ability_id)
                    if pair_key not in seen_ability_pairs:
                        seen_ability_pairs.add(pair_key)

                        level = None
                        if 'level' in row and pd.notna(row['level']):
                            level = self.normalize_ability_level(row['level'])

                        rel = OccupationRelationship(
                            occupation_id=occupation_id,
                            entity_id=ability_id,
                            level=level
                        )
                        relationships['occupation_ability'].append(rel)

        # FUNCTION RELATIONSHIPS
        if 'functions' in self.sf_data:
            logger.info(
                f"Processing {len(self.sf_data['functions'])} function relationships")

            # Use set for O(1) duplicate checking
            seen_function_pairs = set()

            for _, row in self.sf_data['functions'].iterrows():
                occupation_name = str(row['occupation']).strip()
                function_name = str(row['function']).strip()

                occupation_id = self.sf_to_ksamds_ids.get(
                    f"occupation:{occupation_name}")
                function_id = self.sf_to_ksamds_ids.get(
                    f"function:{function_name}")

                if occupation_id and function_id:
                    pair_key = (occupation_id, function_id)
                    if pair_key not in seen_function_pairs:
                        seen_function_pairs.add(pair_key)
                        relationships['occupation_function'].append(pair_key)

        # TASK RELATIONSHIPS
        if 'tasks' in self.sf_data:
            logger.info(
                f"Processing {len(self.sf_data['tasks'])} task relationships")

            # Use set for O(1) duplicate checking
            seen_task_pairs = set()

            for _, row in self.sf_data['tasks'].iterrows():
                occupation_name = str(row['occupation']).strip()
                task_name = str(row['task']).strip()

                occupation_id = self.sf_to_ksamds_ids.get(
                    f"occupation:{occupation_name}")
                task_id = self.sf_to_ksamds_ids.get(f"task:{task_name}")

                if occupation_id and task_id:
                    pair_key = (occupation_id, task_id)
                    if pair_key not in seen_task_pairs:
                        seen_task_pairs.add(pair_key)
                        relationships['occupation_task'].append(pair_key)

        for rel_type, rel_list in relationships.items():
            logger.info(f"Mapped {len(rel_list)} {rel_type} relationships")

        return relationships

    def convert_relationship_files_to_uuids(self) -> bool:
        """
        Convert knowledge_skill.csv and skill_ability.csv to use UUIDs.

        Reads from data/skillsframework/archive/intermediate and saves to data/skillsframework/archive/relationships.
        """
        logger.info("=" * 70)
        logger.info("CONVERTING RELATIONSHIP FILES TO UUIDS")
        logger.info("=" * 70)

        try:
            # Create name-to-ID lookups from mapped entities
            knowledge_lookup = {}
            skill_lookup = {}
            ability_lookup = {}

            if 'knowledge' in self.ksamds_entities:
                for entity_id, entity in self.ksamds_entities['knowledge'].items():
                    knowledge_lookup[entity.name] = entity_id

            if 'skill' in self.ksamds_entities:
                for entity_id, entity in self.ksamds_entities['skill'].items():
                    skill_lookup[entity.name] = entity_id

            if 'ability' in self.ksamds_entities:
                for entity_id, entity in self.ksamds_entities['ability'].items():
                    ability_lookup[entity.name] = entity_id

            # Convert knowledge_skill.csv
            ks_input_path = self.processed_dir / "knowledge_skill.csv"
            if ks_input_path.exists():
                logger.info(
                    f"Converting knowledge_skill from {ks_input_path}...")
                ks_df = pd.read_csv(ks_input_path)

                converted_rows = []
                for _, row in ks_df.iterrows():
                    skill_name = str(row['skill']).strip()
                    knowledge_name = str(row['knowledge']).strip()

                    skill_id = skill_lookup.get(skill_name)
                    knowledge_id = knowledge_lookup.get(knowledge_name)

                    if skill_id and knowledge_id:
                        converted_rows.append({
                            'source_id': knowledge_id,
                            'target_id': skill_id,
                            'confidence_score': 1.0
                        })

                ks_output_df = pd.DataFrame(converted_rows)
                ks_output_path = self.relationships_dir / "knowledge_skill.csv"
                ks_output_df.to_csv(ks_output_path, index=False)
                logger.info(
                    f"Saved {len(converted_rows)} knowledge_skill relationships to {ks_output_path}")
            else:
                logger.warning(
                    f"knowledge_skill.csv not found at {ks_input_path}")

            # Convert skill_ability.csv
            sa_input_path = self.processed_dir / "skill_ability.csv"
            if sa_input_path.exists():
                logger.info(
                    f"Converting skill_ability from {sa_input_path}...")
                sa_df = pd.read_csv(sa_input_path)

                converted_rows = []
                for _, row in sa_df.iterrows():
                    skill_name = str(row['skill']).strip()
                    ability_name = str(row['ability']).strip()

                    skill_id = skill_lookup.get(skill_name)
                    ability_id = ability_lookup.get(ability_name)

                    if skill_id and ability_id:
                        converted_rows.append({
                            'source_id': skill_id,
                            'target_id': ability_id,
                            'confidence_score': 1.0
                        })

                sa_output_df = pd.DataFrame(converted_rows)
                sa_output_path = self.relationships_dir / "skill_ability.csv"
                sa_output_df.to_csv(sa_output_path, index=False)
                logger.info(
                    f"Saved {len(converted_rows)} skill_ability relationships to {sa_output_path}")
            else:
                logger.warning(
                    f"skill_ability.csv not found at {sa_input_path}")

            logger.info("=" * 70)
            logger.info("RELATIONSHIP CONVERSION COMPLETED")
            logger.info("=" * 70)

            return True

        except Exception as e:
            logger.error(f"Failed to convert relationship files: {e}")
            return False

    def save_mapped_entities(self) -> bool:
        """Save mapped entities and relationships to CSV files."""
        logger.info("=" * 70)
        logger.info("SAVING MAPPED ENTITIES")
        logger.info("=" * 70)

        try:
            # Save entities
            for entity_type, entities in self.ksamds_entities.items():
                if not entities:
                    continue

                rows = []
                for entity_id, entity in entities.items():
                    row = {
                        'id': entity.id,
                        'name': entity.name,
                        'source_ref': entity.source_ref,
                        'description': entity.description if entity.description else '',
                        'type_dimensions': '|'.join(entity.type_dims) if entity.type_dims else '',
                        'level_dimensions': '|'.join(entity.level_dims) if entity.level_dims else '',
                        'basis_dimensions': '|'.join(entity.basis_dims) if entity.basis_dims else '',
                        'environment_dimensions': '|'.join(entity.environment_dims) if entity.environment_dims else '',
                        'mode_dimensions': '|'.join(entity.mode_dims) if entity.mode_dims else '',
                        'physicality_dimensions': '|'.join(entity.physicality_dims) if entity.physicality_dims else '',
                        'cognitive_dimensions': '|'.join(entity.cognitive_dims) if entity.cognitive_dims else ''
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)
                output_path = self.mapped_dir / f"sf_{entity_type}.csv"
                df.to_csv(output_path, index=False)
                logger.info(
                    f"Saved {len(rows)} {entity_type} entities to {output_path}")

            # Save relationships
            for rel_type, rel_list in self.ksamds_relationships.items():
                if not rel_list:
                    continue

                rows = []
                for rel in rel_list:
                    if isinstance(rel, OccupationRelationship):
                        row = {
                            'occupation_id': rel.occupation_id,
                            'entity_id': rel.entity_id,
                            'level': rel.level if rel.level else ''
                        }
                    else:  # Tuple (for function and task relationships)
                        row = {
                            'occupation_id': rel[0],
                            'entity_id': rel[1]
                        }
                    rows.append(row)

                df = pd.DataFrame(rows)
                output_path = self.mapped_dir / f"sf_{rel_type}.csv"
                df.to_csv(output_path, index=False)
                logger.info(
                    f"Saved {len(rows)} {rel_type} relationships to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save mapped entities: {e}")
            return False

    def cleanup_intermediate_files(self):
        """Clean up intermediate Skills Framework CSV files after mapping is complete."""
        logger.info("Cleaning up intermediate Skills Framework CSV files...")

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
        Map all Skills Framework entities to KSAMDS structure.

        Args:
            cleanup_after: Whether to clean up intermediate files after mapping

        Returns:
            bool: True if successful
        """
        logger.info("=" * 70)
        logger.info("STARTING SKILLS FRAMEWORK TO KSAMDS MAPPING")
        logger.info("=" * 70)

        if not self.load_sf_data():
            return False

        self.ksamds_entities['occupation'] = self.map_occupation_entities()
        self.ksamds_entities['knowledge'] = self.map_knowledge_entities()
        self.ksamds_entities['skill'] = self.map_skill_entities()
        self.ksamds_entities['ability'] = self.map_ability_entities()
        self.ksamds_entities['function'] = self.map_function_entities()
        self.ksamds_entities['task'] = self.map_task_entities()

        self.ksamds_relationships = self.map_occupation_relationships()

        if not self.save_mapped_entities():
            return False

        # Convert relationship files to UUIDs
        if not self.convert_relationship_files_to_uuids():
            logger.warning(
                "Failed to convert relationship files, but mapping was successful")

        # Clean up intermediate files if requested
        if cleanup_after:
            self.cleanup_intermediate_files()

        logger.info(
            "Skills Framework to KSAMDS mapping completed successfully!")
        return True

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of mapping results."""
        summary = {
            'entities': {},
            'relationships': {},
            'id_mappings': len(self.sf_to_ksamds_ids)
        }

        for entity_type, entities in self.ksamds_entities.items():
            summary['entities'][entity_type] = len(entities)

        for rel_type, relationships in self.ksamds_relationships.items():
            summary['relationships'][rel_type] = len(relationships)

        return summary


def main():
    """Main function to run the Skills Framework mapper."""
    # Initialize mapper
    mapper = SkillsFrameworkMapper()
    success = mapper.map_all_entities(cleanup_after=True)

    if success:
        summary = mapper.get_mapping_summary()
        logger.info("=" * 70)
        logger.info("SKILLS FRAMEWORK TO KSAMDS MAPPING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"ID mappings: {summary['id_mappings']}")

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
