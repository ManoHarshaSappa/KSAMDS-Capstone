"""
O*NET Data Mapper for KSAMDS Project - DETERMINISTIC VERSION

This module handles the complex mapping and transformation of O*NET data
into the KSAMDS multi-dimensional structure. Uses deterministic UUIDs to
ensure idempotent pipeline execution.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import re
from dataclasses import dataclass
from fuzzywuzzy import fuzz  # type: ignore
from collections import defaultdict
import uuid

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
    type_dims: List[str] = None
    level_dims: List[str] = None
    basis_dims: List[str] = None
    environment_dims: List[str] = None
    mode_dims: List[str] = None
    physicality_dims: List[str] = None
    cognitive_dims: List[str] = None


class ONetMapper:
    """Map O*NET data to KSAMDS multi-dimensional structure."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the O*NET mapper."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed" / "onet"
        self.mapped_dir = self.data_dir / "processed" / "ksamds"
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

    def load_onet_data(self) -> bool:
        """Load processed O*NET CSV files."""
        required_files = ['knowledge', 'skills', 'abilities', 'occupation_data',
                          'task_ratings', 'task_statements', 'work_activities',
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

    def get_level_from_rating(self, lv_score: float, entity_type: str) -> str:
        """Map O*NET LV (Level) score to KSAMDS level dimension."""
        if pd.isna(lv_score):
            return 'Basic'
        if lv_score >= 6.0:
            return 'Expert'
        elif lv_score >= 4.5:
            return 'Advanced'
        elif lv_score >= 3.0:
            return 'Intermediate'
        else:
            return 'Basic'

    def get_task_environment(self, task_name: str, task_description: str) -> str:
        """Infer task environment from task content."""
        text = f"{task_name} {task_description}".lower()
        if any(word in text for word in ['office', 'computer', 'desk', 'meeting', 'email']):
            return 'Office_Tasks'
        elif any(word in text for word in ['field', 'site', 'outdoor', 'construction']):
            return 'Field_Tasks'
        elif any(word in text for word in ['laboratory', 'lab', 'test', 'sample']):
            return 'Laboratory_Tasks'
        elif any(word in text for word in ['factory', 'plant', 'manufacturing']):
            return 'Factory'
        elif any(word in text for word in ['remote', 'virtual', 'online']):
            return 'Remote'
        return 'Office_Tasks'

    def get_task_mode(self, task_name: str, task_description: str) -> str:
        """Infer task mode from task content."""
        text = f"{task_name} {task_description}".lower()
        if any(word in text for word in ['use', 'operate', 'tool', 'equipment']):
            return 'Tool'
        elif any(word in text for word in ['follow', 'procedure', 'process', 'protocol']):
            return 'Process'
        elif any(word in text for word in ['analyze', 'calculate', 'research', 'theory']):
            return 'Theory'
        elif any(word in text for word in ['design', 'create', 'develop', 'innovate']):
            return 'Creative'
        return 'Process'

    def get_function_environment(self, function_name: str, function_description: str) -> str:
        """Infer function environment from content."""
        text = f"{function_name} {function_description}".lower()
        if any(word in text for word in ['office', 'desk', 'computer']):
            return 'Office'
        elif any(word in text for word in ['outdoor', 'field', 'site']):
            return 'Outdoor'
        elif any(word in text for word in ['laboratory', 'lab', 'research']):
            return 'Laboratory'
        elif any(word in text for word in ['factory', 'manufacturing']):
            return 'Factory'
        elif any(word in text for word in ['remote', 'virtual', 'online']):
            return 'Remote'
        return 'Office'

    def get_function_physicality(self, function_name: str, function_description: str) -> str:
        """Infer function physicality demands."""
        text = f"{function_name} {function_description}".lower()
        if any(word in text for word in ['lifting', 'carrying', 'physical', 'manual']):
            return 'Heavy'
        elif any(word in text for word in ['walking', 'standing', 'moving']):
            return 'Moderate'
        return 'Light'

    def get_function_cognitive(self, function_name: str, function_description: str) -> str:
        """Infer function cognitive demands."""
        text = f"{function_name} {function_description}".lower()
        if any(word in text for word in ['analyzing', 'complex', 'research', 'strategic']):
            return 'Heavy'
        elif any(word in text for word in ['reviewing', 'coordinating', 'organizing']):
            return 'Moderate'
        return 'Light'

    def get_basis_from_content(self, name: str, description: str, entity_type: str) -> str:
        """Infer basis dimension from content analysis."""
        text = f"{name} {description}".lower()

        if entity_type == 'S':
            if any(word in text for word in ['mathematics', 'science', 'engineering', 'programming']):
                return 'Formal_Education'
            elif any(word in text for word in ['equipment operation', 'machinery']):
                return 'On_the_Job_Training'
            elif any(word in text for word in ['certification', 'license', 'protocol']):
                return 'Training'
            return 'Experience'
        elif entity_type == 'K':
            if any(word in text for word in ['theory', 'principles', 'academic']):
                return 'Formal_Education'
            elif any(word in text for word in ['procedures', 'methods', 'techniques']):
                return 'Training'
            return 'Experience'
        elif entity_type == 'A':
            if any(word in text for word in ['vision', 'hearing', 'strength']):
                return 'Natural'
            return 'Developed'
        return 'Experience'

    def map_occupation_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET occupation data to KSAMDS occupation entities."""
        if 'occupation_data' not in self.onet_data:
            return {}

        logger.info("Mapping occupation entities...")
        df = self.onet_data['occupation_data']
        occupation_entities = {}

        for _, row in df.iterrows():
            occupation_title = row.get('Title', '')
            onet_code = row.get('O*NET-SOC Code', '')

            ksamds_id = generate_deterministic_uuid(
                'occupation', occupation_title)

            if onet_code:
                self.onet_to_ksamds_ids[onet_code] = ksamds_id

            entity = KSAMDSEntity(
                id=ksamds_id, name=occupation_title,
                source_ref="O*NET"
            )
            # Since occupation table DOES have a description column
            entity.description = row.get('Description', '')

            occupation_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(occupation_entities)} occupation entities")
        return occupation_entities

    def map_knowledge_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET knowledge data to KSAMDS knowledge entities."""
        if 'knowledge' not in self.onet_data:
            return {}

        logger.info("Mapping knowledge entities...")
        df = self.onet_data['knowledge']

        knowledge_items = []
        element_ratings = {}

        for _, row in df.iterrows():
            element_id = row.get('Element ID', '')
            scale_id = row.get('Scale ID', '')
            data_value = row.get('Data Value', 0)

            if element_id not in element_ratings:
                element_ratings[element_id] = {}
            element_ratings[element_id][scale_id] = data_value

            knowledge_items.append({
                'element_id': element_id,
                'name': row.get('Element Name', ''),
                'definition': row.get('Description', ''),
                'onet_soc_code': row.get('O*NET-SOC Code', '')
            })

        # Deduplicate by name
        name_groups = defaultdict(list)
        for item in knowledge_items:
            clean_name = item['name'].strip()
            if clean_name:
                name_groups[clean_name].append(item)

        unique_knowledge = {}
        for knowledge_name, duplicate_items in name_groups.items():
            base_item = duplicate_items[0]
            all_element_ids = []
            best_definition = base_item['definition']

            for item in duplicate_items:
                if item['element_id']:
                    all_element_ids.append(item['element_id'])
                if len(item['definition']) > len(best_definition):
                    best_definition = item['definition']

            unique_knowledge[knowledge_name] = {
                'name': knowledge_name,
                'definition': best_definition,
                'primary_element_id': all_element_ids[0] if all_element_ids else '',
                'all_element_ids': list(set(all_element_ids))
            }

        logger.info(
            f"Deduplicated {len(knowledge_items)} to {len(unique_knowledge)} unique knowledge")

        knowledge_entities = {}
        for knowledge_name, item in unique_knowledge.items():
            ksamds_id = generate_deterministic_uuid(
                'knowledge', knowledge_name)

            for element_id in item['all_element_ids']:
                if element_id:
                    self.onet_to_ksamds_ids[element_id] = ksamds_id

            primary_element_id = item['primary_element_id']
            type_dim = self.get_type_from_hierarchy(primary_element_id)
            lv_rating = element_ratings.get(
                primary_element_id, {}).get('LV', 3.0)
            level_dim = self.get_level_from_rating(lv_rating, 'K')
            basis_dim = self.get_basis_from_content(
                item['name'], '', 'K')

            entity = KSAMDSEntity(
                id=ksamds_id, name=item['name'],
                source_ref="O*NET",
                type_dims=[type_dim], level_dims=[
                    level_dim], basis_dims=[basis_dim]
            )
            knowledge_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(knowledge_entities)} knowledge entities")
        return knowledge_entities

    def map_skill_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET skills data to KSAMDS skill entities."""
        if 'skills' not in self.onet_data:
            return {}

        logger.info("Mapping skill entities...")
        df = self.onet_data['skills']

        skill_items = []
        element_ratings = {}

        for _, row in df.iterrows():
            element_id = row.get('Element ID', '')
            scale_id = row.get('Scale ID', '')
            data_value = row.get('Data Value', 0)

            if element_id not in element_ratings:
                element_ratings[element_id] = {}
            element_ratings[element_id][scale_id] = data_value

            skill_items.append({
                'element_id': element_id,
                'name': row.get('Element Name', ''),
                'definition': row.get('Description', ''),
                'onet_soc_code': row.get('O*NET-SOC Code', '')
            })

        name_groups = defaultdict(list)
        for item in skill_items:
            clean_name = item['name'].strip()
            if clean_name:
                name_groups[clean_name].append(item)

        unique_skills = {}
        for skill_name, duplicate_items in name_groups.items():
            base_item = duplicate_items[0]
            all_element_ids = []
            best_definition = base_item['definition']

            for item in duplicate_items:
                if item['element_id']:
                    all_element_ids.append(item['element_id'])
                if len(item['definition']) > len(best_definition):
                    best_definition = item['definition']

            unique_skills[skill_name] = {
                'name': skill_name,
                'definition': best_definition,
                'primary_element_id': all_element_ids[0] if all_element_ids else '',
                'all_element_ids': list(set(all_element_ids))
            }

        logger.info(
            f"Deduplicated {len(skill_items)} to {len(unique_skills)} unique skills")

        skill_entities = {}
        for skill_name, item in unique_skills.items():
            ksamds_id = generate_deterministic_uuid('skill', skill_name)

            for element_id in item['all_element_ids']:
                if element_id:
                    self.onet_to_ksamds_ids[element_id] = ksamds_id

            primary_element_id = item['primary_element_id']
            type_dim = self.get_type_from_hierarchy(primary_element_id)
            lv_rating = element_ratings.get(
                primary_element_id, {}).get('LV', 3.0)
            level_dim = self.get_level_from_rating(lv_rating, 'S')
            basis_dim = self.get_basis_from_content(
                item['name'], '', 'S')

            entity = KSAMDSEntity(
                id=ksamds_id, name=item['name'],
                source_ref="O*NET",
                type_dims=[type_dim], level_dims=[
                    level_dim], basis_dims=[basis_dim]
            )
            skill_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(skill_entities)} skill entities")
        return skill_entities

    def map_ability_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET abilities data to KSAMDS ability entities."""
        if 'abilities' not in self.onet_data:
            return {}

        logger.info("Mapping ability entities...")
        df = self.onet_data['abilities']

        ability_items = []
        element_ratings = {}

        for _, row in df.iterrows():
            element_id = row.get('Element ID', '')
            scale_id = row.get('Scale ID', '')
            data_value = row.get('Data Value', 0)

            if element_id not in element_ratings:
                element_ratings[element_id] = {}
            element_ratings[element_id][scale_id] = data_value

            ability_items.append({
                'element_id': element_id,
                'name': row.get('Element Name', ''),
                'definition': row.get('Description', ''),
                'onet_soc_code': row.get('O*NET-SOC Code', '')
            })

        name_groups = defaultdict(list)
        for item in ability_items:
            clean_name = item['name'].strip()
            if clean_name:
                name_groups[clean_name].append(item)

        unique_abilities = {}
        for ability_name, duplicate_items in name_groups.items():
            base_item = duplicate_items[0]
            all_element_ids = []
            best_definition = base_item['definition']

            for item in duplicate_items:
                if item['element_id']:
                    all_element_ids.append(item['element_id'])
                if len(item['definition']) > len(best_definition):
                    best_definition = item['definition']

            unique_abilities[ability_name] = {
                'name': ability_name,
                'definition': best_definition,
                'primary_element_id': all_element_ids[0] if all_element_ids else '',
                'all_element_ids': list(set(all_element_ids))
            }

        logger.info(
            f"Deduplicated {len(ability_items)} to {len(unique_abilities)} unique abilities")

        ability_entities = {}
        for ability_name, item in unique_abilities.items():
            ksamds_id = generate_deterministic_uuid('ability', ability_name)

            for element_id in item['all_element_ids']:
                if element_id:
                    self.onet_to_ksamds_ids[element_id] = ksamds_id

            primary_element_id = item['primary_element_id']
            type_dim = self.get_type_from_hierarchy(primary_element_id)
            lv_rating = element_ratings.get(
                primary_element_id, {}).get('LV', 3.0)
            level_dim = self.get_level_from_rating(lv_rating, 'A')
            basis_dim = self.get_basis_from_content(
                item['name'], '', 'A')

            entity = KSAMDSEntity(
                id=ksamds_id, name=item['name'],
                source_ref="O*NET",
                type_dims=[type_dim], level_dims=[
                    level_dim], basis_dims=[basis_dim]
            )
            ability_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(ability_entities)} ability entities")
        return ability_entities

    def map_task_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET task data to KSAMDS task entities."""
        if 'task_ratings' not in self.onet_data and 'task_statements' not in self.onet_data:
            return {}

        logger.info("Mapping task entities...")
        task_items = []

        if 'task_ratings' in self.onet_data:
            df_ratings = self.onet_data['task_ratings']
            for _, row in df_ratings.iterrows():
                task_items.append({
                    'onet_id': f"TASK_{row.get('Task ID', '')}",
                    'name': row.get('Task', ''),
                    'description': row.get('Task', '')  # Keep for processing
                })

        if 'task_statements' in self.onet_data:
            df_statements = self.onet_data['task_statements']
            task_lookup = {item['onet_id']: item for item in task_items}
            for _, row in df_statements.iterrows():
                task_id = f"TASK_{row.get('Task ID', '')}"
                task_description = row.get('Task', '')
                if task_id in task_lookup:
                    if len(task_description) > len(task_lookup[task_id]['description']):
                        task_lookup[task_id]['description'] = task_description
                else:
                    task_items.append({
                        'onet_id': task_id,
                        'name': task_description[:100] + "..." if len(task_description) > 100 else task_description,
                        'description': task_description
                    })

        name_groups = defaultdict(list)
        for item in task_items:
            clean_name = item['name'].strip()
            if clean_name:
                name_groups[clean_name].append(item)

        unique_tasks = {}
        for task_name, duplicate_items in name_groups.items():
            base_item = duplicate_items[0]
            all_onet_ids = []
            best_description = base_item['description']

            for item in duplicate_items:
                if item['onet_id']:
                    all_onet_ids.append(item['onet_id'])
                if len(item['description']) > len(best_description):
                    best_description = item['description']

            unique_tasks[task_name] = {
                'name': task_name,
                'description': best_description,  # Keep for dimension inference
                'onet_ids': list(set(all_onet_ids))
            }

        logger.info(
            f"Deduplicated {len(task_items)} to {len(unique_tasks)} unique tasks")

        task_entities = {}
        for task_name, item in unique_tasks.items():
            ksamds_id = generate_deterministic_uuid('task', task_name)

            for onet_id in item['onet_ids']:
                if onet_id:
                    self.onet_to_ksamds_ids[onet_id] = ksamds_id

            # Infer type dimension from name/description
            if any(word in item['name'].lower() for word in ['analyze', 'calculate', 'research']):
                type_dim = 'Analytical'
            elif any(word in item['name'].lower() for word in ['operate', 'maintain', 'repair']):
                type_dim = 'Manual'
            elif any(word in item['name'].lower() for word in ['communicate', 'coordinate', 'supervise']):
                type_dim = 'Social'
            else:
                type_dim = 'Analytical'

            # Use description for dimension inference
            environment_dim = self.get_task_environment(
                item['name'], item['description'])
            mode_dim = self.get_task_mode(item['name'], item['description'])

            # Create entity WITHOUT definition/domain (won't be in CSV/DB)
            entity = KSAMDSEntity(
                id=ksamds_id,
                name=item['name'],
                source_ref="O*NET"
            )

            # Attach dimensions
            entity.type_dims = [type_dim]
            entity.environment_dims = [environment_dim]
            entity.mode_dims = [mode_dim]

            task_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(task_entities)} task entities")
        return task_entities

    def map_function_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET work activities to KSAMDS function entities."""
        if 'work_activities' not in self.onet_data:
            return {}

        logger.info("Mapping function entities...")
        df = self.onet_data['work_activities']

        function_items = []
        for _, row in df.iterrows():
            function_items.append({
                'element_id': row.get('Element ID', ''),
                'name': row.get('Element Name', ''),
                # Keep for processing
                'description': row.get('Description', '')
            })

        name_groups = defaultdict(list)
        for item in function_items:
            clean_name = item['name'].strip()
            if clean_name:
                name_groups[clean_name].append(item)

        unique_functions = {}
        for function_name, duplicate_items in name_groups.items():
            base_item = duplicate_items[0]
            all_element_ids = []
            best_description = base_item['description']

            for item in duplicate_items:
                if item['element_id']:
                    all_element_ids.append(item['element_id'])
                if len(item['description']) > len(best_description):
                    best_description = item['description']

            unique_functions[function_name] = {
                'name': function_name,
                'description': best_description,  # Keep for dimension inference
                'primary_element_id': all_element_ids[0] if all_element_ids else '',
                'all_element_ids': list(set(all_element_ids))
            }

        logger.info(
            f"Deduplicated {len(function_items)} to {len(unique_functions)} unique functions")

        function_entities = {}
        for function_name, item in unique_functions.items():
            ksamds_id = generate_deterministic_uuid('function', function_name)

            for element_id in item['all_element_ids']:
                if element_id:
                    self.onet_to_ksamds_ids[element_id] = ksamds_id

            # Use description for dimension inference
            environment_dim = self.get_function_environment(
                item['name'], item['description'])
            physicality_dim = self.get_function_physicality(
                item['name'], item['description'])
            cognitive_dim = self.get_function_cognitive(
                item['name'], item['description'])

            # Create entity WITHOUT definition/domain (won't be in CSV/DB)
            entity = KSAMDSEntity(
                id=ksamds_id,
                name=item['name'],
                source_ref="O*NET"
            )

            # Attach dimensions
            entity.environment_dims = [environment_dim]
            entity.physicality_dims = [physicality_dim]
            entity.cognitive_dims = [cognitive_dim]

            function_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(function_entities)} function entities")
        return function_entities

    def map_education_levels(self) -> Dict[str, KSAMDSEntity]:
        """Map education levels from job zones."""
        logger.info("Mapping education levels...")

        education_levels = [
            ("High School", "High school diploma or equivalent"),
            ("Some College", "Some post-secondary education or training"),
            ("Associate Degree", "Associate degree or equivalent training"),
            ("Bachelor's Degree", "Bachelor's degree"),
            ("Master's Degree", "Master's degree"),
            ("Professional Degree", "Professional degree or certification"),
            ("Doctoral Degree", "Doctoral or professional degree")
        ]

        education_entities = {}
        for name, description in education_levels:
            ksamds_id = generate_deterministic_uuid('education_level', name)
            entity = KSAMDSEntity(
                id=ksamds_id, name=name, source_ref="O*NET"
            )
            education_entities[ksamds_id] = entity

        logger.info(f"Mapped {len(education_entities)} education levels")
        return education_entities

    def save_mapped_entities(self) -> bool:
        """Save all mapped entities to CSV files."""
        try:
            for entity_type, entities in self.ksamds_entities.items():
                if not entities:
                    continue

                rows = []
                for entity_id, entity in entities.items():
                    row = {
                        'id': entity.id,
                        'name': entity.name,
                        'source_ref': 'O*NET'
                    }

                    # Special handling for occupations (has description field)
                    if entity_type == 'occupation' and hasattr(entity, 'description'):
                        row['description'] = entity.description

                    if hasattr(entity, 'type_dims') and entity.type_dims:
                        row['type_dims'] = "|".join(entity.type_dims)
                    if hasattr(entity, 'level_dims') and entity.level_dims:
                        row['level_dims'] = "|".join(entity.level_dims)
                    if hasattr(entity, 'basis_dims') and entity.basis_dims:
                        row['basis_dims'] = "|".join(entity.basis_dims)
                    if hasattr(entity, 'environment_dims') and entity.environment_dims:
                        row['environment_dims'] = "|".join(
                            entity.environment_dims)
                    if hasattr(entity, 'mode_dims') and entity.mode_dims:
                        row['mode_dims'] = "|".join(entity.mode_dims)
                    if hasattr(entity, 'physicality_dims') and entity.physicality_dims:
                        row['physicality_dims'] = "|".join(
                            entity.physicality_dims)
                    if hasattr(entity, 'cognitive_dims') and entity.cognitive_dims:
                        row['cognitive_dims'] = "|".join(entity.cognitive_dims)

                    rows.append(row)

                df = pd.DataFrame(rows)
                csv_path = self.mapped_dir / f"{entity_type}_mapped.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(rows)} {entity_type} entities")

            mapping_df = pd.DataFrame([
                {'onet_id': onet_id, 'ksamds_id': ksamds_id}
                for onet_id, ksamds_id in self.onet_to_ksamds_ids.items()
            ])
            mapping_path = self.mapped_dir / "id_mappings.csv"
            mapping_df.to_csv(mapping_path, index=False)
            logger.info(f"Saved {len(mapping_df)} ID mappings")

            if hasattr(self, 'ksamds_relationships') and self.ksamds_relationships:
                for rel_type, relationships in self.ksamds_relationships.items():
                    if relationships:
                        rel_df = pd.DataFrame(relationships, columns=[
                                              'entity1_id', 'entity2_id'])
                        rel_path = self.mapped_dir / \
                            f"{rel_type}_relationships.csv"
                        rel_df.to_csv(rel_path, index=False)
                        logger.info(
                            f"Saved {len(relationships)} {rel_type} relationships")

            return True
        except Exception as e:
            logger.error(f"Failed to save mapped entities: {e}")
            return False

    def map_occupation_relationships(self) -> Dict[str, List[Tuple[str, str]]]:
        """Map relationships between occupations and K/S/A/T/F using KSAMDS IDs."""
        logger.info("Mapping occupation relationships...")

        relationships = {
            'occupation_knowledge': [], 'occupation_skill': [],
            'occupation_ability': [], 'occupation_task': [],
            'occupation_function': [], 'occupation_education': []
        }

        if 'knowledge' in self.onet_data:
            for _, row in self.onet_data['knowledge'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                knowledge_ksamds_id = self.onet_to_ksamds_ids.get(element_id)
                if occupation_ksamds_id and knowledge_ksamds_id:
                    relationship = (occupation_ksamds_id, knowledge_ksamds_id)
                    if relationship not in relationships['occupation_knowledge']:
                        relationships['occupation_knowledge'].append(
                            relationship)

        if 'skills' in self.onet_data:
            for _, row in self.onet_data['skills'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                skill_ksamds_id = self.onet_to_ksamds_ids.get(element_id)
                if occupation_ksamds_id and skill_ksamds_id:
                    relationship = (occupation_ksamds_id, skill_ksamds_id)
                    if relationship not in relationships['occupation_skill']:
                        relationships['occupation_skill'].append(relationship)

        if 'abilities' in self.onet_data:
            for _, row in self.onet_data['abilities'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                ability_ksamds_id = self.onet_to_ksamds_ids.get(element_id)
                if occupation_ksamds_id and ability_ksamds_id:
                    relationship = (occupation_ksamds_id, ability_ksamds_id)
                    if relationship not in relationships['occupation_ability']:
                        relationships['occupation_ability'].append(
                            relationship)

        if 'task_ratings' in self.onet_data:
            for _, row in self.onet_data['task_ratings'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                task_id = f"TASK_{row.get('Task ID', '')}"
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                task_ksamds_id = self.onet_to_ksamds_ids.get(task_id)
                if occupation_ksamds_id and task_ksamds_id:
                    relationship = (occupation_ksamds_id, task_ksamds_id)
                    if relationship not in relationships['occupation_task']:
                        relationships['occupation_task'].append(relationship)

        if 'work_activities' in self.onet_data:
            for _, row in self.onet_data['work_activities'].iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                function_ksamds_id = self.onet_to_ksamds_ids.get(element_id)
                if occupation_ksamds_id and function_ksamds_id:
                    relationship = (occupation_ksamds_id, function_ksamds_id)
                    if relationship not in relationships['occupation_function']:
                        relationships['occupation_function'].append(
                            relationship)

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

    def map_all_entities(self) -> bool:
        """Map all O*NET entities to KSAMDS structure."""
        logger.info("Starting complete O*NET to KSAMDS mapping...")

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

        logger.info("O*NET to KSAMDS mapping completed successfully!")
        return True

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of mapping results."""
        summary = {
            'entities': {}, 'relationships': {},
            'hierarchy_mappings': len(self.hierarchy_mappings),
            'id_mappings': len(self.onet_to_ksamds_ids),
            'job_zones': len(self.job_zone_lookup)
        }

        for entity_type, entities in self.ksamds_entities.items():
            summary['entities'][entity_type] = len(entities)

        for rel_type, relationships in self.ksamds_relationships.items():
            summary['relationships'][rel_type] = len(relationships)

        return summary


def main():
    """Main function to run the O*NET mapper."""
    mapper = ONetMapper()
    success = mapper.map_all_entities()

    if success:
        summary = mapper.get_mapping_summary()
        print("\n=== O*NET to KSAMDS Mapping Summary ===")
        print(f"Hierarchy mappings: {summary['hierarchy_mappings']}")
        print(f"ID mappings: {summary['id_mappings']}")
        print(f"\n=== Entity Mapping Results ===")
        for entity_type, count in summary['entities'].items():
            if count > 0:
                print(f"{entity_type.upper()}: {count} entities mapped")
        print(f"\n=== Relationship Mapping Results ===")
        for rel_type, count in summary['relationships'].items():
            if count > 0:
                print(f"{rel_type.upper()}: {count} relationships mapped")
        print(f"\nMapped data saved to: {mapper.mapped_dir}")
    else:
        print("Mapping failed. Check logs for details.")


if __name__ == "__main__":
    main()
