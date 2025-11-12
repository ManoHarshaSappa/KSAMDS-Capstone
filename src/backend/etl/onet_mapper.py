"""
O*NET Data Mapper for KSAMDS Project - DETERMINISTIC VERSION WITH DIMENSION MAPPING

This module handles the complex mapping and transformation of O*NET data
into the KSAMDS multi-dimensional structure. It now integrates pre-generated
embeddings and nearest-neighbor search logic to populate dimension fields
(Basis, Type, Level, Mode, Environment, Physicality, Cognitive) directly
in the occupation relationship tables, based on the methodology defined in
onet_occupation_features.py and onet_embedding_generator.py.

FIXED:
1. Ensure OccupationRelationship CSVs (Task, Function) save all attributes,
   including importance_score for Function.
2. Corrected saving loop to ensure Occupation_Skills and Occupation_Abilities
   relationship files are created.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import re
from dataclasses import dataclass, field
from fuzzywuzzy import fuzz  # type: ignore
from collections import defaultdict
import uuid
import shutil
import itertools

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
    """
    unique_identifier = f"{entity_type.lower()}:{name.strip()}"
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
    """Represents an occupation relationship with all dimensions."""
    occupation_id: str
    entity_id: str
    # KSA Dimensions
    level: Optional[str] = None
    basis: Optional[str] = None
    type: Optional[str] = None # KSA Type
    importance_score: Optional[float] = None
    # Task Dimensions
    mode: Optional[str] = None
    # Task/Function Environment (Scope F or T)
    environment: Optional[str] = None
    # Function Dimensions
    physicality: Optional[str] = None
    cognitive: Optional[str] = None


class ONetMapper:
    """Map O*NET data to KSAMDS multi-dimensional structure."""

    def __init__(self):
        """Initialize the O*NET mapper."""
        # Get project root directory (3 levels up from etl folder)
        project_root = Path(__file__).resolve().parents[3]

        # Setup directory structure
        self.processed_dir = project_root / "data/archive/intermediate"
        self.mapped_dir = project_root / "data/archive/mapped"
        self.embeddings_dir = project_root / "data/archive/embeddings"

        self.mapped_dir.mkdir(parents=True, exist_ok=True)

        self.onet_data: Dict[str, pd.DataFrame] = {}
        self.ksamds_entities = {
            'knowledge': {}, 'skill': {}, 'ability': {},
            'occupation': {}, 'function': {}, 'task': {},
            'education_level': {}, 'certification': {}
        }
        self.onet_to_ksamds_ids: Dict[str, str] = {}
        self.ksamds_relationships = defaultdict(list)
        self.job_zone_lookup: Dict[str, int] = {}
        self.similarity_threshold = 85

        # Loaded embeddings and taxonomy data
        self.taxonomy_embeds: Dict[str, Tuple[Optional[np.ndarray], List[str]]] = {}
        self.query_data: Dict[str, Any] = {}
        self.model_name_suffix = "embedding-001" # Default suffix from onet_embedding_generator.py

        # Data structure for KSA Type mapping (for pre-computed entities)
        self.hierarchy_mappings: Dict[str, str] = {}
        
        # New: Track entity-level mappings for occupation relationships
        self.occupation_entity_levels: Dict[Tuple[str, str], str] = {}


    # --- Embedding and Utility Functions (Copied/Adapted from onet_occupation_features.py) ---

    def find_best_match(self, query_embedding: List[float], document_embeddings: np.ndarray, document_names: List[str]) -> str:
        """Finds the best match using cosine similarity."""
        if query_embedding is None or document_embeddings is None:
            return "Embedding_Failed"

        query_norm = np.array(query_embedding) / np.linalg.norm(query_embedding)
        
        # Guard against zero-norm documents (shouldn't happen with the model, but for safety)
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1e-12 # Prevent division by zero
        
        doc_normalized = document_embeddings / doc_norms

        similarities = np.dot(doc_normalized, query_norm)
        best_index = np.argmax(similarities)
        return document_names[best_index]

    def _load_onet_embeddings(self) -> bool:
        """Loads all necessary embeddings and query data from cache files."""
        logger.info("=" * 70)
        logger.info("LOADING EMBEDDINGS AND QUERY DATA")
        logger.info("=" * 70)
        
        # 1. Determine model suffix for file names (A bit brittle, assumes default)
        taxonomy_cache_file = self.embeddings_dir / f"taxonomy_embeddings_{self.model_name_suffix}.pkl"

        # 2. Load Taxonomy Embeddings
        try:
            with open(taxonomy_cache_file, 'rb') as f:
                self.taxonomy_embeds = pickle.load(f)
            logger.info(f"Successfully loaded taxonomy embeddings from {taxonomy_cache_file.name}")
        except Exception as e:
            logger.error(f"Failed to load taxonomy embeddings: {e}")
            return False
            
        # 3. Load Query Embeddings (KSA, Function, Task)
        query_files = ['knowledge', 'skills', 'abilities', 'function', 'task']
        for key in query_files:
            cache_file = self.embeddings_dir / f"{key}_query_data_{self.model_name_suffix}.pkl"
            try:
                with open(cache_file, 'rb') as f:
                    self.query_data[key] = pickle.load(f)
                logger.info(f"Loaded {key} query data from {cache_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {key} query data: {e}")
                return False
                
        # 4. Prepare lookups from query data (Element ID or Task Text -> Embeddings/Data)
        self._prepare_query_lookups()
        
        return True

    def _prepare_query_lookups(self):
        """
        Transforms loaded query data into efficient lookups:
        - KSA/Function: (O*NET-SOC Code, Element ID) -> (Query Embedding, Row Data)
        - Task: (O*NET-SOC Code, Task Text) -> (Query Embedding, Task Type)
        """
        self.query_lookups = defaultdict(dict)
        
        # --- KSA and Function Lookups ---
        for key in ['knowledge', 'skills', 'abilities', 'function']:
            if key in self.query_data:
                # The 'valid_rows' in onet_occupation_features.py contains the row data for all valid queries
                embeds = self.query_data[key]['embeddings']
                # FIX: Use 'valid_rows' key instead of 'valid_rows_data'
                rows_data = self.query_data[key]['valid_rows'] 
                
                for emb, row in zip(embeds, rows_data):
                    # KSA/Function uses Element ID as a unique identifier per SOC
                    soc_code = row['O*NET-SOC Code']
                    element_id = row['Element ID']
                    # Store (embedding, row_data) keyed by (SOC_Code, Element_ID)
                    self.query_lookups[key][(soc_code, element_id)] = (emb, row)
                    
        # --- Task Lookup ---
        if 'task' in self.query_data:
            # The 'valid_tasks_info' structure is correct for tasks
            embeds = self.query_data['task']['embeddings']
            tasks_info = self.query_data['task']['valid_tasks_info']
            
            # The Task Statement DF doesn't contain Element ID, only Task Text.
            # We need the task statement DF to get the Task ID for the join key.
            task_df = self.onet_data.get('task_statements')
            if task_df is None:
                logger.error("Task statements data not loaded, cannot build task lookups.")
                return

            # A quick way to get SOC code from title:
            occupation_df = self.onet_data.get('occupation_data')
            if occupation_df is not None:
                # Create a Title to SOC Code lookup (for many-to-one mapping)
                title_to_soc = pd.Series(occupation_df['O*NET-SOC Code'].values, index=occupation_df['Title']).to_dict()
                
                for emb, (title, task_text, task_type) in zip(embeds, tasks_info):
                    soc_code = title_to_soc.get(title)
                    if soc_code:
                        # Store (embedding, task_type) keyed by (SOC_Code, Task Text)
                        self.query_lookups['task'][(soc_code, task_text)] = (emb, task_type)
                    else:
                         logger.warning(f"Could not find SOC code for title: {title}")

    def load_onet_data(self) -> bool:
        """Load processed O*NET CSV files."""
        # ... (Same as original)
        logger.info("=" * 70)
        logger.info("LOADING O*NET DATA")
        logger.info("=" * 70)

        required_files = ['knowledge', 'skills', 'abilities', 'occupation_data',
                          'task_statements', 'work_activities',
                          'content_model_reference', 'job_zones', 'task_ratings']

        try:
            for file_key in required_files:
                csv_path = self.processed_dir / f"{file_key}.csv"
                if csv_path.exists():
                    # Use low_memory=False for large files
                    self.onet_data[file_key] = pd.read_csv(csv_path, low_memory=False)
                    logger.info(
                        f"Loaded {file_key}: {len(self.onet_data[file_key])} records")
                else:
                    logger.error(f"Required file not found: {csv_path}")
                    return False

            # Load optional files
            optional_files = ['scales_reference',
                              'education_training_experience']
            for file_key in optional_files:
                csv_path = self.processed_dir / f"{file_key}.csv"
                if csv_path.exists():
                    self.onet_data[file_key] = pd.read_csv(csv_path, low_memory=False)
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

        # KSA Type mappings from Element ID prefix
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
        """Get type dimension from O*NET hierarchy (for KSA entities)."""
        for hierarchy_prefix, ksamds_type in self.hierarchy_mappings.items():
            if element_id.startswith(hierarchy_prefix):
                return ksamds_type
        return 'General'

    def map_level_value_to_category(self, lv_value: float, scope: str) -> str:
        """
        Map O*NET Level (LV) scale value to KSAMDS level category (replicating onet_occupation_features.py logic).
        """
        if pd.isna(lv_value):
            # Fallback to middle level if no data
            if scope == 'K':
                return 'Intermediate'
            elif scope == 'S':
                return 'Proficient'
            else:  # 'A'
                return 'Moderate'

        # Map based on scope (Logic copied from onet_occupation_features.py)
        if scope == 'K':  # Knowledge: Basic, Intermediate, Advanced, Expert
            # NOTE: The feature generation script uses different cutoffs: (0-1.99 basic, 2-3.99 intermediate, 4-7 advanced)
            # We will use the explicit cutoffs from the feature script for consistency in the relationship table.
            if lv_value < 2.0:
                return 'Basic'
            elif lv_value < 4.0:
                return 'Intermediate'
            else:
                return 'Advanced' # Changed from Expert to Advanced for consistency with onet_occupation_features.py

        elif scope == 'S':  # Skills: Novice, Proficient, Expert, Master
            # NOTE: The feature generation script uses different cutoffs: (0-1.99 basic, 2-3.99 intermediate, 4-6 advanced)
            if lv_value < 2.0:
                return 'Basic' # Changed from Novice to Basic for consistency with onet_occupation_features.py
            elif lv_value < 4.0:
                return 'Intermediate' # Changed from Proficient to Intermediate for consistency with onet_occupation_features.py
            else:
                return 'Advanced' # Changed from Expert/Master to Advanced for consistency with onet_occupation_features.py

        else:  # 'A' - Abilities: Low, Moderate, High
            # NOTE: The feature generation script uses different cutoffs: (0-1.99 basic, 2-3.99 intermediate, 4-6 advanced)
            if lv_value < 2.0:
                return 'Basic' # Changed from Low to Basic for consistency with onet_occupation_features.py
            elif lv_value < 4.0:
                return 'Intermediate' # Changed from Moderate to Intermediate for consistency with onet_occupation_features.py
            else:
                return 'Advanced' # Changed from High to Advanced for consistency with onet_occupation_features.py

    def get_basis_from_job_zone(self, job_zone: int) -> List[str]:
        """Map job zone to basis dimensions (Used for Entity Basis Dims)."""
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
        # ... (Same as original)
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
                        str(name).lower(), str(other_name).lower())
                    if similarity_score >= self.similarity_threshold:
                        similar_group.append(other_name)
                        processed.add(other_name)

            for similar_name in similar_group:
                name_to_canonical[similar_name] = canonical_name
            processed.add(name)

        return name_to_canonical

    def map_occupation_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET occupation data to KSAMDS occupation entities."""
        # ... (Same as original)
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
            description = row.get('Description', None) # Get the description
            # Generate UUID based on occupation TITLE, not SOC code
            entity_id = generate_deterministic_uuid('occupation', title)
            self.onet_to_ksamds_ids[onet_soc] = entity_id

            occupation = KSAMDSEntity(
                id=entity_id, name=title, source_ref=onet_soc,
                type_dims=[description] if pd.notna(description) else [])
            occupations[entity_id] = occupation

        logger.info(f"Mapped {len(occupations)} occupation entities")
        return occupations

    def map_knowledge_entities(self) -> Dict[str, KSAMDSEntity]:
        """
        Map O*NET knowledge data to KSAMDS knowledge entities.
        (Kept for Entity generation, but Relationship mapping is now the source of truth for dimensions)
        """
        logger.info("=" * 70)
        logger.info("MAPPING KNOWLEDGE ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'knowledge' not in self.onet_data:
            logger.error("Knowledge data not loaded")
            return {}

        df = self.onet_data['knowledge']

        # Filter for IM and Data Value LV columns (assuming enrichment from extractor)
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]

        # Group by Element Name/ID to get unique entities
        unique_entities = df_filtered.groupby(['Element ID', 'Element Name']).agg({
            'Data Value LV': 'mean' # Use mean level for the entity's level
        }).reset_index()

        name_to_canonical = self.map_similar_entities(unique_entities)

        knowledge_entities = {}
        for _, row in unique_entities.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('knowledge', canonical_name)
            self.onet_to_ksamds_ids[element_id] = entity_id

            if entity_id not in knowledge_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                level_category = self.map_level_value_to_category(lv_value, 'K') # Use mean level

                knowledge = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                    # Basis is not derived here, will be populated on relationships
                )
                knowledge_entities[entity_id] = knowledge

        logger.info(
            f"Mapped {len(knowledge_entities)} unique knowledge entities (Entity Level based on mean LV)")
        return knowledge_entities

    def map_skill_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET skills data to KSAMDS skill entities (Entity Level based on mean LV)."""
        logger.info("=" * 70)
        logger.info("MAPPING SKILL ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'skills' not in self.onet_data:
            logger.error("Skills data not loaded")
            return {}

        df = self.onet_data['skills']
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]
        unique_entities = df_filtered.groupby(['Element ID', 'Element Name']).agg({
            'Data Value LV': 'mean'
        }).reset_index()
        name_to_canonical = self.map_similar_entities(unique_entities)
        skill_entities = {}

        for _, row in unique_entities.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('skill', canonical_name)
            self.onet_to_ksamds_ids[element_id] = entity_id

            if entity_id not in skill_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                level_category = self.map_level_value_to_category(lv_value, 'S')

                skill = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                )
                skill_entities[entity_id] = skill

        logger.info(f"Mapped {len(skill_entities)} unique skill entities")
        return skill_entities

    def map_ability_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET abilities data to KSAMDS ability entities (Entity Level based on mean LV)."""
        logger.info("=" * 70)
        logger.info("MAPPING ABILITY ENTITIES WITH LEVELS")
        logger.info("=" * 70)

        if 'abilities' not in self.onet_data:
            logger.error("Abilities data not loaded")
            return {}

        df = self.onet_data['abilities']
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]
        unique_entities = df_filtered.groupby(['Element ID', 'Element Name']).agg({
            'Data Value LV': 'mean'
        }).reset_index()
        name_to_canonical = self.map_similar_entities(unique_entities)
        ability_entities = {}

        for _, row in unique_entities.iterrows():
            element_name = row.get('Element Name', '')
            element_id = row.get('Element ID', '')
            lv_value = row.get('Data Value LV', None)

            canonical_name = name_to_canonical.get(element_name, element_name)
            entity_id = generate_deterministic_uuid('ability', canonical_name)
            self.onet_to_ksamds_ids[element_id] = entity_id

            if entity_id not in ability_entities:
                ksamds_type = self.get_type_from_hierarchy(element_id)
                level_category = self.map_level_value_to_category(lv_value, 'A')

                ability = KSAMDSEntity(
                    id=entity_id,
                    name=canonical_name,
                    source_ref=element_id,
                    type_dims=[ksamds_type],
                    level_dims=[level_category],
                )
                ability_entities[entity_id] = ability

        logger.info(f"Mapped {len(ability_entities)} unique ability entities")
        return ability_entities

    def map_task_entities(self) -> Dict[str, KSAMDSEntity]:
        """Map O*NET task statements to KSAMDS task entities."""
        # ... (Same as original, used for ID mapping)
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
            # Need to create a unique ONET ID for tasks from SOC Code and Task ID
            task_id_onet = f"TASK_{row.get('O*NET-SOC Code', '')}_{row.get('Task ID', '')}"

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
        # ... (Same as original, used for ID mapping)
        logger.info("=" * 70)
        logger.info("MAPPING FUNCTION ENTITIES")
        logger.info("=" * 70)

        if 'work_activities' not in self.onet_data:
            logger.error("Work activities not loaded")
            return {}

        df = self.onet_data['work_activities']
        df_filtered = df[(df['Scale ID'] == 'IM') & (df['Data Value'] >= 3.0)]

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
        # ... (Same as original)
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
            """Save all mapped entities and relationships to CSV files (MODIFIED for robust relationship schema)."""
            logger.info("=" * 70)
            logger.info("SAVING MAPPED ENTITIES")
            logger.info("=" * 70)
            logger.info(f"Relationship keys: {list(self.ksamds_relationships.keys())}")
            for k, v in self.ksamds_relationships.items():
                logger.info(f"{k}: {len(v)} items")

            try:
                # Save core entities (Entity files like knowledge_mapped.csv, occupation_mapped.csv, etc.)
                for entity_type, entities in self.ksamds_entities.items():
                    if not entities:
                        continue
                    rows = []
                    for entity in entities.values():
                        description = ''
                        if entity_type == 'occupation' and entity.type_dims:
                            # Retrieve the description we stored in type_dims
                            description = entity.type_dims[0]
                        row = {
                            'id': entity.id,
                            'name': entity.name,
                            'source_ref': entity.source_ref,
                            'description': description,
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
                        f"Saved {len(rows)} {entity_type} entities to {csv_path.name}")

                # Save relationships (MODIFIED for robust column handling and loop logic)
                for rel_type, relationships in self.ksamds_relationships.items():
                    if not relationships:
                        continue
                    
                    rows = []
                    
                    # 1. Handle OccupationRelationship types (KSA, Task, Function)
                    if len(relationships) > 0 and isinstance(relationships[0], OccupationRelationship):
                        
                        if rel_type in ['occupation_knowledge', 'occupation_skills', 'occupation_abilities']:
                            # KSA Relationships (All share the same dimension columns)
                            for rel in relationships:
                                rows.append({
                                    'occupation_id': rel.occupation_id,
                                    'entity_id': rel.entity_id,
                                    'level': rel.level if rel.level else '',
                                    'basis': rel.basis if rel.basis else '',
                                    'type': rel.type if rel.type else '',
                                    'importance_score': rel.importance_score if rel.importance_score is not None else '' # FIX: Check against None
                                })
                        
                        elif rel_type == 'occupation_task':
                            # Task Relationships
                            for rel in relationships:
                                rows.append({
                                    'occupation_id': rel.occupation_id,
                                    'entity_id': rel.entity_id,
                                    'mode': rel.mode if rel.mode else '',
                                    'environment': rel.environment if rel.environment else '',
                                    'type': rel.type if rel.type else '', # Task Type
                                    'importance_score': rel.importance_score if rel.importance_score is not None else ''
                                })
                                
                        elif rel_type == 'occupation_function':
                            # Function Relationships
                            for rel in relationships:
                                rows.append({
                                    'occupation_id': rel.occupation_id,
                                    'entity_id': rel.entity_id,
                                    'physicality': rel.physicality if rel.physicality else '',
                                    'cognitive': rel.cognitive if rel.cognitive else '',
                                    'environment': rel.environment if rel.environment else '', # Function Environment
                                    'importance_score': rel.importance_score if rel.importance_score is not None else '' # FIX: Added importance_score
                                })
                    
                    # 2. Handle Simple Tuple Relationships (e.g., occupation_education)
                    else:
                        rows = relationships
                        
                    # 3. Save the DataFrame
                    if rows:
                        df = pd.DataFrame(rows)
                        # Rename columns for simple tuple relationships
                        if not isinstance(relationships[0], OccupationRelationship) and df.columns[0] == 0:
                             df.rename(columns={0: 'source_id', 1: 'target_id'}, inplace=True)

                        # FIX: Renamed rel_type for Skills and Abilities to match loader expectation
                        if rel_type == 'occupation_skills':
                            csv_name = 'occupation_skills_relationships.csv'
                        elif rel_type == 'occupation_abilities':
                            csv_name = 'occupation_abilities_relationships.csv'
                        else:
                            csv_name = f"{rel_type}_relationships.csv"
                            
                        csv_path = self.mapped_dir / csv_name
                        df.to_csv(csv_path, index=False)
                        logger.info(
                            f"Saved {len(relationships)} {rel_type} relationships to {csv_path.name}")

                logger.info("All mapped data saved successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to save mapped entities: {e}")
                return False

    def map_occupation_relationships(self) -> Dict[str, List]:
        """
        Map occupation relationships to KSA/Function/Task, populating dimensions
        using the loaded embeddings and query data. (HEAVILY MODIFIED)
        """
        logger.info("=" * 70)
        logger.info("MAPPING OCCUPATION RELATIONSHIPS WITH ALL DIMENSIONS")
        logger.info("=" * 70)
        task_ratings_df = self.onet_data.get('task_ratings')
        relationships = defaultdict(list)
        
        # --- KSA RELATIONSHIPS (with all dimensions) ---
        for entity_type in ['knowledge', 'skills', 'abilities']:
            df_key = entity_type
            rel_type = f'occupation_{df_key}'
            scope = df_key[0].upper()
            basis_embeds, basis_names = self.taxonomy_embeds[f'{scope.lower()}_basis']

            if df_key in self.onet_data:
                # Filter is already implicitly applied by the query data lookup (IM >= 3.0)
                df_filtered = self.onet_data[df_key][(self.onet_data[df_key]['Scale ID'] == 'IM') &
                                                     (self.onet_data[df_key]['Data Value'] >= 3.0)]
                logger.info(f"Processing {len(df_filtered)} {df_key} relationships")

                for _, row in df_filtered.iterrows():
                    onet_soc = row.get('O*NET-SOC Code', '')
                    element_id = row.get('Element ID', '')
                    importance_score = row.get('Data Value', None)

                    # Lookup pre-embedded query and row data
                    query_lookup_key = (onet_soc, element_id)
                    lookup_result = self.query_lookups[df_key].get(query_lookup_key)

                    if lookup_result:
                        query_emb, row_data = lookup_result
                        
                        # NOTE: The correct importance_score for KSA is already in the original row/variable.
                        # This block (lines 789-797) was incorrectly inserted from Task logic and must be removed.
                        
                        occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                        entity_ksamds_id = self.onet_to_ksamds_ids.get(element_id)

                        if occupation_ksamds_id and entity_ksamds_id:
                            # 1. Level (Derived from LV score in the query row data)
                            lv_value = row_data.get('LV', np.nan)
                            level_category = self.map_level_value_to_category(lv_value, scope)
                            
                            # 2. Basis (Derived from embedding match)
                            basis_name = self.find_best_match(query_emb, basis_embeds, basis_names)
                            
                            # 3. Type (Derived from O*NET hierarchy in onet_mapper.py)
                            type_name = self.get_type_from_hierarchy(element_id)

                            rel = OccupationRelationship(
                                occupation_id=occupation_ksamds_id,
                                entity_id=entity_ksamds_id,
                                level=level_category,
                                basis=basis_name,
                                type=type_name,
                                importance_score=importance_score # Use the score from line 778
                            )
                            # Avoid duplicates by checking key columns only
                            if not any(r.occupation_id == rel.occupation_id and r.entity_id == rel.entity_id
                                       for r in relationships[rel_type]):
                                relationships[rel_type].append(rel)

        # --- FUNCTION RELATIONSHIPS (with dimensions) ---
        rel_type = 'occupation_function'
        f_phys_embeds, f_phys_names = self.taxonomy_embeds['f_phys']
        f_cog_embeds, f_cog_names = self.taxonomy_embeds['f_cog']
        f_env_embeds, f_env_names = self.taxonomy_embeds['f_env']
        
        if 'work_activities' in self.onet_data:
            # Filter is already implicitly applied by the query data lookup (IM >= 3.0)
            df_filtered = self.onet_data['work_activities'][(self.onet_data['work_activities']['Scale ID'] == 'IM') &
                                                            (self.onet_data['work_activities']['Data Value'] >= 3.0)]
            logger.info(f"Processing {len(df_filtered)} function relationships")

            for _, row in df_filtered.iterrows():
                onet_soc = row.get('O*NET-SOC Code', '')
                element_id = row.get('Element ID', '')
                importance_score = row.get('Data Value', None)

                query_lookup_key = (onet_soc, element_id)
                lookup_result = self.query_lookups['function'].get(query_lookup_key)

                if lookup_result:
                    query_emb, _ = lookup_result # No extra row data needed for functions
                    
                    occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                    function_ksamds_id = self.onet_to_ksamds_ids.get(element_id)

                    if occupation_ksamds_id and function_ksamds_id:
                        # Find dimensions using embeddings
                        phys_name = self.find_best_match(query_emb, f_phys_embeds, f_phys_names)
                        cog_name = self.find_best_match(query_emb, f_cog_embeds, f_cog_names)
                        env_name = self.find_best_match(query_emb, f_env_embeds, f_env_names)

                        rel = OccupationRelationship(
                            occupation_id=occupation_ksamds_id,
                            entity_id=function_ksamds_id,
                            physicality=phys_name,
                            cognitive=cog_name,
                            environment=env_name,
                            importance_score=importance_score
                        )
                        if not any(r.occupation_id == rel.occupation_id and r.entity_id == rel.entity_id
                                   for r in relationships[rel_type]):
                            relationships[rel_type].append(rel)

        # --- TASK RELATIONSHIPS (with dimensions) ---
        rel_type = 'occupation_task'
        t_mode_embeds, t_mode_names = self.taxonomy_embeds['t_mode']
        t_env_embeds, t_env_names = self.taxonomy_embeds['t_env']
        
        if 'task_statements' in self.onet_data:
            task_df = self.onet_data['task_statements']
            logger.info(f"Processing {len(task_df)} task relationships")
            
            # The query lookup contains only IM >= 3.0 tasks, so we iterate over the task lookup keys
            for (onet_soc, task_text), lookup_result in self.query_lookups['task'].items():
                query_emb, task_type = lookup_result
                
                # Find the unique O*NET task ID (SOC+Task ID) that corresponds to this SOC and Task Text
                # Use a merged dataframe or direct lookup if needed, but for simplicity, we rely on the
                # unique task text for a given SOC to map back to the unique Task entity ID.
                # Since the task entity ID was created using (SOC_Code, Task ID), we need to find the Task ID.
                
                task_match = task_df[(task_df['O*NET-SOC Code'] == onet_soc) & 
                                     (task_df['Task'] == task_text)]
                                     
                if not task_match.empty:
                    task_id = task_match.iloc[0]['Task ID']
                    task_id_onet = f"TASK_{onet_soc}_{task_id}"

                    # --- START OF MODIFICATION ---
                    # Look up the importance score from task_ratings
                    importance_score = None
                    if task_ratings_df is not None:
                        # Find the 'IM' (Importance) score
                        rating_row = task_ratings_df[
                            (task_ratings_df['O*NET-SOC Code'] == onet_soc) &
                            (task_ratings_df['Task ID'] == task_id) &
                            (task_ratings_df['Scale ID'] == 'IM')
                        ]
                        if not rating_row.empty:
                            importance_score = rating_row.iloc[0]['Data Value']
                    # --- END OF MODIFICATION ---

                    occupation_ksamds_id = self.onet_to_ksamds_ids.get(onet_soc)
                    task_ksamds_id = self.onet_to_ksamds_ids.get(task_id_onet)
                    
                    if occupation_ksamds_id and task_ksamds_id:
                        # 1. Type (Directly from task_type in query data)
                        type_name = task_type
                        
                        # 2. Mode and Environment (Derived from embedding match)
                        mode_name = self.find_best_match(query_emb, t_mode_embeds, t_mode_names)
                        env_name = self.find_best_match(query_emb, t_env_embeds, t_env_names)

                        rel = OccupationRelationship(
                            occupation_id=occupation_ksamds_id,
                            entity_id=task_ksamds_id,
                            type=type_name,
                            mode=mode_name,
                            environment=env_name,
                            importance_score=importance_score # <--MODIFIED
                        )
                        if not any(r.occupation_id == rel.occupation_id and r.entity_id == rel.entity_id
                                   for r in relationships[rel_type]):
                            relationships[rel_type].append(rel)
                else:
                    logger.warning(f"Could not find Task ID for SOC: {onet_soc} and Task Text: {task_text[:30]}...")


        # --- EDUCATION RELATIONSHIPS (No change) ---
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
        # ... (Same as original)
        logger.info("Cleaning up intermediate O*NET CSV files...")

        # Clean up the intermediate directory
        if self.processed_dir.exists() and self.processed_dir.parts[-1] == 'intermediate':
            try:
                file_count = len(list(self.processed_dir.glob('*.csv')))
                # Do not remove the embeddings folder
                shutil.rmtree(self.processed_dir)
                logger.info(
                    f"Removed {file_count} intermediate CSV files from {self.processed_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up intermediate files: {e}")

    def map_all_entities(self, cleanup_after: bool = False) -> bool:
        """
        Map all O*NET entities to KSAMDS structure.
        """
        logger.info("=" * 70)
        logger.info("STARTING O*NET TO KSAMDS MAPPING")
        logger.info("=" * 70)

        if not self.load_onet_data():
            return False
            
        if not self._load_onet_embeddings():
            logger.error("Failed to load embeddings/query data. Halting.")
            return False
            
        if not self.build_hierarchy_mappings():
            return False

        # Map Entities
        self.ksamds_entities['occupation'] = self.map_occupation_entities()
        self.ksamds_entities['knowledge'] = self.map_knowledge_entities()
        self.ksamds_entities['skill'] = self.map_skill_entities()
        self.ksamds_entities['ability'] = self.map_ability_entities()
        self.ksamds_entities['task'] = self.map_task_entities()
        self.ksamds_entities['function'] = self.map_function_entities()
        self.ksamds_entities['education_level'] = self.map_education_levels()

        # Map Relationships (Now includes dimension-finding logic)
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
    # Assuming embeddings have already been generated by onet_embedding_generator.py
    success = mapper.map_all_entities(cleanup_after=False) 

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

        logger.info("\nRELATIONSHIP MAPPING RESULTS (Including Dimensions)")
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