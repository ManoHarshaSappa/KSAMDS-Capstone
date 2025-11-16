"""
Skills Framework Database Loader for KSAMDS Project

This module handles loading the mapped Skills Framework data into the PostgreSQL
KSAMDS database. It manages database connections, inserts entities,
creates relationships, and handles constraint violations gracefully.

Handles dimensional information in:
1. Entity definitions (knowledge_level, skill_level, ability_level, etc. junction tables)
2. Occupation relationships with full dimensional tracking:
   - occupation_knowledge/skill/ability: type_id, level_id, basis_id
   - occupation_function: environment_id, physicality_id, cognitive_id
   - occupation_task: type_id, environment_id, mode_id
3. Inferred relationships with confidence scores
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
from contextlib import contextmanager
import uuid
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "ksamds"


class SkillsFrameworkLoader:
    """Load mapped Skills Framework data into KSAMDS PostgreSQL database."""

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the Skills Framework loader.

        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        # Get project root directory (3 levels up from skillsframework_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.data_dir = project_root / "data"
        self.mapped_dir = self.data_dir / "skillsframework" / "archive" / "mapped"
        self.relationships_dir = self.data_dir / \
            "skillsframework" / "archive" / "relationships"

        # Load mapped data
        self.mapped_data = {}
        self.id_mappings = {}

        # Dimension lookup tables (populated from database)
        self.dimension_lookups = {
            'type_dim': {},
            'level_dim': {},
            'basis_dim': {},
            'environment_dim': {},
            'mode_dim': {},
            'physicality_dim': {},
            'cognitive_dim': {}
        }

        # Track insertion statistics
        self.insertion_stats = {
            'knowledge': 0,
            'skill': 0,
            'ability': 0,
            'occupation': 0,
            'function': 0,
            'task': 0,
            'occupation_relationships': 0,
            'inferred_relationships': 0,
            'entity_dimensions': 0,
            'errors': 0
        }

        logger.info(f"Mapped data directory: {self.mapped_dir}")
        logger.info(f"Relationships directory: {self.relationships_dir}")

    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password
            )
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def load_mapped_data(self) -> bool:
        """Load all mapped CSV files."""
        try:
            logger.info("="*70)
            logger.info("LOADING MAPPED DATA")
            logger.info("="*70)

            # Load entity mappings
            entity_files = ['sf_knowledge', 'sf_skill', 'sf_ability',
                            'sf_occupation', 'sf_task', 'sf_function']

            for file_name in entity_files:
                csv_path = self.mapped_dir / f"{file_name}.csv"
                if csv_path.exists():
                    self.mapped_data[file_name] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {file_name}: {len(self.mapped_data[file_name])} records")
                else:
                    logger.warning(f"File not found: {csv_path}")

            # Load relationship files
            relationship_files = [
                'sf_occupation_knowledge',
                'sf_occupation_skill',
                'sf_occupation_ability',
                'sf_occupation_task',
                'sf_occupation_function'
            ]

            for file_name in relationship_files:
                csv_path = self.mapped_dir / f"{file_name}.csv"
                if csv_path.exists():
                    self.mapped_data[file_name] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {file_name}: {len(self.mapped_data[file_name])} records")
                else:
                    logger.warning(f"Relationship file not found: {csv_path}")

            logger.info("="*70)
            return len(self.mapped_data) > 0

        except Exception as e:
            logger.error(f"Failed to load mapped data: {e}")
            return False

    def initialize_dimensions(self) -> bool:
        """
        Initialize dimension tables by extracting unique values from mapped CSV files.
        This dynamically discovers dimension values rather than using hardcoded defaults.
        """
        try:
            logger.info("=" * 70)
            logger.info("INITIALIZING DIMENSION TABLES FROM DATA")
            logger.info("-" * 70)

            # Collect all unique dimension values from CSV files
            dimensions = {
                'type_dim': {},      # scope -> set of (name, description)
                # scope -> set of (name, ordinal, description)
                'level_dim': {},
                'basis_dim': {},     # scope -> set of (name, description)
                'environment_dim': {},  # scope -> set of (name, description)
                'mode_dim': set(),   # set of (name, description) - no scope
                'physicality_dim': set(),  # set of (name, description) - no scope
                'cognitive_dim': set()     # set of (name, description) - no scope
            }

            # File to scope mapping
            file_scope_mapping = {
                'sf_knowledge': 'K',
                'sf_skill': 'S',
                'sf_ability': 'A',
                'sf_function': 'F',
                'sf_task': 'T'
            }

            # Level ordinal mapping (defines progression order)
            level_ordinals = {
                'K': {'Basic': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4},
                'S': {'Novice': 1, 'Proficient': 2, 'Expert': 3},
                'A': {'Low': 1, 'Moderate': 2, 'High': 3}
            }

            # Generate descriptions based on scope and name
            def get_description(scope, name, dim_type):
                """Generate appropriate description for dimension."""
                descriptions = {
                    'type_dim': {
                        'K': f'{name} knowledge',
                        'S': f'{name} skills',
                        'A': f'{name} abilities',
                        'T': f'{name} task type'
                    },
                    'level_dim': {
                        'K': f'{name} level of knowledge',
                        'S': f'{name} level of skill proficiency',
                        'A': f'{name} level of ability requirement'
                    },
                    'basis_dim': {
                        'K': f'Knowledge acquired through {name.lower()}',
                        'S': f'Skill developed through {name.lower()}',
                        'A': f'Ability developed through {name.lower()}'
                    },
                    'environment_dim': {
                        'F': f'{name} environment for functions',
                        'T': f'{name} environment for tasks'
                    }
                }
                return descriptions.get(dim_type, {}).get(scope, name)

            logger.info("⏳ Scanning CSV files for dimension values...")

            # Process each mapped file
            for file_name, scope in file_scope_mapping.items():
                if file_name not in self.mapped_data:
                    continue

                df = self.mapped_data[file_name]
                logger.info(f"  Processing {file_name} ({scope})...")

                # Extract dimensions from each dimension column
                for col in df.columns:
                    if not col.endswith('_dimensions'):
                        continue

                    dim_type = col.replace('_dimensions', '') + '_dim'
                    if dim_type not in dimensions:
                        continue

                    # Parse pipe-separated values
                    values = df[col].dropna().unique()
                    for val_str in values:
                        if not val_str or pd.isna(val_str):
                            continue

                        # Split by pipe for multi-value cells
                        for val in str(val_str).split('|'):
                            val = val.strip()
                            if not val:
                                continue

                            # Add to appropriate dimension collection
                            if dim_type in ['mode_dim', 'physicality_dim', 'cognitive_dim']:
                                # No scope for these dimensions
                                desc = f'{val} level' if dim_type in [
                                    'physicality_dim', 'cognitive_dim'] else val
                                dimensions[dim_type].add((val, desc))
                            elif dim_type == 'level_dim':
                                # Level has ordinal
                                ordinal = level_ordinals.get(
                                    scope, {}).get(val, 0)
                                desc = get_description(scope, val, dim_type)
                                if scope not in dimensions[dim_type]:
                                    dimensions[dim_type][scope] = set()
                                dimensions[dim_type][scope].add(
                                    (val, ordinal, desc))
                            else:
                                # Type, basis, environment (with scope)
                                desc = get_description(scope, val, dim_type)
                                if scope not in dimensions[dim_type]:
                                    dimensions[dim_type][scope] = set()
                                dimensions[dim_type][scope].add((val, desc))

            logger.info("✓ Dimension values extracted from CSV files")
            logger.info("-" * 70)

            # Insert dimensions into database
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert type dimensions
                if dimensions['type_dim']:
                    type_data = []
                    for scope, values in dimensions['type_dim'].items():
                        for name, desc in values:
                            type_data.append((scope, name, desc))

                    logger.info("⏳ Inserting type dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO type_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        type_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(type_data)} type dimensions")

                # Insert level dimensions
                if dimensions['level_dim']:
                    level_data = []
                    for scope, values in dimensions['level_dim'].items():
                        for name, ordinal, desc in values:
                            level_data.append((scope, name, ordinal, desc))

                    logger.info("⏳ Inserting level dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO level_dim (scope, name, ordinal, description) VALUES (%s, %s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        level_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(level_data)} level dimensions")

                # Insert basis dimensions
                if dimensions['basis_dim']:
                    basis_data = []
                    for scope, values in dimensions['basis_dim'].items():
                        for name, desc in values:
                            basis_data.append((scope, name, desc))

                    logger.info("⏳ Inserting basis dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO basis_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        basis_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(basis_data)} basis dimensions")

                # Insert environment dimensions
                if dimensions['environment_dim']:
                    env_data = []
                    for scope, values in dimensions['environment_dim'].items():
                        for name, desc in values:
                            env_data.append((scope, name, desc))

                    logger.info("⏳ Inserting environment dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO environment_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        env_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(env_data)} environment dimensions")

                # Insert mode dimensions (no scope)
                if dimensions['mode_dim']:
                    mode_data = list(dimensions['mode_dim'])
                    logger.info("⏳ Inserting mode dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO mode_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        mode_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(mode_data)} mode dimensions")

                # Insert physicality dimensions (no scope)
                if dimensions['physicality_dim']:
                    phys_data = list(dimensions['physicality_dim'])
                    logger.info("⏳ Inserting physicality dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO physicality_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        phys_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(phys_data)} physicality dimensions")

                # Insert cognitive dimensions (no scope)
                if dimensions['cognitive_dim']:
                    cog_data = list(dimensions['cognitive_dim'])
                    logger.info("⏳ Inserting cognitive dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO cognitive_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        cog_data,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(cog_data)} cognitive dimensions")

                conn.commit()
                logger.info("=" * 70)
                logger.info("✓ DIMENSION TABLES INITIALIZED FROM DATA")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to initialize dimensions: {e}", exc_info=True)
            return False

    def load_dimension_lookups(self) -> bool:
        """Load dimension ID lookups from database."""
        try:
            logger.info("=" * 70)
            logger.info("LOADING DIMENSION LOOKUPS")
            logger.info("-" * 70)

            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Load all dimension tables
                for dim_table in self.dimension_lookups.keys():
                    logger.info(f"⏳ Loading {dim_table}...")
                    if dim_table in ['type_dim', 'level_dim', 'basis_dim', 'environment_dim']:
                        cursor.execute(
                            f"SELECT id, scope, name FROM {dim_table}")
                        for row in cursor.fetchall():
                            # Store with scope prefix for proper lookup
                            key_with_scope = f"{row['scope']}:{row['name']}"
                            self.dimension_lookups[dim_table][key_with_scope] = row['id']
                            # Also store without scope for backward compatibility
                            if row['name'] not in self.dimension_lookups[dim_table]:
                                self.dimension_lookups[dim_table][row['name']
                                                                  ] = row['id']
                    # mode_dim, physicality_dim, cognitive_dim (no scope)
                    else:
                        cursor.execute(f"SELECT id, name FROM {dim_table}")
                        for row in cursor.fetchall():
                            self.dimension_lookups[dim_table][row['name']
                                                              ] = row['id']
                    logger.info(
                        f"✓ Loaded {len(self.dimension_lookups[dim_table])} {dim_table} entries")

                logger.info("=" * 70)
                logger.info("✓ DIMENSION LOOKUPS LOADED SUCCESSFULLY")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(f"Failed to load dimension lookups: {e}")
            return False

    def _get_dimension_id(self, dim_table: str, value: str, scope: Optional[str] = None) -> Optional[str]:
        """
        Get dimension ID from lookup table.

        Args:
            dim_table: Name of dimension table
            value: Dimension value
            scope: Optional scope (K/S/A/F/T)

        Returns:
            Dimension ID (UUID as string) or None if not found
        """
        if not value or pd.isna(value):
            return None

        value = str(value).strip()

        # Try with scope first if provided
        if scope and dim_table in ['type_dim', 'level_dim', 'basis_dim', 'environment_dim']:
            key_with_scope = f"{scope}:{value}"
            dim_id = self.dimension_lookups[dim_table].get(key_with_scope)
            if dim_id:
                return str(dim_id)

        # Fallback to value without scope
        dim_id = self.dimension_lookups[dim_table].get(value)
        return str(dim_id) if dim_id else None

    def _parse_dimension_list(self, dim_string: str) -> List[str]:
        """Parse pipe-separated dimension string into list."""
        if not dim_string or pd.isna(dim_string):
            return []
        return [d.strip() for d in str(dim_string).split('|') if d.strip()]

    # ========================================================================
    # ENTITY INSERTION METHODS
    # ========================================================================

    def insert_occupation_entities(self) -> bool:
        """Insert occupation entities into database."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING OCCUPATION ENTITIES")
            logger.info("-" * 70)

            if 'sf_occupation' not in self.mapped_data:
                logger.warning("No occupation data loaded")
                return True

            df = self.mapped_data['sf_occupation']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                insert_query = """
                    INSERT INTO occupation (id, title, source_ref, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                occupation_data = []
                for _, row in df.iterrows():
                    description = row.get('description', None)
                    if pd.notna(description):
                        description = str(description).strip()
                    else:
                        description = None

                    occupation_data.append((
                        row['id'],
                        row['name'],
                        row['source_ref'],
                        description
                    ))

                execute_batch(cursor, insert_query,
                              occupation_data, page_size=1000)
                conn.commit()

                self.insertion_stats['occupation'] = len(occupation_data)
                logger.info(f"✓ Inserted {len(occupation_data)} occupations")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(f"Failed to insert occupations: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_knowledge_entities(self) -> bool:
        """Insert knowledge entities and their dimensional relationships."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING KNOWLEDGE ENTITIES")
            logger.info("-" * 70)

            if 'sf_knowledge' not in self.mapped_data:
                logger.warning("No knowledge data loaded")
                return True

            df = self.mapped_data['sf_knowledge']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert knowledge entities
                insert_query = """
                    INSERT INTO knowledge (id, name, source_ref)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                knowledge_data = []
                dimension_data = {'type': [], 'level': [], 'basis': []}

                for _, row in df.iterrows():
                    knowledge_id = row['id']
                    knowledge_data.append((
                        knowledge_id,
                        row['name'],
                        row['source_ref']
                    ))

                    # Process dimensions
                    types = self._parse_dimension_list(
                        row.get('type_dimensions', ''))
                    for type_val in types:
                        type_id = self._get_dimension_id(
                            'type_dim', type_val, 'K')
                        if type_id:
                            dimension_data['type'].append(
                                (knowledge_id, type_id))

                    levels = self._parse_dimension_list(
                        row.get('level_dimensions', ''))
                    for level_val in levels:
                        level_id = self._get_dimension_id(
                            'level_dim', level_val, 'K')
                        if level_id:
                            dimension_data['level'].append(
                                (knowledge_id, level_id))

                    bases = self._parse_dimension_list(
                        row.get('basis_dimensions', ''))
                    for basis_val in bases:
                        basis_id = self._get_dimension_id(
                            'basis_dim', basis_val, 'K')
                        if basis_id:
                            dimension_data['basis'].append(
                                (knowledge_id, basis_id))

                execute_batch(cursor, insert_query,
                              knowledge_data, page_size=1000)
                logger.info(
                    f"✓ Inserted {len(knowledge_data)} knowledge entities")

                # Insert dimensional relationships
                for dim_type, dim_data in dimension_data.items():
                    if dim_data:
                        dim_query = f"""
                            INSERT INTO knowledge_{dim_type} (knowledge_id, {dim_type}_id)
                            VALUES (%s, %s)
                            ON CONFLICT (knowledge_id, {dim_type}_id) DO NOTHING
                        """
                        execute_batch(cursor, dim_query,
                                      dim_data, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(dim_data)} knowledge_{dim_type} relationships")
                        self.insertion_stats['entity_dimensions'] += len(
                            dim_data)

                conn.commit()
                self.insertion_stats['knowledge'] = len(knowledge_data)
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert knowledge entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_skill_entities(self) -> bool:
        """Insert skill entities and their dimensional relationships."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING SKILL ENTITIES")
            logger.info("-" * 70)

            if 'sf_skill' not in self.mapped_data:
                logger.warning("No skill data loaded")
                return True

            df = self.mapped_data['sf_skill']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                insert_query = """
                    INSERT INTO skill (id, name, source_ref)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                skill_data = []
                dimension_data = {'type': [], 'level': [], 'basis': []}

                for _, row in df.iterrows():
                    skill_id = row['id']
                    skill_data.append((
                        skill_id,
                        row['name'],
                        row['source_ref']
                    ))

                    # Process dimensions
                    types = self._parse_dimension_list(
                        row.get('type_dimensions', ''))
                    for type_val in types:
                        type_id = self._get_dimension_id(
                            'type_dim', type_val, 'S')
                        if type_id:
                            dimension_data['type'].append((skill_id, type_id))

                    levels = self._parse_dimension_list(
                        row.get('level_dimensions', ''))
                    for level_val in levels:
                        level_id = self._get_dimension_id(
                            'level_dim', level_val, 'S')
                        if level_id:
                            dimension_data['level'].append(
                                (skill_id, level_id))

                    bases = self._parse_dimension_list(
                        row.get('basis_dimensions', ''))
                    for basis_val in bases:
                        basis_id = self._get_dimension_id(
                            'basis_dim', basis_val, 'S')
                        if basis_id:
                            dimension_data['basis'].append(
                                (skill_id, basis_id))

                execute_batch(cursor, insert_query, skill_data, page_size=1000)
                logger.info(f"✓ Inserted {len(skill_data)} skill entities")

                # Insert dimensional relationships
                for dim_type, dim_data in dimension_data.items():
                    if dim_data:
                        dim_query = f"""
                            INSERT INTO skill_{dim_type} (skill_id, {dim_type}_id)
                            VALUES (%s, %s)
                            ON CONFLICT (skill_id, {dim_type}_id) DO NOTHING
                        """
                        execute_batch(cursor, dim_query,
                                      dim_data, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(dim_data)} skill_{dim_type} relationships")
                        self.insertion_stats['entity_dimensions'] += len(
                            dim_data)

                conn.commit()
                self.insertion_stats['skill'] = len(skill_data)
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert skill entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_ability_entities(self) -> bool:
        """Insert ability entities and their dimensional relationships."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING ABILITY ENTITIES")
            logger.info("-" * 70)

            if 'sf_ability' not in self.mapped_data:
                logger.warning("No ability data loaded")
                return True

            df = self.mapped_data['sf_ability']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                insert_query = """
                    INSERT INTO ability (id, name, source_ref)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                ability_data = []
                dimension_data = {'type': [], 'level': [], 'basis': []}

                for _, row in df.iterrows():
                    ability_id = row['id']
                    ability_data.append((
                        ability_id,
                        row['name'],
                        row['source_ref']
                    ))

                    # Process dimensions
                    types = self._parse_dimension_list(
                        row.get('type_dimensions', ''))
                    for type_val in types:
                        type_id = self._get_dimension_id(
                            'type_dim', type_val, 'A')
                        if type_id:
                            dimension_data['type'].append(
                                (ability_id, type_id))

                    levels = self._parse_dimension_list(
                        row.get('level_dimensions', ''))
                    for level_val in levels:
                        level_id = self._get_dimension_id(
                            'level_dim', level_val, 'A')
                        if level_id:
                            dimension_data['level'].append(
                                (ability_id, level_id))

                    bases = self._parse_dimension_list(
                        row.get('basis_dimensions', ''))
                    for basis_val in bases:
                        basis_id = self._get_dimension_id(
                            'basis_dim', basis_val, 'A')
                        if basis_id:
                            dimension_data['basis'].append(
                                (ability_id, basis_id))

                execute_batch(cursor, insert_query,
                              ability_data, page_size=1000)
                logger.info(f"✓ Inserted {len(ability_data)} ability entities")

                # Insert dimensional relationships
                for dim_type, dim_data in dimension_data.items():
                    if dim_data:
                        dim_query = f"""
                            INSERT INTO ability_{dim_type} (ability_id, {dim_type}_id)
                            VALUES (%s, %s)
                            ON CONFLICT (ability_id, {dim_type}_id) DO NOTHING
                        """
                        execute_batch(cursor, dim_query,
                                      dim_data, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(dim_data)} ability_{dim_type} relationships")
                        self.insertion_stats['entity_dimensions'] += len(
                            dim_data)

                conn.commit()
                self.insertion_stats['ability'] = len(ability_data)
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert ability entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_function_entities(self) -> bool:
        """Insert function entities and their dimensional relationships."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING FUNCTION ENTITIES")
            logger.info("-" * 70)

            if 'sf_function' not in self.mapped_data:
                logger.warning("No function data loaded")
                return True

            df = self.mapped_data['sf_function']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                insert_query = """
                    INSERT INTO function (id, name, source_ref)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                function_data = []
                dimension_data = {'environment': [],
                                  'physicality': [], 'cognitive': []}

                for _, row in df.iterrows():
                    function_id = row['id']
                    function_data.append((
                        function_id,
                        row['name'],
                        row['source_ref']
                    ))

                    # Process dimensions
                    environments = self._parse_dimension_list(
                        row.get('environment_dimensions', ''))
                    for env_val in environments:
                        env_id = self._get_dimension_id(
                            'environment_dim', env_val, 'F')
                        if env_id:
                            dimension_data['environment'].append(
                                (function_id, env_id))

                    physicalities = self._parse_dimension_list(
                        row.get('physicality_dimensions', ''))
                    for phys_val in physicalities:
                        phys_id = self._get_dimension_id(
                            'physicality_dim', phys_val)
                        if phys_id:
                            dimension_data['physicality'].append(
                                (function_id, phys_id))

                    cognitives = self._parse_dimension_list(
                        row.get('cognitive_dimensions', ''))
                    for cog_val in cognitives:
                        cog_id = self._get_dimension_id(
                            'cognitive_dim', cog_val)
                        if cog_id:
                            dimension_data['cognitive'].append(
                                (function_id, cog_id))

                execute_batch(cursor, insert_query,
                              function_data, page_size=1000)
                logger.info(
                    f"✓ Inserted {len(function_data)} function entities")

                # Insert dimensional relationships
                for dim_type, dim_data in dimension_data.items():
                    if dim_data:
                        # Map dimension type to actual table name
                        table_suffix = 'env' if dim_type == 'environment' else dim_type
                        dim_query = f"""
                            INSERT INTO function_{table_suffix} (function_id, {dim_type}_id)
                            VALUES (%s, %s)
                            ON CONFLICT (function_id, {dim_type}_id) DO NOTHING
                        """
                        execute_batch(cursor, dim_query,
                                      dim_data, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(dim_data)} function_{table_suffix} relationships")
                        self.insertion_stats['entity_dimensions'] += len(
                            dim_data)

                conn.commit()
                self.insertion_stats['function'] = len(function_data)
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert function entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_task_entities(self) -> bool:
        """Insert task entities and their dimensional relationships."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING TASK ENTITIES")
            logger.info("-" * 70)

            if 'sf_task' not in self.mapped_data:
                logger.warning("No task data loaded")
                return True

            df = self.mapped_data['sf_task']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                insert_query = """
                    INSERT INTO task (id, name, source_ref)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                task_data = []
                dimension_data = {'type': [], 'environment': [], 'mode': []}

                for _, row in df.iterrows():
                    task_id = row['id']
                    task_data.append((
                        task_id,
                        row['name'],
                        row['source_ref']
                    ))

                    # Process dimensions
                    types = self._parse_dimension_list(
                        row.get('type_dimensions', ''))
                    for type_val in types:
                        type_id = self._get_dimension_id(
                            'type_dim', type_val, 'T')
                        if type_id:
                            dimension_data['type'].append((task_id, type_id))

                    environments = self._parse_dimension_list(
                        row.get('environment_dimensions', ''))
                    for env_val in environments:
                        env_id = self._get_dimension_id(
                            'environment_dim', env_val, 'T')
                        if env_id:
                            dimension_data['environment'].append(
                                (task_id, env_id))

                    modes = self._parse_dimension_list(
                        row.get('mode_dimensions', ''))
                    for mode_val in modes:
                        mode_id = self._get_dimension_id('mode_dim', mode_val)
                        if mode_id:
                            dimension_data['mode'].append((task_id, mode_id))

                execute_batch(cursor, insert_query, task_data, page_size=1000)
                logger.info(f"✓ Inserted {len(task_data)} task entities")

                # Insert dimensional relationships
                for dim_type, dim_data in dimension_data.items():
                    if dim_data:
                        # Map dimension type to actual table name
                        table_suffix = 'env' if dim_type == 'environment' else dim_type
                        dim_query = f"""
                            INSERT INTO task_{table_suffix} (task_id, {dim_type}_id)
                            VALUES (%s, %s)
                            ON CONFLICT (task_id, {dim_type}_id) DO NOTHING
                        """
                        execute_batch(cursor, dim_query,
                                      dim_data, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(dim_data)} task_{table_suffix} relationships")
                        self.insertion_stats['entity_dimensions'] += len(
                            dim_data)

                conn.commit()
                self.insertion_stats['task'] = len(task_data)
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(f"Failed to insert task entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    # ========================================================================
    # RELATIONSHIP INSERTION METHODS
    # ========================================================================

    def insert_occupation_relationships(self) -> bool:
        """Insert occupation-entity relationships with dimensional tracking."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING OCCUPATION RELATIONSHIPS")
            logger.info("=" * 70)

            total_relationships = 0

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Knowledge relationships
                if 'sf_occupation_knowledge' in self.mapped_data:
                    df = self.mapped_data['sf_occupation_knowledge']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation_knowledge relationships...")

                    # Fetch knowledge type and basis mappings from entity tables
                    cursor.execute(
                        "SELECT knowledge_id, type_id FROM knowledge_type")
                    knowledge_types = {row[0]: row[1]
                                       for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT knowledge_id, basis_id FROM knowledge_basis")
                    knowledge_bases = {row[0]: row[1]
                                       for row in cursor.fetchall()}

                    relationships = []
                    for _, row in df.iterrows():
                        knowledge_id = row['entity_id']
                        level_id = self._get_dimension_id('level_dim', row.get(
                            'level'), 'K') if pd.notna(row.get('level')) else None
                        type_id = knowledge_types.get(knowledge_id)
                        basis_id = knowledge_bases.get(knowledge_id)

                        relationships.append((
                            row['occupation_id'],
                            knowledge_id,
                            type_id,
                            level_id,
                            basis_id
                        ))

                    insert_query = """
                        INSERT INTO occupation_knowledge (occupation_id, knowledge_id, type_id, level_id, basis_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (occupation_id, knowledge_id) DO NOTHING
                    """
                    execute_batch(cursor, insert_query,
                                  relationships, page_size=1000)
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation_knowledge relationships")
                    total_relationships += len(relationships)

                # Skill relationships
                if 'sf_occupation_skill' in self.mapped_data:
                    df = self.mapped_data['sf_occupation_skill']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation_skill relationships...")

                    # Fetch skill type and basis mappings from entity tables
                    cursor.execute("SELECT skill_id, type_id FROM skill_type")
                    skill_types = {row[0]: row[1] for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT skill_id, basis_id FROM skill_basis")
                    skill_bases = {row[0]: row[1] for row in cursor.fetchall()}

                    relationships = []
                    for _, row in df.iterrows():
                        skill_id = row['entity_id']
                        level_id = self._get_dimension_id('level_dim', row.get(
                            'level'), 'S') if pd.notna(row.get('level')) else None
                        type_id = skill_types.get(skill_id)
                        basis_id = skill_bases.get(skill_id)

                        relationships.append((
                            row['occupation_id'],
                            skill_id,
                            type_id,
                            level_id,
                            basis_id
                        ))

                    insert_query = """
                        INSERT INTO occupation_skill (occupation_id, skill_id, type_id, level_id, basis_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (occupation_id, skill_id) DO NOTHING
                    """
                    execute_batch(cursor, insert_query,
                                  relationships, page_size=1000)
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation_skill relationships")
                    total_relationships += len(relationships)

                # Ability relationships
                if 'sf_occupation_ability' in self.mapped_data:
                    df = self.mapped_data['sf_occupation_ability']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation_ability relationships...")

                    # Fetch ability type and basis mappings from entity tables
                    cursor.execute(
                        "SELECT ability_id, type_id FROM ability_type")
                    ability_types = {row[0]: row[1]
                                     for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT ability_id, basis_id FROM ability_basis")
                    ability_bases = {row[0]: row[1]
                                     for row in cursor.fetchall()}

                    relationships = []
                    for _, row in df.iterrows():
                        ability_id = row['entity_id']
                        level_id = self._get_dimension_id('level_dim', row.get(
                            'level'), 'A') if pd.notna(row.get('level')) else None
                        type_id = ability_types.get(ability_id)
                        basis_id = ability_bases.get(ability_id)

                        relationships.append((
                            row['occupation_id'],
                            ability_id,
                            type_id,
                            level_id,
                            basis_id
                        ))

                    insert_query = """
                        INSERT INTO occupation_ability (occupation_id, ability_id, type_id, level_id, basis_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (occupation_id, ability_id) DO NOTHING
                    """
                    execute_batch(cursor, insert_query,
                                  relationships, page_size=1000)
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation_ability relationships")
                    total_relationships += len(relationships)

                # Function relationships
                if 'sf_occupation_function' in self.mapped_data:
                    df = self.mapped_data['sf_occupation_function']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation_function relationships...")

                    # Fetch function dimension mappings from entity tables
                    cursor.execute(
                        "SELECT function_id, environment_id FROM function_env")
                    function_envs = {row[0]: row[1]
                                     for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT function_id, physicality_id FROM function_physicality")
                    function_physicalities = {row[0]: row[1]
                                              for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT function_id, cognitive_id FROM function_cognitive")
                    function_cognitives = {row[0]: row[1]
                                           for row in cursor.fetchall()}

                    relationships = []
                    for _, row in df.iterrows():
                        function_id = row['entity_id']
                        environment_id = function_envs.get(function_id)
                        physicality_id = function_physicalities.get(
                            function_id)
                        cognitive_id = function_cognitives.get(function_id)

                        relationships.append((
                            row['occupation_id'],
                            function_id,
                            environment_id,
                            physicality_id,
                            cognitive_id
                        ))

                    insert_query = """
                        INSERT INTO occupation_function (occupation_id, function_id, environment_id, physicality_id, cognitive_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (occupation_id, function_id) DO NOTHING
                    """
                    execute_batch(cursor, insert_query,
                                  relationships, page_size=1000)
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation_function relationships")
                    total_relationships += len(relationships)

                # Task relationships
                if 'sf_occupation_task' in self.mapped_data:
                    df = self.mapped_data['sf_occupation_task']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation_task relationships...")

                    # Fetch task dimension mappings from entity tables
                    cursor.execute("SELECT task_id, type_id FROM task_type")
                    task_types = {row[0]: row[1] for row in cursor.fetchall()}
                    cursor.execute(
                        "SELECT task_id, environment_id FROM task_env")
                    task_envs = {row[0]: row[1] for row in cursor.fetchall()}
                    cursor.execute("SELECT task_id, mode_id FROM task_mode")
                    task_modes = {row[0]: row[1] for row in cursor.fetchall()}

                    relationships = []
                    for _, row in df.iterrows():
                        task_id = row['entity_id']
                        type_id = task_types.get(task_id)
                        environment_id = task_envs.get(task_id)
                        mode_id = task_modes.get(task_id)

                        relationships.append((
                            row['occupation_id'],
                            task_id,
                            type_id,
                            environment_id,
                            mode_id
                        ))

                    insert_query = """
                        INSERT INTO occupation_task (occupation_id, task_id, type_id, environment_id, mode_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (occupation_id, task_id) DO NOTHING
                    """
                    execute_batch(cursor, insert_query,
                                  relationships, page_size=1000)
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation_task relationships")
                    total_relationships += len(relationships)

                conn.commit()
                self.insertion_stats['occupation_relationships'] = total_relationships
                logger.info("=" * 70)
                logger.info(
                    f"✓ TOTAL OCCUPATION RELATIONSHIPS: {total_relationships}")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert occupation relationships: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_inferred_relationships(self) -> bool:
        """Insert inferred relationships with confidence scores."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING INFERRED RELATIONSHIPS")
            logger.info("=" * 70)

            total_inferred_relationships = 0

            # Relationship configurations
            relationship_configs = {
                'knowledge_skill': ('knowledge_skill', 'knowledge_id', 'skill_id'),
                'skill_ability': ('skill_ability', 'skill_id', 'ability_id'),
                'knowledge_function_inferred': ('knowledge_function', 'knowledge_id', 'function_id'),
                'ability_task_inferred': ('ability_task', 'ability_id', 'task_id'),
                'function_task_inferred': ('function_task', 'function_id', 'task_id')
            }

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                for filename, (table_name, source_col, target_col) in relationship_configs.items():
                    filepath = self.relationships_dir / f"{filename}.csv"

                    if not filepath.exists():
                        logger.warning(f"⚠ File not found: {filepath}")
                        continue

                    logger.info(f"⏳ Processing {filename}...")
                    rel_df = pd.read_csv(filepath)

                    relationships = []
                    for _, row in rel_df.iterrows():
                        relationships.append((
                            row['source_id'],
                            row['target_id'],
                            row.get('confidence_score', 1.0)
                        ))

                    logger.info(f"  Found {len(relationships)} relationships")

                    if relationships:
                        insert_query = f"""
                            INSERT INTO {table_name} ({source_col}, {target_col}, confidence_score)
                            VALUES (%s, %s, %s)
                            ON CONFLICT ({source_col}, {target_col}) DO NOTHING
                        """
                        execute_batch(cursor, insert_query,
                                      relationships, page_size=1000)
                        logger.info(
                            f"✓ Inserted {len(relationships)} {table_name} relationships")
                        total_inferred_relationships += len(relationships)

                conn.commit()
                self.insertion_stats['inferred_relationships'] = total_inferred_relationships
                logger.info("=" * 70)
                logger.info(
                    f"✓ TOTAL INFERRED RELATIONSHIPS: {total_inferred_relationships}")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert inferred relationships: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def _save_load_metadata(self):
        """Save metadata about the loading process."""
        logger.info("=" * 70)
        logger.info("SAVING LOAD METADATA")
        logger.info("-" * 70)

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'mapped_directory': str(self.mapped_dir),
            'relationships_directory': str(self.relationships_dir),
            'insertion_statistics': self.insertion_stats,
            'database': {
                'host': self.db_config.host,
                'database': self.db_config.database,
                'schema': self.db_config.schema
            }
        }

        # Save to reports directory
        reports_dir = self.data_dir / "skillsframework" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self.data_dir / "skillsframework" / \
            "reports" / "sf_database_load_metadata.json"
        logger.info(f"⏳ Writing metadata to {metadata_path}...")

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved successfully")
        logger.info("=" * 70)

    def load_all_data(self) -> bool:
        """Complete data loading pipeline."""
        logger.info("="*70)
        logger.info("STARTING SKILLS FRAMEWORK DATA LOADING PIPELINE")
        logger.info("="*70)

        # Load mapped data
        if not self.load_mapped_data():
            logger.error("Failed to load mapped data")
            return False

        # Initialize dimensions
        if not self.initialize_dimensions():
            logger.error("Failed to initialize dimensions")
            return False

        # Load dimension lookups
        if not self.load_dimension_lookups():
            logger.error("Failed to load dimension lookups")
            return False

        # Insert all entities
        logger.info("\n" + "="*70)
        logger.info("INSERTING ENTITIES")
        logger.info("="*70)

        success = True
        success &= self.insert_occupation_entities()
        success &= self.insert_knowledge_entities()
        success &= self.insert_skill_entities()
        success &= self.insert_ability_entities()
        success &= self.insert_function_entities()
        success &= self.insert_task_entities()

        # Insert relationships
        logger.info("\n" + "="*70)
        logger.info("INSERTING RELATIONSHIPS")
        logger.info("="*70)

        success &= self.insert_occupation_relationships()
        success &= self.insert_inferred_relationships()

        # Save metadata
        self._save_load_metadata()

        if success:
            logger.info("\n" + "="*70)
            logger.info(
                "SKILLS FRAMEWORK DATA LOADING COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            self._print_insertion_summary()
        else:
            logger.error("Data loading completed with errors")

        return success

    def _print_insertion_summary(self):
        """Log summary of insertion statistics."""
        logger.info("=" * 70)
        logger.info("DATABASE INSERTION SUMMARY")
        logger.info("-" * 70)

        for entity_type, count in self.insertion_stats.items():
            if entity_type != 'errors' and count > 0:
                logger.info(
                    f"✓ {entity_type.upper()}: {count:,} records inserted")

        if self.insertion_stats['errors'] > 0:
            logger.error(
                f"✗ ERRORS: {self.insertion_stats['errors']} errors occurred")

        logger.info("-" * 70)
        logger.info("✓ Database loading completed!")
        logger.info("=" * 70)

    @classmethod
    def from_environment(cls) -> 'SkillsFrameworkLoader':
        """Create loader instance from environment variables."""
        logger.info("=" * 70)
        logger.info("INITIALIZING FROM ENVIRONMENT")
        logger.info("-" * 70)

        if not all(os.getenv(var) for var in ['DB_USER', 'DB_PASSWORD']):
            logger.error("✗ Missing required environment variables")
            raise ValueError(
                "DB_USER and DB_PASSWORD environment variables must be set")

        db_config = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'ksamds'),
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            schema=os.getenv('DB_SCHEMA', 'ksamds')
        )

        logger.info(f"✓ Database Host: {db_config.host}")
        logger.info(f"✓ Database Name: {db_config.database}")
        logger.info(f"✓ Schema: {db_config.schema}")
        logger.info("=" * 70)

        return cls(db_config)


def main():
    """Main function to run the Skills Framework loader."""
    logger.info("=" * 70)
    logger.info("SKILLS FRAMEWORK LOADER - MAIN EXECUTION")
    logger.info("=" * 70)

    try:
        # Load database config from environment
        loader = SkillsFrameworkLoader.from_environment()

        # Run the data loading pipeline
        success = loader.load_all_data()

        if not success:
            logger.error("✗ Data loading failed")
            exit(1)

        logger.info("✓ Program completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ PROGRAM FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


if __name__ == "__main__":
    main()
