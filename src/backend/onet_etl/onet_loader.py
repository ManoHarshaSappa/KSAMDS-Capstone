"""
O*NET Database Loader for KSAMDS Project - UPDATED FOR FULL DIMENSIONAL TRACKING

This module handles loading the mapped O*NET data into the PostgreSQL
KSAMDS database. It manages database connections, inserts entities,
creates relationships, and handles constraint violations gracefully.

UPDATED: Now properly handles dimensional information in:
1. Entity definitions (knowledge_level, skill_level, ability_level, etc. junction tables)
2. Occupation relationships with full dimensional tracking:
   - occupation_knowledge/skill/ability: type_id, level_id, basis_id
   - occupation_function: environment_id, physicality_id, cognitive_id
   - occupation_task: type_id, environment_id, mode_id
3. Removed certification table support (no longer in schema)
4. Fixed mode_dim to not use scope column (single-purpose dimension)
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


class ONetLoader:
    """Load mapped O*NET data into KSAMDS PostgreSQL database."""

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the O*NET loader.

        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        # Get project root directory (3 levels up from onet_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.data_dir = project_root / "data" / "onet"
        self.mapped_dir = self.data_dir / "archive" / "mapped"
        self.relationships_dir = self.data_dir / "archive" / "relationships"

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
            'education_level': 0,
            'occupation_relationships': 0,
            'inferred_relationships': 0,
            'entity_levels': 0,  # NEW: Track entity-level associations
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
            entity_files = ['knowledge_mapped', 'skill_mapped', 'ability_mapped',
                            'occupation_mapped', 'task_mapped', 'function_mapped',
                            'education_level_mapped']

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
                'occupation_knowledge_relationships',
                'occupation_skill_relationships',
                'occupation_ability_relationships',
                'occupation_task_relationships',
                'occupation_function_relationships',
                'occupation_education_relationships'
            ]

            for file_name in relationship_files:
                csv_path = self.mapped_dir / f"{file_name}.csv"
                if csv_path.exists():
                    self.mapped_data[file_name] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {file_name}: {len(self.mapped_data[file_name])} records")
                else:
                    logger.warning(f"Relationship file not found: {csv_path}")

            # Load ID mappings
            id_mapping_path = self.mapped_dir / "id_mappings.csv"
            if id_mapping_path.exists():
                mapping_df = pd.read_csv(id_mapping_path)
                self.id_mappings = dict(
                    zip(mapping_df['onet_id'], mapping_df['ksamds_id']))
                logger.info(f"Loaded {len(self.id_mappings)} ID mappings")

            logger.info("="*70)
            return len(self.mapped_data) > 0

        except Exception as e:
            logger.error(f"Failed to load mapped data: {e}")
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

    def initialize_dimensions(self) -> bool:
        """
        Initialize dimension tables by extracting unique values from mapped data files.
        This ensures dimensions are data-driven and adapt to new data sources.
        """
        try:
            logger.info("=" * 70)
            logger.info("INITIALIZING DIMENSION TABLES FROM DATA")
            logger.info("-" * 70)

            # Track all unique dimension values from the data
            dimension_sets = {
                'type_dim': {},      # scope -> set of names
                'level_dim': {},     # scope -> set of names
                'basis_dim': {},     # scope -> set of names
                'environment_dim': {},  # scope -> set of names
                'mode_dim': set(),
                'physicality_dim': set(),
                'cognitive_dim': set()
            }

            # Scope mappings for entities
            scope_map = {
                'knowledge_mapped': 'K',
                'skill_mapped': 'S',
                'ability_mapped': 'A',
                'task_mapped': 'T',
                'function_mapped': 'F'
            }

            # Extract dimensions from each mapped file
            for file_key, scope in scope_map.items():
                if file_key not in self.mapped_data:
                    logger.warning(f"Skipping {file_key} - not loaded")
                    continue

                df = self.mapped_data[file_key]
                logger.info(
                    f"📊 Extracting dimensions from {file_key} ({len(df)} rows)...")

                # Extract type dimensions
                if 'type_dims' in df.columns:
                    for types in df['type_dims'].dropna():
                        for type_name in str(types).split('|'):
                            type_name = type_name.strip()
                            if type_name:
                                if scope not in dimension_sets['type_dim']:
                                    dimension_sets['type_dim'][scope] = set()
                                dimension_sets['type_dim'][scope].add(
                                    type_name)

                # Extract level dimensions
                if 'level_dims' in df.columns:
                    for levels in df['level_dims'].dropna():
                        for level_name in str(levels).split('|'):
                            level_name = level_name.strip()
                            if level_name:
                                if scope not in dimension_sets['level_dim']:
                                    dimension_sets['level_dim'][scope] = set()
                                dimension_sets['level_dim'][scope].add(
                                    level_name)

                # Extract basis dimensions
                if 'basis_dims' in df.columns:
                    for bases in df['basis_dims'].dropna():
                        for basis_name in str(bases).split('|'):
                            basis_name = basis_name.strip()
                            if basis_name:
                                if scope not in dimension_sets['basis_dim']:
                                    dimension_sets['basis_dim'][scope] = set()
                                dimension_sets['basis_dim'][scope].add(
                                    basis_name)

                # Extract environment dimensions
                if 'environment_dims' in df.columns:
                    for envs in df['environment_dims'].dropna():
                        for env_name in str(envs).split('|'):
                            env_name = env_name.strip()
                            if env_name:
                                if scope not in dimension_sets['environment_dim']:
                                    dimension_sets['environment_dim'][scope] = set(
                                    )
                                dimension_sets['environment_dim'][scope].add(
                                    env_name)

                # Extract mode dimensions (no scope)
                if 'mode_dims' in df.columns:
                    for modes in df['mode_dims'].dropna():
                        for mode_name in str(modes).split('|'):
                            mode_name = mode_name.strip()
                            if mode_name:
                                dimension_sets['mode_dim'].add(mode_name)

                # Extract physicality dimensions (no scope)
                if 'physicality_dims' in df.columns:
                    for phys in df['physicality_dims'].dropna():
                        for phys_name in str(phys).split('|'):
                            phys_name = phys_name.strip()
                            if phys_name:
                                dimension_sets['physicality_dim'].add(
                                    phys_name)

                # Extract cognitive dimensions (no scope)
                if 'cognitive_dims' in df.columns:
                    for cog in df['cognitive_dims'].dropna():
                        for cog_name in str(cog).split('|'):
                            cog_name = cog_name.strip()
                            if cog_name:
                                dimension_sets['cognitive_dim'].add(cog_name)

            # Insert into database
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert type dimensions
                type_dims = []
                for scope, names in dimension_sets['type_dim'].items():
                    for name in sorted(names):
                        type_dims.append((scope, name, None))

                if type_dims:
                    logger.info(
                        f"⏳ Inserting {len(type_dims)} type dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO type_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        type_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(type_dims)} type dimensions")

                # Insert level dimensions with ordinal assignment
                level_dims = []
                # Define ordinal mappings for common level names
                level_ordinals = {
                    'K': {'Basic': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4},
                    'S': {'Novice': 1, 'Proficient': 2, 'Expert': 3, 'Master': 4},
                    'A': {'Low': 1, 'Moderate': 2, 'High': 3}
                }

                for scope, names in dimension_sets['level_dim'].items():
                    for name in sorted(names):
                        ordinal = level_ordinals.get(scope, {}).get(name, None)
                        level_dims.append((scope, name, ordinal, None))

                if level_dims:
                    logger.info(
                        f"⏳ Inserting {len(level_dims)} level dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO level_dim (scope, name, ordinal, description) VALUES (%s, %s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        level_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(level_dims)} level dimensions")

                # Insert basis dimensions
                basis_dims = []
                for scope, names in dimension_sets['basis_dim'].items():
                    for name in sorted(names):
                        basis_dims.append((scope, name, None))

                if basis_dims:
                    logger.info(
                        f"⏳ Inserting {len(basis_dims)} basis dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO basis_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        basis_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(basis_dims)} basis dimensions")

                # Insert environment dimensions
                env_dims = []
                for scope, names in dimension_sets['environment_dim'].items():
                    for name in sorted(names):
                        env_dims.append((scope, name, None))

                if env_dims:
                    logger.info(
                        f"⏳ Inserting {len(env_dims)} environment dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO environment_dim (scope, name, description) VALUES (%s, %s, %s) ON CONFLICT (scope, name) DO NOTHING",
                        env_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(env_dims)} environment dimensions")

                # Insert mode dimensions
                mode_dims = [(name, None)
                             for name in sorted(dimension_sets['mode_dim'])]
                if mode_dims:
                    logger.info(
                        f"⏳ Inserting {len(mode_dims)} mode dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO mode_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        mode_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(mode_dims)} mode dimensions")

                # Insert physicality dimensions
                phys_dims = [(name, None) for name in sorted(
                    dimension_sets['physicality_dim'])]
                if phys_dims:
                    logger.info(
                        f"⏳ Inserting {len(phys_dims)} physicality dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO physicality_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        phys_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(phys_dims)} physicality dimensions")

                # Insert cognitive dimensions
                cog_dims = [(name, None)
                            for name in sorted(dimension_sets['cognitive_dim'])]
                if cog_dims:
                    logger.info(
                        f"⏳ Inserting {len(cog_dims)} cognitive dimensions...")
                    execute_batch(
                        cursor,
                        "INSERT INTO cognitive_dim (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        cog_dims,
                        page_size=100
                    )
                    logger.info(
                        f"✓ Inserted/verified {len(cog_dims)} cognitive dimensions")

                conn.commit()
                logger.info("=" * 70)
                logger.info("✓ DIMENSION TABLES INITIALIZED FROM DATA")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to initialize dimensions: {e}", exc_info=True)
            return False

    def _get_level_id(self, level_name: str, scope: str) -> Optional[str]:
        """
        Get level dimension ID for a specific scope.

        Args:
            level_name: Name of the level (e.g., 'Basic', 'Novice', 'Low')
            scope: Scope ('K', 'S', or 'A')

        Returns:
            UUID of the level dimension, or None if not found
        """
        key = f"{scope}:{level_name}"
        return self.dimension_lookups['level_dim'].get(key)

    def _get_type_id(self, type_name: str, scope: str) -> Optional[str]:
        """
        Get type dimension ID for a specific scope.

        Args:
            type_name: Name of the type (e.g., 'Technical', 'Analytical')
            scope: Scope ('K', 'S', 'A', or 'T')

        Returns:
            UUID of the type dimension, or None if not found
        """
        # Try scope-specific lookup first
        key = f"{scope}:{type_name}"
        type_id = self.dimension_lookups['type_dim'].get(key)

        # Fall back to non-scoped lookup if not found
        if not type_id:
            type_id = self.dimension_lookups['type_dim'].get(type_name)

        return type_id

    def _get_basis_id(self, basis_name: str, scope: str) -> Optional[str]:
        """
        Get basis dimension ID for a specific scope.

        Args:
            basis_name: Name of the basis (e.g., 'Academic', 'On-the-Job Training')
            scope: Scope ('K', 'S', or 'A')

        Returns:
            UUID of the basis dimension, or None if not found
        """
        # Try scope-specific lookup first
        key = f"{scope}:{basis_name}"
        basis_id = self.dimension_lookups['basis_dim'].get(key)

        # Fall back to non-scoped lookup if not found
        if not basis_id:
            basis_id = self.dimension_lookups['basis_dim'].get(basis_name)

        return basis_id

    def insert_knowledge_entities(self) -> bool:
        """
        Insert knowledge entities and their level associations.

        UPDATED: Now inserts into knowledge_level junction table.
        """
        try:
            logger.info("=" * 70)
            logger.info("INSERTING KNOWLEDGE ENTITIES WITH LEVELS")
            logger.info("-" * 70)

            if 'knowledge_mapped' not in self.mapped_data:
                logger.warning("No knowledge data to insert")
                return True

            df = self.mapped_data['knowledge_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert knowledge entities
                knowledge_data = []
                level_associations = []

                for _, row in df.iterrows():
                    knowledge_data.append((
                        row['id'],
                        row['name'],
                        row['source_ref'] if pd.notna(
                            row['source_ref']) else None
                    ))

                    # Parse level dimensions
                    level_dims = row['level_dims']
                    if pd.notna(level_dims) and level_dims:
                        for level_name in str(level_dims).split('|'):
                            level_name = level_name.strip()
                            if level_name:
                                level_id = self._get_level_id(level_name, 'K')
                                if level_id:
                                    level_associations.append(
                                        (row['id'], level_id))

                # Insert entities
                logger.info(
                    f"⏳ Inserting {len(knowledge_data)} knowledge entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO knowledge (id, name, source_ref) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    knowledge_data,
                    page_size=1000
                )

                # Insert level associations
                if level_associations:
                    logger.info(
                        f"⏳ Inserting {len(level_associations)} knowledge-level associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO knowledge_level (knowledge_id, level_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        level_associations,
                        page_size=1000
                    )
                    self.insertion_stats['entity_levels'] += len(
                        level_associations)

                # Insert type associations
                type_associations = []
                for _, row in df.iterrows():
                    type_dims = row['type_dims']
                    if pd.notna(type_dims) and type_dims:
                        for type_name in str(type_dims).split('|'):
                            type_name = type_name.strip()
                            if type_name:
                                type_id = self._get_type_id(type_name, 'K')
                                if type_id:
                                    type_associations.append(
                                        (row['id'], type_id))
                                else:
                                    logger.warning(
                                        f"Type dimension not found for Knowledge: K:{type_name}")

                if type_associations:
                    logger.info(
                        f"⏳ Inserting {len(type_associations)} knowledge-type associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO knowledge_type (knowledge_id, type_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        type_associations,
                        page_size=1000
                    )

                # Insert basis associations
                basis_associations = []
                for _, row in df.iterrows():
                    basis_dims = row['basis_dims']
                    if pd.notna(basis_dims) and basis_dims:
                        for basis_name in str(basis_dims).split('|'):
                            basis_name = basis_name.strip()
                            if basis_name:
                                basis_id = self._get_basis_id(basis_name, 'K')
                                if basis_id:
                                    basis_associations.append(
                                        (row['id'], basis_id))
                                else:
                                    logger.warning(
                                        f"Basis dimension not found for Knowledge: K:{basis_name}")

                if basis_associations:
                    logger.info(
                        f"⏳ Inserting {len(basis_associations)} knowledge-basis associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO knowledge_basis (knowledge_id, basis_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        basis_associations,
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['knowledge'] = len(knowledge_data)
                logger.info(
                    f"✓ Inserted {len(knowledge_data)} knowledge entities")
                logger.info(
                    f"✓ Inserted {len(level_associations)} level associations")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert knowledge entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_skill_entities(self) -> bool:
        """
        Insert skill entities and their level associations.

        UPDATED: Now inserts into skill_level junction table.
        """
        try:
            logger.info("=" * 70)
            logger.info("INSERTING SKILL ENTITIES WITH LEVELS")
            logger.info("-" * 70)

            if 'skill_mapped' not in self.mapped_data:
                logger.warning("No skill data to insert")
                return True

            df = self.mapped_data['skill_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert skill entities
                skill_data = []
                level_associations = []

                for _, row in df.iterrows():
                    skill_data.append((
                        row['id'],
                        row['name'],
                        row['source_ref'] if pd.notna(
                            row['source_ref']) else None
                    ))

                    # Parse level dimensions
                    level_dims = row['level_dims']
                    if pd.notna(level_dims) and level_dims:
                        for level_name in str(level_dims).split('|'):
                            level_name = level_name.strip()
                            if level_name:
                                level_id = self._get_level_id(level_name, 'S')
                                if level_id:
                                    level_associations.append(
                                        (row['id'], level_id))

                # Insert entities
                logger.info(f"⏳ Inserting {len(skill_data)} skill entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO skill (id, name, source_ref) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    skill_data,
                    page_size=1000
                )

                # Insert level associations
                if level_associations:
                    logger.info(
                        f"⏳ Inserting {len(level_associations)} skill-level associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO skill_level (skill_id, level_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        level_associations,
                        page_size=1000
                    )
                    self.insertion_stats['entity_levels'] += len(
                        level_associations)

                # Insert type associations
                type_associations = []
                for _, row in df.iterrows():
                    type_dims = row['type_dims']
                    if pd.notna(type_dims) and type_dims:
                        for type_name in str(type_dims).split('|'):
                            type_name = type_name.strip()
                            if type_name:
                                type_id = self._get_type_id(type_name, 'S')
                                if type_id:
                                    type_associations.append(
                                        (row['id'], type_id))
                                else:
                                    logger.warning(
                                        f"Type dimension not found for Skill: S:{type_name}")

                if type_associations:
                    logger.info(
                        f"⏳ Inserting {len(type_associations)} skill-type associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO skill_type (skill_id, type_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        type_associations,
                        page_size=1000
                    )

                # Insert basis associations
                basis_associations = []
                for _, row in df.iterrows():
                    basis_dims = row['basis_dims']
                    if pd.notna(basis_dims) and basis_dims:
                        for basis_name in str(basis_dims).split('|'):
                            basis_name = basis_name.strip()
                            if basis_name:
                                basis_id = self._get_basis_id(basis_name, 'S')
                                if basis_id:
                                    basis_associations.append(
                                        (row['id'], basis_id))
                                else:
                                    logger.warning(
                                        f"Basis dimension not found for Skill: S:{basis_name}")

                if basis_associations:
                    logger.info(
                        f"⏳ Inserting {len(basis_associations)} skill-basis associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO skill_basis (skill_id, basis_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        basis_associations,
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['skill'] = len(skill_data)
                logger.info(f"✓ Inserted {len(skill_data)} skill entities")
                logger.info(
                    f"✓ Inserted {len(level_associations)} level associations")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert skill entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_ability_entities(self) -> bool:
        """
        Insert ability entities and their level associations.

        UPDATED: Now inserts into ability_level junction table.
        """
        try:
            logger.info("=" * 70)
            logger.info("INSERTING ABILITY ENTITIES WITH LEVELS")
            logger.info("-" * 70)

            if 'ability_mapped' not in self.mapped_data:
                logger.warning("No ability data to insert")
                return True

            df = self.mapped_data['ability_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Insert ability entities
                ability_data = []
                level_associations = []

                for _, row in df.iterrows():
                    ability_data.append((
                        row['id'],
                        row['name'],
                        row['source_ref'] if pd.notna(
                            row['source_ref']) else None
                    ))

                    # Parse level dimensions
                    level_dims = row['level_dims']
                    if pd.notna(level_dims) and level_dims:
                        for level_name in str(level_dims).split('|'):
                            level_name = level_name.strip()
                            if level_name:
                                level_id = self._get_level_id(level_name, 'A')
                                if level_id:
                                    level_associations.append(
                                        (row['id'], level_id))

                # Insert entities
                logger.info(
                    f"⏳ Inserting {len(ability_data)} ability entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO ability (id, name, source_ref) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    ability_data,
                    page_size=1000
                )

                # Insert level associations
                if level_associations:
                    logger.info(
                        f"⏳ Inserting {len(level_associations)} ability-level associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO ability_level (ability_id, level_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        level_associations,
                        page_size=1000
                    )
                    self.insertion_stats['entity_levels'] += len(
                        level_associations)

                # Insert type associations
                type_associations = []
                for _, row in df.iterrows():
                    type_dims = row['type_dims']
                    if pd.notna(type_dims) and type_dims:
                        for type_name in str(type_dims).split('|'):
                            type_name = type_name.strip()
                            if type_name:
                                type_id = self._get_type_id(type_name, 'A')
                                if type_id:
                                    type_associations.append(
                                        (row['id'], type_id))
                                else:
                                    logger.warning(
                                        f"Type dimension not found for Ability: A:{type_name}")

                if type_associations:
                    logger.info(
                        f"⏳ Inserting {len(type_associations)} ability-type associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO ability_type (ability_id, type_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        type_associations,
                        page_size=1000
                    )

                # Insert basis associations
                basis_associations = []
                for _, row in df.iterrows():
                    basis_dims = row['basis_dims']
                    if pd.notna(basis_dims) and basis_dims:
                        for basis_name in str(basis_dims).split('|'):
                            basis_name = basis_name.strip()
                            if basis_name:
                                basis_id = self._get_basis_id(basis_name, 'A')
                                if basis_id:
                                    basis_associations.append(
                                        (row['id'], basis_id))
                                else:
                                    logger.warning(
                                        f"Basis dimension not found for Ability: A:{basis_name}")

                if basis_associations:
                    logger.info(
                        f"⏳ Inserting {len(basis_associations)} ability-basis associations...")
                    execute_batch(
                        cursor,
                        "INSERT INTO ability_basis (ability_id, basis_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        basis_associations,
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['ability'] = len(ability_data)
                logger.info(f"✓ Inserted {len(ability_data)} ability entities")
                logger.info(
                    f"✓ Inserted {len(level_associations)} level associations")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert ability entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_occupation_entities(self) -> bool:
        """Insert occupation entities."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING OCCUPATION ENTITIES")
            logger.info("-" * 70)

            if 'occupation_mapped' not in self.mapped_data:
                logger.warning("No occupation data to insert")
                return True

            df = self.mapped_data['occupation_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                occupation_data = [
                    (row['id'], row['name'], row.get('description') if pd.notna(
                        row.get('description')) else None, row['source_ref'] if pd.notna(
                        row['source_ref']) else None)
                    for _, row in df.iterrows()
                ]

                logger.info(
                    f"⏳ Inserting {len(occupation_data)} occupation entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO occupation (id, title, description, source_ref) VALUES (%s, %s, %s, %s) ON CONFLICT (title) DO NOTHING",
                    occupation_data,
                    page_size=1000
                )

                conn.commit()
                self.insertion_stats['occupation'] = len(occupation_data)
                logger.info(
                    f"✓ Inserted {len(occupation_data)} occupation entities")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert occupation entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_task_entities(self) -> bool:
        """Insert task entities and their dimension associations."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING TASK ENTITIES WITH DIMENSIONS")
            logger.info("-" * 70)

            if 'task_mapped' not in self.mapped_data:
                logger.warning("No task data to insert")
                return True

            df = self.mapped_data['task_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                task_data = [
                    (row['id'], row['name'], row['source_ref'] if pd.notna(
                        row['source_ref']) else None)
                    for _, row in df.iterrows()
                ]

                logger.info(f"⏳ Inserting {len(task_data)} task entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO task (id, name, source_ref) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    task_data,
                    page_size=1000
                )
                logger.info(f"✓ Inserted {len(task_data)} task entities")

                # Insert task dimension associations
                dim_count = 0

                # Task type associations
                if 'type_dims' in df.columns:
                    type_rels = []
                    for _, row in df.iterrows():
                        type_dims = row.get('type_dims', '')
                        if pd.notna(type_dims) and type_dims:
                            for type_name in str(type_dims).split('|'):
                                type_name = type_name.strip()
                                if type_name:
                                    type_id = self._get_type_id(type_name, 'T')
                                    if type_id:
                                        type_rels.append((row['id'], type_id))

                    if type_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO task_type (task_id, type_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            type_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(type_rels)} task type associations")
                        dim_count += len(type_rels)

                # Task mode associations
                if 'mode_dims' in df.columns:
                    mode_rels = []
                    for _, row in df.iterrows():
                        mode_dims = row.get('mode_dims', '')
                        if pd.notna(mode_dims) and mode_dims:
                            for mode_name in str(mode_dims).split('|'):
                                mode_name = mode_name.strip()
                                if mode_name:
                                    mode_id = self.dimension_lookups['mode_dim'].get(
                                        mode_name)
                                    if mode_id:
                                        mode_rels.append((row['id'], mode_id))

                    if mode_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO task_mode (task_id, mode_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            mode_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(mode_rels)} task mode associations")
                        dim_count += len(mode_rels)

                # Task environment associations
                if 'environment_dims' in df.columns:
                    env_rels = []
                    for _, row in df.iterrows():
                        env_dims = row.get('environment_dims', '')
                        if pd.notna(env_dims) and env_dims:
                            for env_name in str(env_dims).split('|'):
                                env_name = env_name.strip()
                                if env_name:
                                    env_id = self.dimension_lookups['environment_dim'].get(
                                        f"T:{env_name}",
                                        self.dimension_lookups['environment_dim'].get(
                                            env_name)
                                    )
                                    if env_id:
                                        env_rels.append((row['id'], env_id))

                    if env_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO task_env (task_id, environment_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            env_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(env_rels)} task environment associations")
                        dim_count += len(env_rels)

                conn.commit()
                self.insertion_stats['task'] = len(task_data)
                self.insertion_stats['entity_levels'] += dim_count
                logger.info(
                    f"✓ Total task dimension associations: {dim_count}")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(f"Failed to insert task entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_function_entities(self) -> bool:
        """Insert function entities and their dimension associations."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING FUNCTION ENTITIES WITH DIMENSIONS")
            logger.info("-" * 70)

            if 'function_mapped' not in self.mapped_data:
                logger.warning("No function data to insert")
                return True

            df = self.mapped_data['function_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                function_data = [
                    (row['id'], row['name'], row['source_ref'] if pd.notna(
                        row['source_ref']) else None)
                    for _, row in df.iterrows()
                ]

                logger.info(
                    f"⏳ Inserting {len(function_data)} function entities...")
                execute_batch(
                    cursor,
                    "INSERT INTO function (id, name, source_ref) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    function_data,
                    page_size=1000
                )
                logger.info(
                    f"✓ Inserted {len(function_data)} function entities")

                # Insert function dimension associations
                dim_count = 0

                # Function physicality associations
                if 'physicality_dims' in df.columns:
                    phys_rels = []
                    for _, row in df.iterrows():
                        phys_dims = row.get('physicality_dims', '')
                        if pd.notna(phys_dims) and phys_dims:
                            for phys_name in str(phys_dims).split('|'):
                                phys_name = phys_name.strip()
                                if phys_name:
                                    phys_id = self.dimension_lookups['physicality_dim'].get(
                                        phys_name)
                                    if phys_id:
                                        phys_rels.append((row['id'], phys_id))

                    if phys_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO function_physicality (function_id, physicality_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            phys_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(phys_rels)} function physicality associations")
                        dim_count += len(phys_rels)

                # Function cognitive associations
                if 'cognitive_dims' in df.columns:
                    cog_rels = []
                    for _, row in df.iterrows():
                        cog_dims = row.get('cognitive_dims', '')
                        if pd.notna(cog_dims) and cog_dims:
                            for cog_name in str(cog_dims).split('|'):
                                cog_name = cog_name.strip()
                                if cog_name:
                                    cog_id = self.dimension_lookups['cognitive_dim'].get(
                                        cog_name)
                                    if cog_id:
                                        cog_rels.append((row['id'], cog_id))

                    if cog_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO function_cognitive (function_id, cognitive_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            cog_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(cog_rels)} function cognitive associations")
                        dim_count += len(cog_rels)

                # Function environment associations
                if 'environment_dims' in df.columns:
                    env_rels = []
                    for _, row in df.iterrows():
                        env_dims = row.get('environment_dims', '')
                        if pd.notna(env_dims) and env_dims:
                            for env_name in str(env_dims).split('|'):
                                env_name = env_name.strip()
                                if env_name:
                                    env_id = self.dimension_lookups['environment_dim'].get(
                                        f"F:{env_name}",
                                        self.dimension_lookups['environment_dim'].get(
                                            env_name)
                                    )
                                    if env_id:
                                        env_rels.append((row['id'], env_id))

                    if env_rels:
                        execute_batch(
                            cursor,
                            "INSERT INTO function_env (function_id, environment_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            env_rels,
                            page_size=1000
                        )
                        logger.info(
                            f"  ✓ Inserted {len(env_rels)} function environment associations")
                        dim_count += len(env_rels)

                conn.commit()
                self.insertion_stats['function'] = len(function_data)
                self.insertion_stats['entity_levels'] += dim_count
                logger.info(
                    f"✓ Total function dimension associations: {dim_count}")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert function entities: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_education_levels(self) -> bool:
        """Insert education level entities."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING EDUCATION LEVELS")
            logger.info("-" * 70)

            if 'education_level_mapped' not in self.mapped_data:
                logger.warning("No education level data to insert")
                return True

            df = self.mapped_data['education_level_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Parse ordinal from source_ref (format: EDU_1, EDU_2, etc.)
                education_data = []
                for _, row in df.iterrows():
                    ordinal = None
                    if pd.notna(row['source_ref']):
                        try:
                            ordinal = int(row['source_ref'].split('_')[1])
                        except:
                            pass
                    education_data.append((
                        row['id'],
                        row['name'],
                        ordinal
                    ))

                logger.info(
                    f"⏳ Inserting {len(education_data)} education levels...")
                execute_batch(
                    cursor,
                    "INSERT INTO education_level (id, name, ordinal) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                    education_data,
                    page_size=100
                )

                conn.commit()
                self.insertion_stats['education_level'] = len(education_data)
                logger.info(
                    f"✓ Inserted {len(education_data)} education levels")
                logger.info("=" * 70)
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert education levels: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def insert_occupation_relationships(self) -> bool:
        """
        Insert occupation relationships with full dimensional tracking.

        UPDATED: Now properly inserts:
        - occupation_knowledge/skill/ability: type_id, level_id, basis_id
        - occupation_function: environment_id, physicality_id, cognitive_id  
        - occupation_task: type_id, environment_id, mode_id
        """
        try:
            logger.info("=" * 70)
            logger.info("INSERTING OCCUPATION RELATIONSHIPS WITH DIMENSIONS")
            logger.info("-" * 70)

            total_relationships = 0

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Knowledge relationships (with type, level, and basis)
                if 'occupation_knowledge_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_knowledge_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-knowledge relationships...")

                    relationships = []
                    for _, row in df.iterrows():
                        # Get level_id
                        level_id = None
                        if pd.notna(row.get('level')) and row['level']:
                            level_id = self._get_level_id(row['level'], 'K')

                        # Get type_id and basis_id (these may not exist in current mapped data yet)
                        type_id = None
                        if pd.notna(row.get('type')) and row['type']:
                            type_id = self._get_type_id(row['type'], 'K')

                        basis_id = None
                        if pd.notna(row.get('basis')) and row['basis']:
                            basis_id = self._get_basis_id(row['basis'], 'K')

                        importance = row.get('importance_score')
                        if pd.notna(importance):
                            importance = float(importance)
                        else:
                            importance = None

                        relationships.append((
                            row['occupation_id'],
                            row['entity_id'],
                            type_id,
                            level_id,
                            basis_id,
                            importance
                        ))

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_knowledge 
                           (occupation_id, knowledge_id, type_id, level_id, basis_id, importance_score) 
                           VALUES (%s, %s, %s, %s, %s, %s) 
                           ON CONFLICT (occupation_id, knowledge_id) DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-knowledge relationships")
                    total_relationships += len(relationships)

                # Skill relationships (with type, level, and basis)
                if 'occupation_skill_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_skill_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-skill relationships...")

                    relationships = []
                    for _, row in df.iterrows():
                        # Get level_id
                        level_id = None
                        if pd.notna(row.get('level')) and row['level']:
                            level_id = self._get_level_id(row['level'], 'S')

                        # Get type_id and basis_id
                        type_id = None
                        if pd.notna(row.get('type')) and row['type']:
                            type_id = self._get_type_id(row['type'], 'S')

                        basis_id = None
                        if pd.notna(row.get('basis')) and row['basis']:
                            basis_id = self._get_basis_id(row['basis'], 'S')

                        importance = row.get('importance_score')
                        if pd.notna(importance):
                            importance = float(importance)
                        else:
                            importance = None

                        relationships.append((
                            row['occupation_id'],
                            row['entity_id'],
                            type_id,
                            level_id,
                            basis_id,
                            importance
                        ))

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_skill 
                           (occupation_id, skill_id, type_id, level_id, basis_id, importance_score) 
                           VALUES (%s, %s, %s, %s, %s, %s) 
                           ON CONFLICT (occupation_id, skill_id) DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-skill relationships")
                    total_relationships += len(relationships)

                # Ability relationships (with type, level, and basis)
                if 'occupation_ability_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_ability_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-ability relationships...")

                    relationships = []
                    for _, row in df.iterrows():
                        # Get level_id
                        level_id = None
                        if pd.notna(row.get('level')) and row['level']:
                            level_id = self._get_level_id(row['level'], 'A')

                        # Get type_id and basis_id
                        type_id = None
                        if pd.notna(row.get('type')) and row['type']:
                            type_id = self._get_type_id(row['type'], 'A')

                        basis_id = None
                        if pd.notna(row.get('basis')) and row['basis']:
                            basis_id = self._get_basis_id(row['basis'], 'A')

                        importance = row.get('importance_score')
                        if pd.notna(importance):
                            importance = float(importance)
                        else:
                            importance = None

                        relationships.append((
                            row['occupation_id'],
                            row['entity_id'],
                            type_id,
                            level_id,
                            basis_id,
                            importance
                        ))

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_ability 
                           (occupation_id, ability_id, type_id, level_id, basis_id, importance_score) 
                           VALUES (%s, %s, %s, %s, %s, %s) 
                           ON CONFLICT (occupation_id, ability_id) DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-ability relationships")
                    total_relationships += len(relationships)

                # Task relationships (with type, environment, and mode)
                if 'occupation_task_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_task_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-task relationships...")

                    relationships = []
                    for _, row in df.iterrows():
                        # Get type_id, environment_id, and mode_id
                        type_id = None
                        if pd.notna(row.get('type')) and row['type']:
                            type_id = self._get_type_id(row['type'], 'T')

                        environment_id = None
                        if pd.notna(row.get('environment')) and row['environment']:
                            environment_id = self.dimension_lookups['environment_dim'].get(
                                f"T:{row['environment']}",
                                self.dimension_lookups['environment_dim'].get(
                                    row['environment'])
                            )

                        mode_id = None
                        if pd.notna(row.get('mode')) and row['mode']:
                            mode_id = self.dimension_lookups['mode_dim'].get(
                                row['mode'])

                        importance = row.get('importance_score')
                        if pd.notna(importance):
                            importance = float(importance)
                        else:
                            importance = None

                        relationships.append((
                            row['occupation_id'],
                            row['task_id'],
                            type_id,
                            environment_id,
                            mode_id,
                            importance
                        ))

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_task 
                           (occupation_id, task_id, type_id, environment_id, mode_id, importance_score) 
                           VALUES (%s, %s, %s, %s, %s, %s) 
                           ON CONFLICT (occupation_id, task_id) DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-task relationships")
                    total_relationships += len(relationships)

                # Function relationships (with environment, physicality, and cognitive)
                if 'occupation_function_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_function_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-function relationships...")

                    relationships = []
                    for _, row in df.iterrows():
                        # Get environment_id, physicality_id, and cognitive_id
                        environment_id = None
                        if pd.notna(row.get('environment')) and row['environment']:
                            environment_id = self.dimension_lookups['environment_dim'].get(
                                f"F:{row['environment']}",
                                self.dimension_lookups['environment_dim'].get(
                                    row['environment'])
                            )

                        physicality_id = None
                        if pd.notna(row.get('physicality')) and row['physicality']:
                            physicality_id = self.dimension_lookups['physicality_dim'].get(
                                row['physicality'])

                        cognitive_id = None
                        if pd.notna(row.get('cognitive')) and row['cognitive']:
                            cognitive_id = self.dimension_lookups['cognitive_dim'].get(
                                row['cognitive'])

                        importance = row.get('importance_score')
                        if pd.notna(importance):
                            importance = float(importance)
                        else:
                            importance = None

                        relationships.append((
                            row['occupation_id'],
                            row['function_id'],
                            environment_id,
                            physicality_id,
                            cognitive_id,
                            importance
                        ))

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_function 
                           (occupation_id, function_id, environment_id, physicality_id, cognitive_id, importance_score) 
                           VALUES (%s, %s, %s, %s, %s, %s) 
                           ON CONFLICT (occupation_id, function_id) DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-function relationships")
                    total_relationships += len(relationships)

                # Education relationships (no levels)
                if 'occupation_education_relationships' in self.mapped_data:
                    df = self.mapped_data['occupation_education_relationships']
                    logger.info(
                        f"⏳ Processing {len(df)} occupation-education relationships...")

                    relationships = [
                        (row['source_id'], row['target_id'])
                        for _, row in df.iterrows()
                    ]

                    execute_batch(
                        cursor,
                        """INSERT INTO occupation_education (occupation_id, education_level_id) 
                           VALUES (%s, %s) 
                           ON CONFLICT DO NOTHING""",
                        relationships,
                        page_size=1000
                    )
                    logger.info(
                        f"✓ Inserted {len(relationships)} occupation-education relationships")
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
        """Insert embedding-based inferred relationships between core entities."""
        try:
            logger.info("=" * 70)
            logger.info("INSERTING INFERRED RELATIONSHIPS")
            logger.info("-" * 70)

            if not self.relationships_dir.exists():
                logger.warning(
                    f"Relationships directory not found: {self.relationships_dir}")
                return True

            total_inferred_relationships = 0

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Define relationship mappings
                relationship_configs = [
                    ('knowledge_skill_inferred.csv',
                     'knowledge_skill', 'knowledge_id', 'skill_id'),
                    ('knowledge_function_inferred.csv',
                     'knowledge_function', 'knowledge_id', 'function_id'),
                    ('skill_ability_inferred.csv',
                     'skill_ability', 'skill_id', 'ability_id'),
                    ('ability_task_inferred.csv',
                     'ability_task', 'ability_id', 'task_id'),
                    ('function_task_inferred.csv',
                     'function_task', 'function_id', 'task_id')
                ]

                # Build entity name lookups
                lookups = {}
                for entity_type in ['knowledge', 'skill', 'ability', 'function', 'task']:
                    lookups[entity_type] = {}
                    df_key = f"{entity_type}_mapped"
                    if df_key in self.mapped_data:
                        df = self.mapped_data[df_key]
                        lookups[entity_type] = dict(zip(df['name'], df['id']))

                # Process each relationship type
                for csv_file, table_name, source_col, target_col in relationship_configs:
                    csv_path = self.relationships_dir / csv_file

                    if not csv_path.exists():
                        logger.info(f"Skipping {csv_file} (file not found)")
                        continue

                    rel_df = pd.read_csv(csv_path)
                    logger.info(
                        f"⏳ Processing {len(rel_df)} {table_name} relationships from {csv_file}...")

                    source_type = source_col.replace('_id', '')
                    target_type = target_col.replace('_id', '')

                    # Map names to UUIDs
                    valid_relationships = []
                    for _, row in rel_df.iterrows():
                        source_name = row['source_name']
                        target_name = row['target_name']
                        confidence_score = row.get('confidence_score')

                        source_id = lookups[source_type].get(source_name)
                        target_id = lookups[target_type].get(target_name)

                        if source_id and target_id:
                            valid_relationships.append((
                                source_id,
                                target_id,
                                confidence_score
                            ))

                    logger.info(
                        f"  Found {len(valid_relationships)} valid relationships")

                    # Insert relationships with confidence scores
                    if valid_relationships:
                        insert_query = f"""
                            INSERT INTO {table_name} ({source_col}, {target_col}, confidence_score) 
                            VALUES (%s, %s, %s)
                            ON CONFLICT ({source_col}, {target_col}) DO NOTHING
                        """

                        execute_batch(cursor, insert_query,
                                      valid_relationships, page_size=1000)

                        logger.info(
                            f"✓ Inserted {len(valid_relationships)} {table_name} relationships")
                        total_inferred_relationships += len(
                            valid_relationships)

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
        reports_dir = self.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = reports_dir / "database_load_metadata.json"
        logger.info(f"⏳ Writing metadata to {metadata_path}...")

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved successfully")
        logger.info("=" * 70)

    def load_all_data(self) -> bool:
        """Complete data loading pipeline."""
        logger.info("="*70)
        logger.info("STARTING O*NET DATA LOADING PIPELINE")
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
        success &= self.insert_task_entities()
        success &= self.insert_function_entities()
        success &= self.insert_education_levels()

        # Insert O*NET occupation relationships
        logger.info("\n" + "="*70)
        logger.info("INSERTING RELATIONSHIPS")
        logger.info("="*70)

        success &= self.insert_occupation_relationships()

        # Insert inferred relationships
        success &= self.insert_inferred_relationships()

        # Save metadata
        self._save_load_metadata()

        if success:
            logger.info("\n" + "="*70)
            logger.info("O*NET DATA LOADING COMPLETED SUCCESSFULLY!")
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
    def from_environment(cls) -> 'ONetLoader':
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
    """Main function to run the O*NET loader."""
    logger.info("=" * 70)
    logger.info("O*NET LOADER - MAIN EXECUTION")
    logger.info("=" * 70)

    try:
        # Load database config from environment
        loader = ONetLoader.from_environment()

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
