"""
O*NET Database Loader for KSAMDS Project

This module handles loading the mapped O*NET data into the PostgreSQL
KSAMDS database. It manages database connections, inserts entities,
creates relationships, and handles constraint violations gracefully.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    def __init__(self, db_config: DatabaseConfig, data_dir: str = "data"):
        """
        Initialize the O*NET loader.

        Args:
            db_config: Database connection configuration
            data_dir: Base directory containing mapped data
        """
        self.db_config = db_config
        self.data_dir = Path(data_dir)
        self.mapped_dir = self.data_dir / "processed" / "ksamds"

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
            'relationships': 0,
            'errors': 0
        }

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

            # Load ID mappings
            id_mapping_path = self.mapped_dir / "id_mappings.csv"
            if id_mapping_path.exists():
                mapping_df = pd.read_csv(id_mapping_path)
                self.id_mappings = dict(
                    zip(mapping_df['onet_id'], mapping_df['ksamds_id']))
                logger.info(f"Loaded {len(self.id_mappings)} ID mappings")

            return len(self.mapped_data) > 0

        except Exception as e:
            logger.error(f"Failed to load mapped data: {e}")
            return False

    def load_dimension_lookups(self) -> bool:
        """Load dimension ID lookups from database."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Load all dimension tables
                for dim_table in self.dimension_lookups.keys():
                    if dim_table in ['type_dim', 'level_dim', 'basis_dim', 'environment_dim']:
                        cursor.execute(
                            f"SELECT id, scope, name FROM {dim_table}")
                        for row in cursor.fetchall():
                            # Use just the name without scope prefix since mapper outputs clean names
                            self.dimension_lookups[dim_table][row['name']
                                                              ] = row['id']
                    elif dim_table == 'mode_dim':
                        cursor.execute(f"SELECT id, name FROM {dim_table}")
                        for row in cursor.fetchall():
                            self.dimension_lookups[dim_table][row['name']
                                                              ] = row['id']
                    else:  # physicality_dim, cognitive_dim
                        cursor.execute(f"SELECT id, name FROM {dim_table}")
                        for row in cursor.fetchall():
                            self.dimension_lookups[dim_table][row['name']
                                                              ] = row['id']

                logger.info("Loaded dimension lookups")
                return True

        except Exception as e:
            logger.error(f"Failed to load dimension lookups: {e}")
            return False

    def initialize_dimensions(self) -> bool:
        """Initialize dimension tables with default values."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()

                # Set schema
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Type dimensions - clean names without prefixes
                type_dims = [
                    ('K', 'Technical', 'Technical knowledge and expertise'),
                    ('K', 'Analytical', 'Analytical and quantitative knowledge'),
                    ('K', 'Business', 'Business and commercial knowledge'),
                    ('K', 'Management', 'Management and leadership knowledge'),
                    ('K', 'Social', 'Social and interpersonal knowledge'),
                    ('K', 'Scientific', 'Scientific and research knowledge'),
                    ('K', 'Safety', 'Safety and regulatory knowledge'),
                    ('K', 'General', 'General knowledge'),
                    ('S', 'Technical', 'Technical skills and expertise'),
                    ('S', 'Analytical', 'Analytical and problem-solving skills'),
                    ('S', 'Communication', 'Communication and presentation skills'),
                    ('S', 'Management', 'Management and leadership skills'),
                    ('S', 'Social', 'Social and interpersonal skills'),
                    ('S', 'General', 'General skills'),
                    ('A', 'Cognitive', 'Cognitive and mental abilities'),
                    ('A', 'Physical', 'Physical and motor abilities'),
                    ('A', 'Sensory', 'Sensory and perceptual abilities'),
                    ('A', 'General', 'General abilities'),
                    ('T', 'Manual', 'Manual and hands-on tasks'),
                    ('T', 'Analytical', 'Analytical and cognitive tasks'),
                    ('T', 'Social', 'Social and interpersonal tasks')
                ]

                for scope, name, desc in type_dims:
                    cursor.execute(
                        """INSERT INTO type_dim (id, scope, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (scope, name, desc)
                    )

                # Level dimensions - clean names without prefixes
                level_dims = [
                    ('K', 'Basic', 'Fundamental knowledge level'),
                    ('K', 'Intermediate', 'Intermediate knowledge level'),
                    ('K', 'Advanced', 'Advanced knowledge level'),
                    ('K', 'Expert', 'Expert knowledge level'),
                    ('S', 'Basic', 'Fundamental skill level'),
                    ('S', 'Intermediate', 'Intermediate skill level'),
                    ('S', 'Advanced', 'Advanced skill level'),
                    ('S', 'Expert', 'Expert skill level'),
                    ('A', 'Basic', 'Fundamental ability level'),
                    ('A', 'Intermediate', 'Intermediate ability level'),
                    ('A', 'Advanced', 'Advanced ability level'),
                    ('A', 'Expert', 'Expert ability level')
                ]

                for scope, name, desc in level_dims:
                    cursor.execute(
                        """INSERT INTO level_dim (id, scope, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (scope, name, desc)
                    )

                # Basis dimensions - clean names without prefixes
                basis_dims = [
                    ('K', 'Formal_Education',
                     'Knowledge acquired through formal education'),
                    ('K', 'Training', 'Knowledge acquired through training programs'),
                    ('K', 'Experience', 'Knowledge acquired through work experience'),
                    ('K', 'Certification',
                     'Knowledge acquired through certification programs'),
                    ('S', 'Formal_Education',
                     'Skills acquired through formal education'),
                    ('S', 'On_the_Job_Training',
                     'Skills acquired through on-the-job training'),
                    ('S', 'Experience', 'Skills acquired through work experience'),
                    ('S', 'Certification',
                     'Skills acquired through certification programs'),
                    ('A', 'Natural', 'Natural or innate abilities'),
                    ('A', 'Developed', 'Abilities developed through practice'),
                    ('A', 'Training', 'Abilities developed through training')
                ]

                for scope, name, desc in basis_dims:
                    cursor.execute(
                        """INSERT INTO basis_dim (id, scope, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (scope, name, desc)
                    )

                # Environment dimensions - clean names for tasks and functions
                env_dims = [
                    ('F', 'Office', 'Office environment'),
                    ('F', 'Outdoor', 'Outdoor environment'),
                    ('F', 'Laboratory', 'Laboratory environment'),
                    ('F', 'Factory', 'Manufacturing environment'),
                    ('F', 'Remote', 'Remote work environment'),
                    ('T', 'Office_Tasks', 'Office-based tasks'),
                    ('T', 'Field_Tasks', 'Field-based tasks'),
                    ('T', 'Laboratory_Tasks', 'Laboratory-based tasks'),
                    ('T', 'Factory', 'Factory-based tasks'),
                    ('T', 'Remote', 'Remote tasks')
                ]

                for scope, name, desc in env_dims:
                    cursor.execute(
                        """INSERT INTO environment_dim (id, scope, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (scope, name, desc)
                    )

                # Mode dimensions
                mode_dims = [
                    ('Tool', 'Tool-based tasks'),
                    ('Process', 'Process-oriented tasks'),
                    ('Theory', 'Theoretical tasks'),
                    ('Creative', 'Creative tasks')
                ]

                for name, desc in mode_dims:
                    cursor.execute(
                        """INSERT INTO mode_dim (id, scope, name, description) 
                           VALUES (uuid_generate_v4(), 'T', %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (name, desc)
                    )

                # Physicality dimensions
                phys_dims = [
                    ('Light', 'Light physical demands'),
                    ('Moderate', 'Moderate physical demands'),
                    ('Heavy', 'Heavy physical demands')
                ]

                for name, desc in phys_dims:
                    cursor.execute(
                        """INSERT INTO physicality_dim (id, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (name, desc)
                    )

                # Cognitive dimensions
                cog_dims = [
                    ('Light', 'Light cognitive demands'),
                    ('Moderate', 'Moderate cognitive demands'),
                    ('Heavy', 'Heavy cognitive demands')
                ]

                for name, desc in cog_dims:
                    cursor.execute(
                        """INSERT INTO cognitive_dim (id, name, description) 
                           VALUES (uuid_generate_v4(), %s, %s) 
                           ON CONFLICT (name) DO NOTHING""",
                        (name, desc)
                    )

                conn.commit()
                logger.info("Initialized dimension tables")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize dimensions: {e}")
            return False

    def insert_knowledge_entities(self) -> bool:
        """Insert knowledge entities into the database."""
        if 'knowledge_mapped' not in self.mapped_data:
            logger.warning("No knowledge data to insert")
            return True

        try:
            df = self.mapped_data['knowledge_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Get existing knowledge names to avoid conflicts
                cursor.execute("SELECT name FROM knowledge")
                existing_knowledge = {row[0] for row in cursor.fetchall()}

                # Prepare data for batch insert
                knowledge_data = []
                junction_data = {
                    'type': [],
                    'level': [],
                    'basis': []
                }

                for _, row in df.iterrows():
                    knowledge_id = row['id']
                    knowledge_name = row['name']

                    # Skip if knowledge already exists
                    if knowledge_name in existing_knowledge:
                        logger.debug(
                            f"Skipping existing knowledge: {knowledge_name}")
                        continue

                    # Core entity
                    knowledge_data.append((
                        knowledge_id,
                        knowledge_name,
                        row.get('source_ref', 'O*NET')
                    ))

                    # Process dimensions only for knowledge that will be inserted
                    if pd.notna(row.get('type_dims')):
                        for type_name in row['type_dims'].split('|'):
                            type_name_clean = type_name.strip()
                            if type_name_clean in self.dimension_lookups['type_dim']:
                                junction_data['type'].append((
                                    knowledge_id,
                                    self.dimension_lookups['type_dim'][type_name_clean]
                                ))

                    if pd.notna(row.get('level_dims')):
                        for level_name in row['level_dims'].split('|'):
                            level_name_clean = level_name.strip()
                            if level_name_clean in self.dimension_lookups['level_dim']:
                                junction_data['level'].append((
                                    knowledge_id,
                                    self.dimension_lookups['level_dim'][level_name_clean]
                                ))

                    if pd.notna(row.get('basis_dims')):
                        for basis_name in row['basis_dims'].split('|'):
                            basis_name_clean = basis_name.strip()
                            if basis_name_clean in self.dimension_lookups['basis_dim']:
                                junction_data['basis'].append((
                                    knowledge_id,
                                    self.dimension_lookups['basis_dim'][basis_name_clean]
                                ))

                # Insert core entities
                if knowledge_data:
                    execute_batch(
                        cursor,
                        """INSERT INTO knowledge (id, name, source_ref) 
                        VALUES (%s, %s, %s)""",
                        knowledge_data,
                        page_size=1000
                    )

                # Insert junction tables
                if junction_data['type']:
                    execute_batch(
                        cursor,
                        """INSERT INTO knowledge_type (knowledge_id, type_id) VALUES (%s, %s)""",
                        junction_data['type'],
                        page_size=1000
                    )

                if junction_data['level']:
                    execute_batch(
                        cursor,
                        """INSERT INTO knowledge_level (knowledge_id, level_id) VALUES (%s, %s)""",
                        junction_data['level'],
                        page_size=1000
                    )

                if junction_data['basis']:
                    execute_batch(
                        cursor,
                        """INSERT INTO knowledge_basis (knowledge_id, basis_id) VALUES (%s, %s)""",
                        junction_data['basis'],
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['knowledge'] = len(knowledge_data)
                logger.info(
                    f"Inserted {len(knowledge_data)} knowledge entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert knowledge entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_skill_entities(self) -> bool:
        """Insert skill entities into the database."""
        if 'skill_mapped' not in self.mapped_data:
            logger.warning("No skill data to insert")
            return True

        try:
            df = self.mapped_data['skill_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Get existing skill names to avoid conflicts
                cursor.execute("SELECT name FROM skill")
                existing_skills = {row[0] for row in cursor.fetchall()}

                # Prepare data for batch insert
                skill_data = []
                junction_data = {
                    'type': [],
                    'level': [],
                    'basis': []
                }

                for _, row in df.iterrows():
                    skill_id = row['id']
                    skill_name = row['name']

                    # Skip if skill already exists
                    if skill_name in existing_skills:
                        logger.debug(f"Skipping existing skill: {skill_name}")
                        continue

                    # Core entity
                    skill_data.append((
                        skill_id,
                        skill_name,
                        row.get('source_ref', 'O*NET')
                    ))

                    # Process dimensions only for skills that will be inserted
                    if pd.notna(row.get('type_dims')):
                        for type_name in row['type_dims'].split('|'):
                            type_name_clean = type_name.strip()
                            if type_name_clean in self.dimension_lookups['type_dim']:
                                junction_data['type'].append((
                                    skill_id,
                                    self.dimension_lookups['type_dim'][type_name_clean]
                                ))

                    if pd.notna(row.get('level_dims')):
                        for level_name in row['level_dims'].split('|'):
                            level_name_clean = level_name.strip()
                            if level_name_clean in self.dimension_lookups['level_dim']:
                                junction_data['level'].append((
                                    skill_id,
                                    self.dimension_lookups['level_dim'][level_name_clean]
                                ))

                    if pd.notna(row.get('basis_dims')):
                        for basis_name in row['basis_dims'].split('|'):
                            basis_name_clean = basis_name.strip()
                            if basis_name_clean in self.dimension_lookups['basis_dim']:
                                junction_data['basis'].append((
                                    skill_id,
                                    self.dimension_lookups['basis_dim'][basis_name_clean]
                                ))

                # Insert core entities
                if skill_data:
                    execute_batch(
                        cursor,
                        """INSERT INTO skill (id, name, source_ref) 
                        VALUES (%s, %s, %s)""",
                        skill_data,
                        page_size=1000
                    )

                # Insert junction tables
                if junction_data['type']:
                    execute_batch(
                        cursor,
                        """INSERT INTO skill_type (skill_id, type_id) VALUES (%s, %s)""",
                        junction_data['type'],
                        page_size=1000
                    )

                if junction_data['level']:
                    execute_batch(
                        cursor,
                        """INSERT INTO skill_level (skill_id, level_id) VALUES (%s, %s)""",
                        junction_data['level'],
                        page_size=1000
                    )

                if junction_data['basis']:
                    execute_batch(
                        cursor,
                        """INSERT INTO skill_basis (skill_id, basis_id) VALUES (%s, %s)""",
                        junction_data['basis'],
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['skill'] = len(skill_data)
                logger.info(f"Inserted {len(skill_data)} skill entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert skill entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_ability_entities(self) -> bool:
        """Insert ability entities into the database."""
        if 'ability_mapped' not in self.mapped_data:
            logger.warning("No ability data to insert")
            return True

        try:
            df = self.mapped_data['ability_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Get existing ability names to avoid conflicts
                cursor.execute("SELECT name FROM ability")
                existing_abilities = {row[0] for row in cursor.fetchall()}

                # Prepare data for batch insert
                ability_data = []
                junction_data = {
                    'type': [],
                    'level': [],
                    'basis': []
                }

                for _, row in df.iterrows():
                    ability_id = row['id']
                    ability_name = row['name']

                    # Skip if ability already exists
                    if ability_name in existing_abilities:
                        logger.debug(
                            f"Skipping existing ability: {ability_name}")
                        continue

                    # Core entity
                    ability_data.append((
                        ability_id,
                        ability_name,
                        row.get('source_ref', 'O*NET')
                    ))

                    # Process dimensions only for abilities that will be inserted
                    if pd.notna(row.get('type_dims')):
                        for type_name in row['type_dims'].split('|'):
                            type_name_clean = type_name.strip()
                            if type_name_clean in self.dimension_lookups['type_dim']:
                                junction_data['type'].append((
                                    ability_id,
                                    self.dimension_lookups['type_dim'][type_name_clean]
                                ))

                    if pd.notna(row.get('level_dims')):
                        for level_name in row['level_dims'].split('|'):
                            level_name_clean = level_name.strip()
                            if level_name_clean in self.dimension_lookups['level_dim']:
                                junction_data['level'].append((
                                    ability_id,
                                    self.dimension_lookups['level_dim'][level_name_clean]
                                ))

                    if pd.notna(row.get('basis_dims')):
                        for basis_name in row['basis_dims'].split('|'):
                            basis_name_clean = basis_name.strip()
                            if basis_name_clean in self.dimension_lookups['basis_dim']:
                                junction_data['basis'].append((
                                    ability_id,
                                    self.dimension_lookups['basis_dim'][basis_name_clean]
                                ))

                # Insert core entities
                if ability_data:
                    execute_batch(
                        cursor,
                        """INSERT INTO ability (id, name, source_ref) 
                        VALUES (%s, %s, %s)""",
                        ability_data,
                        page_size=1000
                    )

                # Insert junction tables
                if junction_data['type']:
                    execute_batch(
                        cursor,
                        """INSERT INTO ability_type (ability_id, type_id) VALUES (%s, %s)""",
                        junction_data['type'],
                        page_size=1000
                    )

                if junction_data['level']:
                    execute_batch(
                        cursor,
                        """INSERT INTO ability_level (ability_id, level_id) VALUES (%s, %s)""",
                        junction_data['level'],
                        page_size=1000
                    )

                if junction_data['basis']:
                    execute_batch(
                        cursor,
                        """INSERT INTO ability_basis (ability_id, basis_id) VALUES (%s, %s)""",
                        junction_data['basis'],
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['ability'] = len(ability_data)
                logger.info(f"Inserted {len(ability_data)} ability entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert ability entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_task_entities(self) -> bool:
        """Insert task entities into the database."""
        if 'task_mapped' not in self.mapped_data:
            logger.warning("No task data to insert")
            return True

        try:
            df = self.mapped_data['task_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Get existing task names to avoid conflicts
                cursor.execute("SELECT name FROM task")
                existing_tasks = {row[0] for row in cursor.fetchall()}

                # Prepare data for batch insert
                task_data = []
                junction_data = {
                    'type': [],
                    'environment': [],
                    'mode': []
                }

                successful_task_ids = set()

                for _, row in df.iterrows():
                    task_id = row['id']
                    task_name = row['name']

                    # Skip if task already exists
                    if task_name in existing_tasks:
                        logger.debug(f"Skipping existing task: {task_name}")
                        continue

                    # Core entity
                    task_data.append((
                        task_id,
                        task_name,
                        row.get('source_ref', 'O*NET'),
                    ))

                    successful_task_ids.add(task_id)

                    # Process dimensions only for tasks that will be inserted
                    if pd.notna(row.get('type_dims')):
                        for type_name in row['type_dims'].split('|'):
                            type_name_clean = type_name.strip()
                            if type_name_clean in self.dimension_lookups['type_dim']:
                                junction_data['type'].append((
                                    task_id,
                                    self.dimension_lookups['type_dim'][type_name_clean]
                                ))

                    if pd.notna(row.get('environment_dims')):
                        for env_name in row['environment_dims'].split('|'):
                            env_name_clean = env_name.strip()
                            if env_name_clean in self.dimension_lookups['environment_dim']:
                                junction_data['environment'].append((
                                    task_id,
                                    self.dimension_lookups['environment_dim'][env_name_clean]
                                ))

                    if pd.notna(row.get('mode_dims')):
                        for mode_name in row['mode_dims'].split('|'):
                            mode_name_clean = mode_name.strip()
                            if mode_name_clean in self.dimension_lookups['mode_dim']:
                                junction_data['mode'].append((
                                    task_id,
                                    self.dimension_lookups['mode_dim'][mode_name_clean]
                                ))

                # Insert core entities (without ON CONFLICT since we pre-filtered)
                if task_data:
                    execute_batch(
                        cursor,
                        """INSERT INTO task (id, name, source_ref) VALUES (%s, %s, %s)""",
                        task_data,
                        page_size=1000
                    )

                # Insert junction tables only for successfully inserted tasks
                if junction_data['type']:
                    execute_batch(
                        cursor,
                        """INSERT INTO task_type (task_id, type_id) VALUES (%s, %s)""",
                        junction_data['type'],
                        page_size=1000
                    )

                if junction_data['environment']:
                    execute_batch(
                        cursor,
                        """INSERT INTO task_env (task_id, environment_id) VALUES (%s, %s)""",
                        junction_data['environment'],
                        page_size=1000
                    )

                if junction_data['mode']:
                    execute_batch(
                        cursor,
                        """INSERT INTO task_mode (task_id, mode_id) VALUES (%s, %s)""",
                        junction_data['mode'],
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['task'] = len(task_data)
                logger.info(f"Inserted {len(task_data)} task entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert task entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_function_entities(self) -> bool:
        """Insert function entities into the database."""
        if 'function_mapped' not in self.mapped_data:
            logger.warning("No function data to insert")
            return True

        try:
            df = self.mapped_data['function_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Get existing function names to avoid conflicts
                cursor.execute("SELECT name FROM function")
                existing_functions = {row[0] for row in cursor.fetchall()}

                # Prepare data for batch insert
                function_data = []
                junction_data = {
                    'environment': [],
                    'physicality': [],
                    'cognitive': []
                }

                for _, row in df.iterrows():
                    function_id = row['id']
                    function_name = row['name']

                    # Skip if function already exists
                    if function_name in existing_functions:
                        logger.debug(
                            f"Skipping existing function: {function_name}")
                        continue

                    # Core entity
                    function_data.append((
                        function_id,
                        function_name,
                        row.get('source_ref', 'O*NET'),
                    ))

                    # Process dimensions only for functions that will be inserted
                    if pd.notna(row.get('environment_dims')):
                        for env_name in row['environment_dims'].split('|'):
                            env_name_clean = env_name.strip()
                            if env_name_clean in self.dimension_lookups['environment_dim']:
                                junction_data['environment'].append((
                                    function_id,
                                    self.dimension_lookups['environment_dim'][env_name_clean]
                                ))

                    if pd.notna(row.get('physicality_dims')):
                        for phys_name in row['physicality_dims'].split('|'):
                            phys_name_clean = phys_name.strip()
                            if phys_name_clean in self.dimension_lookups['physicality_dim']:
                                junction_data['physicality'].append((
                                    function_id,
                                    self.dimension_lookups['physicality_dim'][phys_name_clean]
                                ))

                    if pd.notna(row.get('cognitive_dims')):
                        for cog_name in row['cognitive_dims'].split('|'):
                            cog_name_clean = cog_name.strip()
                            if cog_name_clean in self.dimension_lookups['cognitive_dim']:
                                junction_data['cognitive'].append((
                                    function_id,
                                    self.dimension_lookups['cognitive_dim'][cog_name_clean]
                                ))

                # Insert core entities
                if function_data:
                    execute_batch(
                        cursor,
                        """INSERT INTO function (id, name, source_ref) VALUES (%s, %s, %s)""",
                        function_data,
                        page_size=1000
                    )

                # Insert junction tables
                if junction_data['environment']:
                    execute_batch(
                        cursor,
                        """INSERT INTO function_env (function_id, environment_id) VALUES (%s, %s)""",
                        junction_data['environment'],
                        page_size=1000
                    )

                if junction_data['physicality']:
                    execute_batch(
                        cursor,
                        """INSERT INTO function_physicality (function_id, physicality_id) VALUES (%s, %s)""",
                        junction_data['physicality'],
                        page_size=1000
                    )

                if junction_data['cognitive']:
                    execute_batch(
                        cursor,
                        """INSERT INTO function_cognitive (function_id, cognitive_id) VALUES (%s, %s)""",
                        junction_data['cognitive'],
                        page_size=1000
                    )

                conn.commit()
                self.insertion_stats['function'] = len(function_data)
                logger.info(f"Inserted {len(function_data)} function entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert function entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_occupation_entities(self) -> bool:
        """Insert occupation entities into the database."""
        if 'occupation_mapped' not in self.mapped_data:
            logger.warning("No occupation data to insert")
            return True

        try:
            df = self.mapped_data['occupation_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Prepare data for batch insert
                occupation_data = []

                for _, row in df.iterrows():
                    occupation_data.append((
                        row['id'],
                        row['name'],
                        row.get('definition', ''),
                        row.get('source_ref', 'O*NET')
                    ))

                # Insert core entities
                execute_batch(
                    cursor,
                    """INSERT INTO occupation (id, title, description, source_ref) 
                       VALUES (%s, %s, %s, %s) ON CONFLICT (title) DO NOTHING""",
                    occupation_data,
                    page_size=1000
                )

                conn.commit()
                self.insertion_stats['occupation'] = len(occupation_data)
                logger.info(
                    f"Inserted {len(occupation_data)} occupation entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert occupation entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_education_levels(self) -> bool:
        """Insert education level entities into the database."""
        if 'education_level_mapped' not in self.mapped_data:
            logger.warning("No education level data to insert")
            return True

        try:
            df = self.mapped_data['education_level_mapped']

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Prepare data for batch insert
                education_data = []

                for _, row in df.iterrows():
                    education_data.append((
                        row['id'],
                        row['name']
                    ))

                # Insert core entities
                execute_batch(
                    cursor,
                    """INSERT INTO education_level (id, name) 
                       VALUES (%s, %s) ON CONFLICT (name) DO NOTHING""",
                    education_data,
                    page_size=1000
                )

                conn.commit()
                self.insertion_stats['education_level'] = len(education_data)
                logger.info(
                    f"Inserted {len(education_data)} education level entities")
                return True

        except Exception as e:
            logger.error(f"Failed to insert education level entities: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_relationships(self) -> bool:
        """Insert occupation relationships into the database."""
        logger.info("Loading occupation relationships...")

        # Correct table and column mappings
        relationship_configs = [
            ('occupation_knowledge_relationships.csv',
             'occupation_knowledge', 'knowledge_id'),
            ('occupation_skill_relationships.csv', 'occupation_skill', 'skill_id'),
            ('occupation_ability_relationships.csv',
             'occupation_ability', 'ability_id'),
            ('occupation_task_relationships.csv', 'occupation_task', 'task_id'),
            ('occupation_function_relationships.csv',
             'occupation_function', 'function_id')
        ]

        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                for rel_file, table_name, entity_column in relationship_configs:
                    rel_path = self.mapped_dir / rel_file
                    if not rel_path.exists():
                        logger.warning(
                            f"Relationship file not found: {rel_file}")
                        continue

                    # Load relationships
                    rel_df = pd.read_csv(rel_path)
                    logger.info(
                        f"Processing {len(rel_df)} {table_name} relationships")

                    if len(rel_df) == 0:
                        continue

                    # Validate that both IDs exist in database before inserting
                    valid_relationships = []

                    for _, row in rel_df.iterrows():
                        occupation_id = row['entity1_id']
                        entity_id = row['entity2_id']

                        # Check if occupation exists
                        cursor.execute(
                            "SELECT 1 FROM occupation WHERE id = %s", (occupation_id,))
                        if not cursor.fetchone():
                            continue

                        # Check if entity exists
                        # knowledge, skill, ability, etc.
                        entity_table = table_name.split('_')[1]
                        cursor.execute(
                            f"SELECT 1 FROM {entity_table} WHERE id = %s", (entity_id,))
                        if not cursor.fetchone():
                            continue

                        valid_relationships.append((occupation_id, entity_id))

                    logger.info(
                        f"Found {len(valid_relationships)} valid relationships out of {len(rel_df)}")

                    if valid_relationships:
                        # Use correct column names
                        insert_query = f"""
                            INSERT INTO {table_name} (occupation_id, {entity_column}) 
                            VALUES (%s, %s)
                            ON CONFLICT (occupation_id, {entity_column}) DO NOTHING
                        """

                        execute_batch(cursor, insert_query,
                                      valid_relationships, page_size=1000)

                        logger.info(
                            f"Inserted {len(valid_relationships)} {table_name} relationships")
                        self.insertion_stats['relationships'] += len(
                            valid_relationships)

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert relationships: {e}")
            self.insertion_stats['errors'] += 1
            return False

    def insert_inferred_relationships(self) -> bool:
        """Insert inferred relationships (K→S, S→A, K→F, A→T, F→T) into the database."""
        logger.info("Loading inferred relationships from embeddings...")

        # Map relationship types to their database tables
        relationship_configs = [
            ('knowledge_skill_inferred.csv',
             'knowledge_skill', 'knowledge_id', 'skill_id'),
            ('skill_ability_inferred.csv', 'skill_ability', 'skill_id', 'ability_id'),
            ('knowledge_function_inferred.csv',
             'knowledge_function', 'knowledge_id', 'function_id'),
            ('ability_task_inferred.csv', 'ability_task', 'ability_id', 'task_id'),
            ('function_task_inferred.csv', 'function_task', 'function_id', 'task_id')
        ]

        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")

                # Build lookup dictionaries for name → UUID mapping
                logger.info("Building entity name lookups...")

                # Knowledge lookup
                cursor.execute("SELECT id, name FROM knowledge")
                knowledge_lookup = {row[1]: row[0]
                                    for row in cursor.fetchall()}

                # Skill lookup
                cursor.execute("SELECT id, name FROM skill")
                skill_lookup = {row[1]: row[0] for row in cursor.fetchall()}

                # Ability lookup
                cursor.execute("SELECT id, name FROM ability")
                ability_lookup = {row[1]: row[0] for row in cursor.fetchall()}

                # Function lookup
                cursor.execute("SELECT id, name FROM function")
                function_lookup = {row[1]: row[0] for row in cursor.fetchall()}

                # Task lookup
                cursor.execute("SELECT id, name FROM task")
                task_lookup = {row[1]: row[0] for row in cursor.fetchall()}

                # Combine into one dictionary for easier access
                entity_lookups = {
                    'knowledge': knowledge_lookup,
                    'skill': skill_lookup,
                    'ability': ability_lookup,
                    'function': function_lookup,
                    'task': task_lookup
                }

                logger.info("Entity lookups built successfully")

                # Process each relationship type
                for rel_file, table_name, source_col, target_col in relationship_configs:
                    rel_path = self.mapped_dir.parent / rel_file  # data/processed/

                    if not rel_path.exists():
                        logger.warning(
                            f"Inferred relationship file not found: {rel_file}")
                        continue

                    # Load inferred relationships
                    rel_df = pd.read_csv(rel_path)
                    logger.info(
                        f"Processing {len(rel_df)} {table_name} relationships from {rel_file}")

                    if len(rel_df) == 0:
                        continue

                    # Extract source and target entity types from table name
                    source_type, target_type = table_name.split('_')

                    # Resolve names to UUIDs
                    valid_relationships = []
                    skipped_count = 0

                    for _, row in rel_df.iterrows():
                        source_name = row['source_name']
                        target_name = row['target_name']

                        # Look up UUIDs
                        source_id = entity_lookups[source_type].get(
                            source_name)
                        target_id = entity_lookups[target_type].get(
                            target_name)

                        if source_id and target_id:
                            valid_relationships.append((source_id, target_id))
                        else:
                            skipped_count += 1
                            if not source_id:
                                logger.debug(
                                    f"Source entity not found: {source_name}")
                            if not target_id:
                                logger.debug(
                                    f"Target entity not found: {target_name}")

                    logger.info(
                        f"Resolved {len(valid_relationships)} valid relationships (skipped {skipped_count})")

                    # Insert relationships
                    if valid_relationships:
                        insert_query = f"""
                            INSERT INTO {table_name} ({source_col}, {target_col}) 
                            VALUES (%s, %s)
                            ON CONFLICT ({source_col}, {target_col}) DO NOTHING
                        """

                        execute_batch(cursor, insert_query,
                                      valid_relationships, page_size=1000)

                        logger.info(
                            f"Inserted {len(valid_relationships)} {table_name} relationships")
                        self.insertion_stats['relationships'] += len(
                            valid_relationships)

                conn.commit()
                logger.info("All inferred relationships loaded successfully")
                return True

        except Exception as e:
            logger.error(
                f"Failed to insert inferred relationships: {e}", exc_info=True)
            self.insertion_stats['errors'] += 1
            return False

    def load_all_data(self, clear_inferred_first: bool = False) -> bool:
        """Complete data loading pipeline."""
        logger.info("Starting O*NET data loading pipeline...")

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
        success = True
        success &= self.insert_occupation_entities()
        success &= self.insert_knowledge_entities()
        success &= self.insert_skill_entities()
        success &= self.insert_ability_entities()
        success &= self.insert_task_entities()
        success &= self.insert_function_entities()
        success &= self.insert_education_levels()

        # Insert O*NET occupation relationships
        success &= self.insert_relationships()

        # Insert inferred relationships (NEW!)
        success &= self.insert_inferred_relationships()

        if success:
            logger.info("O*NET data loading completed successfully!")
            self._print_insertion_summary()
        else:
            logger.error("Data loading completed with errors")

        return success

    def _print_insertion_summary(self):
        """Print summary of insertion statistics."""
        print("\n=== Insertion Summary ===")
        for entity_type, count in self.insertion_stats.items():
            if entity_type != 'errors' and count > 0:
                print(f"{entity_type.upper()}: {count} entities inserted")

        if self.insertion_stats['errors'] > 0:
            print(f"ERRORS: {self.insertion_stats['errors']} errors occurred")

        print("Database loading completed!")

    @classmethod
    def from_environment(cls, data_dir: str = "data") -> 'ONetLoader':
        """Create loader instance from environment variables."""
        db_config = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'ksamds'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '8086'),
            schema=os.getenv('DB_SCHEMA', 'ksamds')
        )
        return cls(db_config, data_dir)


def main():
    """Main function to run the O*NET loader."""
    # Load database config from environment
    loader = ONetLoader.from_environment()

    success = loader.load_all_data()

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
