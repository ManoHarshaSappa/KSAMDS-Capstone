"""
O*NET Relationship Inference for KSAMDS Project

This module orchestrates the complete embedding-based relationship inference pipeline:
1. Generates embeddings for Knowledge, Skills, Abilities, Functions, and Tasks
2. Calculates cosine similarity between entity embeddings
3. Infers relationships based on configurable thresholds
4. Saves relationship files ready for database loading

Consolidates functionality from:
- embedding_generator.py
- similarity_calculator.py  
- embedding_relationship_builder.py

Uses Google AI API (models/gemini-embedding-001) for embedding generation
with caching, batch processing, and comprehensive error handling.

MODIFIED:
- Replaced dynamic (hash-based) cache keys with simple, static cache file names.
- Disabled cache cleanup in main() to ensure cache is persistent.
- MODIFIED: Changed cache directory to 'embedding_ksamds'.
- MODIFIED: Removed Google Colab logic, now uses .env file from project root.
"""

import os
import pickle
import json
import hashlib
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv  # --- ADDED THIS IMPORT ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def log_timed_event(message: str, start_time: Optional[float] = None) -> Optional[float]:
    """
    Log a message with timestamp and optional duration.

    Args:
        message: Message to log
        start_time: If provided, calculates duration and logs completion

    Returns:
        Current time if start_time is None, otherwise None
    """
    if start_time:
        duration = time.time() - start_time
        logger.info(f"✅ {message} (Duration: {duration:.2f} seconds)")
        return None
    else:
        logger.info(f"⏳ {message}")
        return time.time()


class ONetRelationshipBuilder:
    """
    Orchestrates the complete embedding-based relationship inference pipeline.

    Handles:
    - Embedding generation with caching
    - Similarity calculation between entities
    - Relationship inference with configurable thresholds
    - Report generation and metadata tracking
    """

    def __init__(
        self,
        embedding_model: str = "models/gemini-embedding-001",
        batch_size: int = 200,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        max_relationships: Optional[Dict[str, int]] = None,
        use_cache: bool = True
    ):
        """
        Initialize the relationship builder.

        Args:
            embedding_model: Google AI model name for embeddings
            batch_size: Number of texts to process per API call
            similarity_thresholds: Minimum similarity scores for each relationship type
            max_relationships: Maximum relationships per entity for each type
            use_cache: Whether to use cached embeddings
        """
        # Get project root directory (3 levels up from etl folder)
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.input_dir = self.project_root / "data/archive/mapped"
        self.output_dir = self.project_root / "data/archive/relationships"
        # --- MODIFICATION: Changed cache directory ---
        self.cache_dir = self.project_root / "data/archive/embedding_ksamds"
        self.reports_dir = self.project_root / "data/reports"

        # Create directories
        for directory in [self.input_dir, self.output_dir, self.cache_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Default similarity thresholds
        self.similarity_thresholds = similarity_thresholds or {
            'knowledge_skill': 0.70,
            'skill_ability': 0.65,
            'knowledge_function': 0.60,
            'ability_task': 0.65,
            'function_task': 0.75
        }

        # Default max relationships per entity
        self.max_relationships = max_relationships or {
            'knowledge_skill': 15,
            'skill_ability': 10,
            'knowledge_function': 20,
            'ability_task': 25,
            'function_task': 10
        }

        # Track cache files created in this session
        self._session_cache_files = set()

        # Statistics tracking
        self._stats = {
            'embedding': {
                'total_embeddings_generated': 0,
                'total_api_calls': 0,
                'cache_hits': 0,
                'cache_misses': 0
            },
            'calculation': {
                'total_relationships_inferred': 0,
                'relationships_by_type': {},
                'calculation_time_by_type': {}
            }
        }

        # Pipeline metadata
        self.metadata = {
            'pipeline_name': 'ONetRelationshipBuilder',
            'embedding_model': embedding_model,
            'batch_size': batch_size,
            'similarity_thresholds': self.similarity_thresholds,
            'max_relationships': self.max_relationships,
            'start_time': None,
            'end_time': None,
            'entity_counts': {},
            'relationship_counts': {},
            'status': 'initialized'
        }

        # Configure Google AI API
        self._configure_google_ai()

        logger.info("=" * 70)
        logger.info("O*NET RELATIONSHIP BUILDER INITIALIZED")
        logger.info("-" * 70)
        logger.info(f"Embedding Model: {self.embedding_model}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")
        logger.info(f"Input Directory: {self.input_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        logger.info("-" * 70)
        logger.info("Similarity Thresholds:")
        for rel_type, threshold in self.similarity_thresholds.items():
            logger.info(f"  {rel_type}: {threshold}")
        logger.info("-" * 70)
        logger.info("Max Relationships:")
        for rel_type, max_val in self.max_relationships.items():
            logger.info(f"  {rel_type}: {max_val}")
        logger.info("=" * 70)

    # --- FUNCTION REPLACED ---
    def _configure_google_ai(self):
        """Configure Google AI API with proper credentials from .env file."""
        
        # Load .env file from project root
        dotenv_path = self.project_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(f"Loaded environment variables from {dotenv_path}")
        else:
            logger.warning(f".env file not found at {dotenv_path}")
        
        api_key = os.environ.get('GOOGLE_API_KEY')

        if api_key:
            genai.configure(api_key=api_key)
            logger.info("Configured Google AI with GOOGLE_API_KEY")
        else:
            logger.error(
                "Could not configure Google AI. Please set GOOGLE_API_KEY in your .env file.")
            raise ValueError("GOOGLE_API_KEY not found in environment.")
    # --- END OF REPLACEMENT ---

    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================

    def _create_entity_text(self, name: str, entity_type: str) -> str:
        """Create rich text representation for embedding generation."""
        return f"{entity_type}: {name}"

    # --- MODIFICATION: Removed _generate_cache_key function ---

    def _get_cache_path(self, entity_type: str) -> Path:
        """
        Get full path for a cache file using a simple, static name.
        """
        # --- MODIFICATION: Use a simple, static name ---
        model_suffix = self.embedding_model.split('/')[-1]
        return self.cache_dir / f"inferred_{entity_type}_embeddings_{model_suffix}.pkl"

    def _generate_embeddings_for_entity_type(
        self,
        df: pd.DataFrame,
        entity_type: str,
        name_col: str = 'name'
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a specific entity type using Google AI.

        Args:
            df: DataFrame with entity data
            entity_type: Type of entity (Knowledge, Skill, etc.)
            name_col: Column name containing entity names

        Returns:
            Dictionary mapping entity names to embedding vectors
        """
        method_start_time = log_timed_event(
            f"Processing embeddings for {len(df)} '{entity_type}' entities...")

        # --- MODIFICATION: Simplified cache path logic ---
        cache_path = self._get_cache_path(entity_type)
        self._session_cache_files.add(cache_path)

        if self.use_cache and cache_path.exists():
            log_timed_event(
                f"Loading embeddings from cache: {cache_path.name}", method_start_time)
            self._stats['embedding']['cache_hits'] += 1
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        self._stats['embedding']['cache_misses'] += 1
        log_timed_event(
            f"Generating new embeddings via Google AI for '{entity_type}'...")

        # Prepare texts
        texts = [self._create_entity_text(row[name_col], entity_type)
                 for _, row in df.iterrows()]
        names = df[name_col].tolist()

        # Generate embeddings with retry logic
        all_embeddings = []
        max_retries = 3
        base_delay = 2

        encode_start_time = log_timed_event(
            f"  [API CALL] Sending {len(texts)} texts to {self.embedding_model}...")

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) - 1) // self.batch_size + 1

            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    all_embeddings.extend(result['embedding'])
                    self._stats['embedding']['total_api_calls'] += 1

                    if total_batches > 1:
                        logger.info(
                            f"    Batch {batch_num}/{total_batches} complete")
                        if i + self.batch_size < len(texts):
                            time.sleep(base_delay)

                    break  # Success

                except Exception as e:
                    wait_time = base_delay * (2 ** attempt)

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"    Batch {batch_num} failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        logger.info(f"    Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"    Batch {batch_num} failed after {max_retries} attempts")
                        # Save partial results
                        if all_embeddings:
                            partial_embeddings = {
                                name: np.array(embedding)
                                for name, embedding in zip(names[:len(all_embeddings)], all_embeddings)
                            }
                            # --- MODIFICATION: Use simple partial name ---
                            partial_path = self.cache_dir / \
                                f"inferred_{entity_type}_embeddings_partial.pkl"
                            with open(partial_path, 'wb') as f:
                                pickle.dump(partial_embeddings, f)
                            logger.info(
                                f"    Saved {len(partial_embeddings)} partial embeddings")
                        raise

        log_timed_event(
            f"  [API CALL] Received all {len(all_embeddings)} embeddings", encode_start_time)

        # Create embeddings dictionary
        embeddings = {name: np.array(embedding)
                      for name, embedding in zip(names, all_embeddings)}

        self._stats['embedding']['total_embeddings_generated'] += len(
            embeddings)

        # Save to cache
        if self.use_cache:
            log_timed_event(f"Saving embeddings to cache: {cache_path.name}")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)

        log_timed_event(
            f"Finished processing '{entity_type}' embeddings", method_start_time)

        return embeddings

    def _generate_all_embeddings(
        self,
        entities: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate embeddings for all entity types.

        Args:
            entities: Dictionary of entity DataFrames

        Returns:
            Dictionary of embeddings for each entity type
        """
        logger.info("=" * 70)
        logger.info("GENERATING EMBEDDINGS FOR ALL ENTITY TYPES")
        logger.info("-" * 70)

        embeddings = {}
        entity_configs = [
            ('knowledge', 'Knowledge'),
            ('skills', 'Skill'),
            ('abilities', 'Ability'),
            ('functions', 'Function'),
            ('tasks', 'Task')
        ]
        
        # --- MODIFICATION: Match entity_type keys to file keys ---
        for key, entity_type_text in entity_configs:
            logger.info(f"\nProcessing: {entity_type_text.upper()}")
            logger.info(f"Records to process: {len(entities[key])}")
            logger.info("-" * 70)

            embeddings[key] = self._generate_embeddings_for_entity_type(
                df=entities[key],
                entity_type=entity_type_text # Use 'Knowledge' as text
            )

        logger.info("=" * 70)
        logger.info("EMBEDDING GENERATION COMPLETED")
        logger.info("-" * 70)

        stats = self._stats['embedding']
        logger.info(
            f"Total Embeddings: {stats['total_embeddings_generated']}")
        logger.info(f"API Calls: {stats['total_api_calls']}")
        logger.info(f"Cache Hits: {stats['cache_hits']}")
        logger.info(f"Cache Misses: {stats['cache_misses']}")
        logger.info("=" * 70)

        return embeddings

    # ========================================================================
    # SIMILARITY CALCULATION
    # ========================================================================

    def _calculate_relationship(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_embeddings: Dict[str, np.ndarray],
        target_embeddings: Dict[str, np.ndarray],
        relationship_type: str
    ) -> pd.DataFrame:
        """
        Calculate relationships between source and target entities.

        Args:
            source_df: DataFrame with source entities
            target_df: DataFrame with target entities
            source_embeddings: Embeddings for source entities
            target_embeddings: Embeddings for target entities
            relationship_type: Type of relationship

        Returns:
            DataFrame with inferred relationships
        """
        start_time = time.time()
        method_start_time = log_timed_event(
            f"Calculating '{relationship_type}' ({len(source_df)} x {len(target_df)})..."
        )

        source_names = source_df['name'].tolist()
        target_names = target_df['name'].tolist()

        # Create embedding matrices
        source_matrix = np.array([source_embeddings[name]
                                 for name in source_names])
        target_matrix = np.array([target_embeddings[name]
                                 for name in target_names])

        # Calculate cosine similarity
        sim_start_time = log_timed_event(
            f"  [COMPUTATION] Computing similarity matrix...")
        similarity_matrix = cosine_similarity(source_matrix, target_matrix)
        log_timed_event(
            f"  [COMPUTATION] Similarity matrix complete", sim_start_time)

        # Extract relationships based on thresholds
        threshold = self.similarity_thresholds.get(relationship_type, 0.60)
        max_per_entity = self.max_relationships.get(relationship_type, 20)

        log_timed_event(
            f"  [FILTERING] Applying threshold={threshold}, max={max_per_entity}...")

        relationships = []
        for i, source_name in enumerate(source_names):
            similarities = similarity_matrix[i]
            candidate_indices = np.where(similarities >= threshold)[0]

            if len(candidate_indices) == 0:
                continue

            # Sort by similarity score (descending)
            candidate_similarities = similarities[candidate_indices]
            sorted_indices = np.argsort(-candidate_similarities)

            # Take top N relationships, excluding self-relationships
            count = 0
            for j, sim in zip(candidate_indices[sorted_indices], candidate_similarities[sorted_indices]):
                if count >= max_per_entity:
                    break
                if source_name != target_names[j]:  # Avoid self-relationships
                    relationships.append({
                        'source_name': source_name,
                        'target_name': target_names[j],
                        'confidence_score': float(sim)
                    })
                    count += 1

        # Update statistics
        self._stats['calculation']['relationships_by_type'][relationship_type] = len(
            relationships)
        self._stats['calculation']['total_relationships_inferred'] += len(
            relationships)
        self._stats['calculation']['calculation_time_by_type'][relationship_type] = time.time(
        ) - start_time

        log_timed_event(
            f"Found {len(relationships)} '{relationship_type}' relationships",
            method_start_time
        )

        return pd.DataFrame(relationships)

    def _calculate_all_relationships(
        self,
        entities: Dict[str, pd.DataFrame],
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all relationship types.

        Args:
            entities: Dictionary of entity DataFrames
            embeddings: Dictionary of embeddings

        Returns:
            Dictionary of relationship DataFrames
        """
        logger.info("=" * 70)
        logger.info("CALCULATING RELATIONSHIPS")
        logger.info("-" * 70)

        relationships = {}

        # Relationship configurations: (type, source, target)
        configs = [
            ('knowledge_skill', 'knowledge', 'skills'),
            ('skill_ability', 'skills', 'abilities'),
            ('knowledge_function', 'knowledge', 'functions'),
            ('ability_task', 'abilities', 'tasks'),
            ('function_task', 'functions', 'tasks')
        ]

        for rel_type, src_key, tgt_key in configs:
            relationships[rel_type] = self._calculate_relationship(
                source_df=entities[src_key],
                target_df=entities[tgt_key],
                source_embeddings=embeddings[src_key],
                target_embeddings=embeddings[tgt_key],
                relationship_type=rel_type
            )
            self.metadata['relationship_counts'][rel_type] = len(
                relationships[rel_type])

        logger.info("=" * 70)
        logger.info("RELATIONSHIP CALCULATION COMPLETED")
        logger.info("-" * 70)
        logger.info(
            f"Total Relationships: {self._stats['calculation']['total_relationships_inferred']}")
        for rel_type, count in self._stats['calculation']['relationships_by_type'].items():
            logger.info(f"  {rel_type}: {count}")
        logger.info("=" * 70)

        return relationships

    # ========================================================================
    # DATA I/O
    # ========================================================================

    def _load_mapped_entities(self) -> Dict[str, pd.DataFrame]:
        """
        Load all mapped entity DataFrames from CSV files.

        Returns:
            Dictionary mapping entity types to DataFrames
        """
        logger.info("=" * 70)
        logger.info("LOADING MAPPED ENTITIES")
        logger.info("=" * 70)

        entity_files = {
            'knowledge': 'knowledge_mapped.csv',
            'skills': 'skill_mapped.csv',
            'abilities': 'ability_mapped.csv',
            'functions': 'function_mapped.csv',
            'tasks': 'task_mapped.csv'
        }

        entities = {}

        for entity_type, filename in entity_files.items():
            filepath = self.input_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {filepath}\n"
                    f"Please ensure onet_mapper.py has been run successfully."
                )

            logger.info(f"Loading {entity_type} from {filename}")
            df = pd.read_csv(filepath)
            entities[entity_type] = df

            self.metadata['entity_counts'][entity_type] = len(df)
            logger.info(f"  Loaded {len(df)} {entity_type} entities")

        total = sum(self.metadata['entity_counts'].values())
        logger.info("=" * 70)
        logger.info(f"Total entities loaded: {total}")
        logger.info("=" * 70)

        return entities

    def _save_relationships(self, relationships: Dict[str, pd.DataFrame]):
        """
        Save relationship DataFrames to CSV files.

        Args:
            relationships: Dictionary of relationship DataFrames
        """
        logger.info("=" * 70)
        logger.info("SAVING RELATIONSHIP DATA")
        logger.info("-" * 70)
        logger.info(f"Output directory: {self.output_dir}")

        for rel_type, rel_df in relationships.items():
            filename = f"{rel_type}_inferred.csv"
            filepath = self.output_dir / filename

            # Sort by confidence score
            if not rel_df.empty:
                rel_df = rel_df.sort_values(
                    'confidence_score', ascending=False)

            rel_df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(rel_df)} {rel_type} to {filename}")

        logger.info("=" * 70)

    # ========================================================================
    # REPORTING AND METADATA
    # ========================================================================

    def _generate_relationship_summary(
        self,
        relationships: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate summary of relationships."""
        lines = [
            "=" * 70,
            "RELATIONSHIP SUMMARY",
            "=" * 70,
        ]

        total = 0
        for rel_type, rel_df in relationships.items():
            count = len(rel_df)
            total += count

            if count > 0:
                min_score = rel_df['confidence_score'].min()
                max_score = rel_df['confidence_score'].max()
                mean_score = rel_df['confidence_score'].mean()

                lines.append(f"\n{rel_type}:")
                lines.append(f"  Count: {count:,}")
                lines.append(
                    f"  Threshold: {self.similarity_thresholds.get(rel_type)}")
                lines.append(
                    f"  Max per entity: {self.max_relationships.get(rel_type)}")
                lines.append(
                    f"  Confidence: {min_score:.3f} - {max_score:.3f} (avg: {mean_score:.3f})")

                calc_time = self._stats['calculation']['calculation_time_by_type'].get(
                    rel_type)
                if calc_time:
                    lines.append(f"  Calculation time: {calc_time:.2f}s")

        lines.extend([
            "",
            "-" * 70,
            f"TOTAL RELATIONSHIPS: {total:,}",
            "=" * 70
        ])

        return "\n".join(lines)

    def _generate_pipeline_report(
        self,
        relationships: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate comprehensive pipeline execution report."""
        lines = [
            "=" * 70,
            "O*NET RELATIONSHIP BUILDER - PIPELINE REPORT",
            "=" * 70,
            f"Pipeline: {self.metadata['pipeline_name']}",
            f"Embedding Model: {self.metadata['embedding_model']}",
            f"Start Time: {self.metadata['start_time']}",
            f"End Time: {self.metadata['end_time']}",
            f"Duration: {self.metadata.get('duration', 'N/A')}",
            f"Status: {self.metadata['status']}",
            "",
            "=" * 70,
            "ENTITY SUMMARY",
            "=" * 70,
        ]

        total_entities = 0
        for entity_type, count in self.metadata['entity_counts'].items():
            lines.append(f"  {entity_type:12s}: {count:5d} entities")
            total_entities += count

        lines.extend([
            "",
            f"  Total Entities: {total_entities}",
            "",
            "=" * 70,
            "EMBEDDING STATISTICS",
            "=" * 70,
        ])

        stats = self._stats['embedding']
        lines.extend([
            f"  Total Embeddings: {stats['total_embeddings_generated']}",
            f"  API Calls: {stats['total_api_calls']}",
            f"  Cache Hits: {stats['cache_hits']}",
            f"  Cache Misses: {stats['cache_misses']}",
        ])

        lines.append("")
        lines.append(self._generate_relationship_summary(relationships))

        lines.extend([
            "",
            "=" * 70,
            "OUTPUT FILES",
            "=" * 70,
        ])

        for rel_type in relationships.keys():
            filename = f"{rel_type}_inferred.csv"
            lines.append(f"  {self.output_dir / filename}")

        lines.extend([
            "",
            "=" * 70,
            "NEXT STEPS",
            "=" * 70,
            "1. Review relationship files in output directory",
            "2. Validate relationship quality and confidence scores",
            "3. Run onet_loader.py to load into database",
            "4. Execute onet_validator.py to verify data integrity",
            "",
            "=" * 70
        ])

        return "\n".join(lines)

    def _save_reports(self, report: str):
        """Save pipeline report to file (only latest version)."""
        # Save only latest report
        latest_file = self.reports_dir / "latest_pipeline_report.txt"
        with open(latest_file, 'w') as f:
            f.write(report)

        logger.info(f"Pipeline report saved to {latest_file}")

    def _save_metadata(self):
        """Save pipeline metadata - no separate JSON files, included in report."""
        # Don't save separate metadata files - everything is in the text report
        logger.info("Metadata included in pipeline report")

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_intermediate_files(self, keep_embeddings: bool = False):
        """
        Clean up intermediate files after pipeline completion.

        Args:
            keep_embeddings: If True, keeps embedding cache files
        """
        logger.info("Cleaning up intermediate files...")

        if not keep_embeddings and self.cache_dir.exists():
            # Remove old cache files not in current session
            all_cache_files = set(self.cache_dir.glob("*.pkl"))
            files_to_remove = all_cache_files - self._session_cache_files

            removed_count = 0
            removed_size = 0

            for cache_file in files_to_remove:
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += file_size
                except Exception as e:
                    logger.warning(f"Failed to remove {cache_file}: {e}")

            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} old cache files ({removed_size / 1024 / 1024:.2f} MB)")
            else:
                logger.info("No old cache files to remove")

        logger.info("Cleanup completed")

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def validate_inputs(self) -> bool:
        """
        Validate that all required input files exist.

        Returns:
            True if all inputs valid, False otherwise
        """
        logger.info("Validating input files...")

        required_files = [
            'knowledge_mapped.csv',
            'skill_mapped.csv',
            'ability_mapped.csv',
            'function_mapped.csv',
            'task_mapped.csv'
        ]

        all_valid = True
        for filename in required_files:
            filepath = self.input_dir / filename
            if filepath.exists():
                logger.info(f"  ✓ {filename}")
            else:
                logger.error(f"  ✗ {filename} NOT FOUND")
                all_valid = False

        return all_valid

    def build_relationships(self, cleanup_after: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete relationship building pipeline.

        Args:
            cleanup_after: Whether to clean up intermediate files

        Returns:
            Dictionary of relationship DataFrames
        """
        try:
            # Update metadata
            self.metadata['start_time'] = datetime.now().isoformat()
            self.metadata['status'] = 'running'

            logger.info("\n" + "=" * 70)
            logger.info("STARTING RELATIONSHIP BUILDING PIPELINE")
            logger.info("=" * 70)
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Model: {self.embedding_model}")
            logger.info(f"Cache: {self.use_cache}")
            logger.info("=" * 70)

            # Step 1: Load mapped entities
            entities = self._load_mapped_entities()

            # Step 2: Generate embeddings
            embeddings = self._generate_all_embeddings(entities)

            # Step 3: Calculate relationships
            relationships = self._calculate_all_relationships(
                entities, embeddings)

            # Step 4: Save relationships
            self._save_relationships(relationships)

            # Step 5: Generate and save reports
            report = self._generate_pipeline_report(relationships)
            self._save_reports(report)

            # Print report
            print("\n" + report)

            # Update metadata
            self.metadata['end_time'] = datetime.now().isoformat()
            start = datetime.fromisoformat(self.metadata['start_time'])
            end = datetime.fromisoformat(self.metadata['end_time'])
            duration = end - start
            self.metadata['duration'] = str(duration)
            self.metadata['status'] = 'completed'

            # Save metadata
            self._save_metadata()

            # Cleanup if requested
            if cleanup_after:
                self.cleanup_intermediate_files()

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Duration: {duration}")
            logger.info("=" * 70)

            return relationships

        except Exception as e:
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['end_time'] = datetime.now().isoformat()

            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            self._save_metadata()

            raise

    def get_statistics(self) -> Dict:
        """Get current pipeline statistics."""
        return {
            'metadata': self.metadata,
            'embedding_stats': self._stats['embedding'],
            'calculation_stats': self._stats['calculation']
        }


def main():
    """Main execution function for standalone usage."""
    logger.info("=" * 70)
    logger.info("O*NET RELATIONSHIP BUILDER")
    logger.info("=" * 70)

    # Initialize builder
    builder = ONetRelationshipBuilder(
        embedding_model="models/gemini-embedding-001",
        use_cache=True
    )

    # Validate inputs
    if not builder.validate_inputs():
        logger.error("Input validation failed. Run onet_mapper.py first.")
        return

    # Build relationships
    try:
        # --- MODIFICATION: Set cleanup_after=False to preserve cache ---
        relationships = builder.build_relationships(cleanup_after=False)
        total = sum(len(df) for df in relationships.values())
        logger.info(f"\n✓ Successfully generated {total:,} relationships!")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()