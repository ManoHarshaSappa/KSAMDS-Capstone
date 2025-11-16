"""
Skills Framework Relationship Inference for KSAMDS Project

This module orchestrates the complete embedding-based relationship inference pipeline:
1. Generates embeddings for Knowledge, Skills, Abilities, Functions, and Tasks
2. Calculates cosine similarity between entity embeddings
3. Infers relationships: ability_task, function_task, knowledge_function
4. Converts existing relationships (knowledge_skill, skill_ability) to use UUIDs
5. Saves relationship files ready for database loading

Uses Google AI API (models/gemini-embedding-001) for embedding generation
with caching, batch processing, and comprehensive error handling.
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

import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

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


class SkillsFrameworkRelationshipBuilder:
    """
    Orchestrates the complete embedding-based relationship inference pipeline.

    Handles:
    - Embedding generation with caching
    - Similarity calculation between entities
    - Relationship inference with configurable thresholds
    - Converting existing relationships to use UUIDs
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
        # Get project root directory (3 levels up from skillsframework_etl folder)
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Setup directory structure
        self.input_dir = self.project_root / "data" / \
            "skillsframework" / "archive" / "mapped"
        self.output_dir = self.project_root / "data" / \
            "skillsframework" / "archive" / "relationships"
        self.cache_dir = self.project_root / "data" / \
            "skillsframework" / "archive" / "embeddings"
        self.reports_dir = self.project_root / "data" / "skillsframework" / "reports"

        # Create directories
        for directory in [self.input_dir, self.output_dir, self.cache_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Default similarity thresholds (only for new relationships)
        self.similarity_thresholds = similarity_thresholds or {
            'ability_task': 0.65,
            'function_task': 0.75,
            'knowledge_function': 0.60
        }

        # Default max relationships per entity
        self.max_relationships = max_relationships or {
            'ability_task': 25,
            'function_task': 10,
            'knowledge_function': 20
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
            },
            'conversion': {
                'knowledge_skill_converted': 0,
                'skill_ability_converted': 0
            }
        }

        # Pipeline metadata
        self.metadata = {
            'pipeline_name': 'SkillsFrameworkRelationshipBuilder',
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
        logger.info("SKILLS FRAMEWORK RELATIONSHIP BUILDER INITIALIZED")
        logger.info("-" * 70)
        logger.info(f"Embedding Model: {self.embedding_model}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")
        logger.info(f"Input Directory: {self.input_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        logger.info("-" * 70)
        logger.info("Similarity Thresholds (New Relationships):")
        for rel_type, threshold in self.similarity_thresholds.items():
            logger.info(f"  {rel_type}: {threshold}")
        logger.info("-" * 70)
        logger.info("Max Relationships (New Relationships):")
        for rel_type, max_val in self.max_relationships.items():
            logger.info(f"  {rel_type}: {max_val}")
        logger.info("=" * 70)

    def _configure_google_ai(self):
        """Configure Google AI API with proper credentials."""
        try:
            if 'GOOGLE_API_KEY' in os.environ:
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                logger.info("Configured Google AI with environment variable")
            else:
                # Try Google Colab userdata
                from google.colab import userdata
                api_key = userdata.get('GOOGLE_API_KEY')
                genai.configure(api_key=api_key)
                logger.info("Configured Google AI with Colab userdata")
        except Exception as e:
            logger.error(
                "Could not configure Google AI. Please set GOOGLE_API_KEY environment variable.")
            raise e

    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================

    def _create_entity_text(self, name: str, entity_type: str) -> str:
        """Create rich text representation for embedding generation."""
        return f"{entity_type}: {name}"

    def _generate_cache_key(self, df: pd.DataFrame, entity_type: str, name_col: str = 'name') -> str:
        """Generate unique cache key based on entity names and model.

        Only uses the sorted list of entity names for cache key generation,
        since that's what determines the embeddings (IDs and other columns don't affect embeddings).
        """
        # Sort names to ensure consistent ordering
        names = sorted(df[name_col].tolist())
        names_str = '|'.join(names)
        content = f"{entity_type}_{self.embedding_model}_{names_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, entity_type: str, cache_key: str) -> Path:
        """Get full path for a cache file."""
        return self.cache_dir / f"{entity_type}_{cache_key}.pkl"

    def _generate_embeddings_for_entity_type(
        self,
        df: pd.DataFrame,
        entity_type: str,
        name_col: str = 'name'
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for a specific entity type using Google AI."""
        method_start_time = log_timed_event(
            f"Processing embeddings for {len(df)} '{entity_type}' entities...")

        # Check cache
        cache_key = self._generate_cache_key(df, entity_type, name_col)
        cache_path = self._get_cache_path(entity_type, cache_key)
        self._session_cache_files.add(cache_path)

        if self.use_cache and cache_path.exists():
            log_timed_event(
                f"Found cache file. Loading embeddings from '{cache_path}'.", method_start_time)
            self._stats['embedding']['cache_hits'] += 1
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        self._stats['embedding']['cache_misses'] += 1
        log_timed_event(
            f"No cache found. Generating new embeddings via Google AI for '{entity_type}'...")

        texts, names = [], []
        for _, row in df.iterrows():
            texts.append(self._create_entity_text(row[name_col], entity_type))
            names.append(row[name_col])

        encode_start_time = log_timed_event(
            f"  [API CALL] Sending {len(texts)} texts to Google '{self.embedding_model}'...")
        all_embeddings = []

        max_retries = 3
        base_delay = 2

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1

            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    all_embeddings.extend(result['embedding'])
                    self._stats['embedding']['total_api_calls'] += 1

                    if len(texts) > self.batch_size:
                        log_timed_event(
                            f"    Batch {batch_num}/{(len(texts)-1)//self.batch_size + 1} complete. Waiting {base_delay} seconds...")
                        time.sleep(base_delay)

                    break

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
                        raise

        log_timed_event(
            f"  [API CALL] Received all {len(all_embeddings)} embeddings from Google.", encode_start_time)

        embeddings = {name: np.array(embedding)
                      for name, embedding in zip(names, all_embeddings)}
        self._stats['embedding']['total_embeddings_generated'] += len(
            embeddings)

        if self.use_cache:
            log_timed_event(
                f"Saving newly generated embeddings to cache file: '{cache_path}'")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)

        log_timed_event(
            f"Finished processing embeddings for '{entity_type}'.", method_start_time)
        return embeddings

    def _load_mapped_entities(self) -> Dict[str, pd.DataFrame]:
        """Load mapped entity files."""
        logger.info("=" * 70)
        logger.info("LOADING MAPPED ENTITIES")
        logger.info("=" * 70)

        entities = {}
        entity_files = {
            'knowledge': 'sf_knowledge.csv',
            'skill': 'sf_skill.csv',
            'ability': 'sf_ability.csv',
            'function': 'sf_function.csv',
            'task': 'sf_task.csv'
        }

        for entity_type, filename in entity_files.items():
            filepath = self.input_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                entities[entity_type] = df
                self.metadata['entity_counts'][entity_type] = len(df)
                logger.info(
                    f"Loaded {entity_type}: {len(df)} entities from {filename}")
            else:
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Missing required file: {filepath}")

        return entities

    def _generate_all_embeddings(self, entities: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate embeddings for all entity types."""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING EMBEDDINGS FOR ALL ENTITY TYPES")
        logger.info("=" * 70)

        all_embeddings = {}

        for entity_type, df in entities.items():
            logger.info("-" * 70)
            logger.info(f"PROCESSING: {entity_type.upper()}")
            logger.info(f"Records to process: {len(df)}")
            logger.info("-" * 70)

            all_embeddings[entity_type] = self._generate_embeddings_for_entity_type(
                df=df,
                entity_type=entity_type.capitalize(),
                name_col='name'
            )

        logger.info("=" * 70)
        logger.info("EMBEDDING GENERATION COMPLETED")
        logger.info("=" * 70)

        return all_embeddings

    # ========================================================================
    # SIMILARITY CALCULATION
    # ========================================================================

    def _calculate_similarity(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_embeddings: Dict[str, np.ndarray],
        target_embeddings: Dict[str, np.ndarray],
        relationship_type: str
    ) -> pd.DataFrame:
        """Calculate relationships between source and target entities."""
        start_time = time.time()
        method_start_time = log_timed_event(
            f"Calculating '{relationship_type}' relationships ({len(source_df)} x {len(target_df)})..."
        )

        source_names = source_df['name'].tolist()
        target_names = target_df['name'].tolist()
        source_ids = source_df['id'].tolist()
        target_ids = target_df['id'].tolist()

        # Create embedding matrices
        source_embedding_matrix = np.array(
            [source_embeddings.get(name) for name in source_names])
        target_embedding_matrix = np.array(
            [target_embeddings.get(name) for name in target_names])

        # Calculate similarities
        sim_start_time = log_timed_event(
            f"  [COMPUTATION] Calculating cosine similarity matrix for '{relationship_type}'..."
        )
        similarity_matrix = cosine_similarity(
            source_embedding_matrix, target_embedding_matrix)
        log_timed_event(
            f"  [COMPUTATION] Similarity matrix calculation finished.", sim_start_time)

        # Extract relationships
        relationships = []
        threshold = self.similarity_thresholds.get(relationship_type, 0.60)
        max_per_entity = self.max_relationships.get(relationship_type, 20)

        log_timed_event(
            f"  [FILTERING] Applying threshold {threshold} and max {max_per_entity} relationships per entity...")

        for i, (source_name, source_id) in enumerate(zip(source_names, source_ids)):
            similarities = similarity_matrix[i]
            candidate_indices = np.where(similarities >= threshold)[0]

            if not len(candidate_indices):
                continue

            candidate_similarities = similarities[candidate_indices]
            sorted_indices = np.argsort(-candidate_similarities)

            count = 0
            for j, sim in zip(candidate_indices[sorted_indices], candidate_similarities[sorted_indices]):
                if count >= max_per_entity:
                    break
                if source_name != target_names[j]:  # No self-relationships
                    relationships.append({
                        'source_id': source_id,
                        'target_id': target_ids[j],
                        'confidence_score': float(sim)
                    })
                    count += 1

        self._stats['calculation']['relationships_by_type'][relationship_type] = len(
            relationships)
        self._stats['calculation']['total_relationships_inferred'] += len(
            relationships)
        self._stats['calculation']['calculation_time_by_type'][relationship_type] = time.time(
        ) - start_time

        log_timed_event(
            f"Finished calculating '{relationship_type}'. Found {len(relationships)} relationships.",
            method_start_time
        )
        return pd.DataFrame(relationships)

    def _calculate_all_relationships(
        self,
        entities: Dict[str, pd.DataFrame],
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate all new relationship types."""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING NEW RELATIONSHIPS")
        logger.info("=" * 70)

        all_relationships = {}

        # New relationships to infer
        configs = [
            ('ability_task', 'ability', 'task'),
            ('function_task', 'function', 'task'),
            ('knowledge_function', 'knowledge', 'function')
        ]

        for rel_type, src_key, tgt_key in configs:
            all_relationships[rel_type] = self._calculate_similarity(
                source_df=entities[src_key],
                target_df=entities[tgt_key],
                source_embeddings=embeddings[src_key],
                target_embeddings=embeddings[tgt_key],
                relationship_type=rel_type
            )

        logger.info("=" * 70)
        logger.info("NEW RELATIONSHIP CALCULATION COMPLETED")
        logger.info("=" * 70)

        return all_relationships

    # ========================================================================
    # LOAD EXISTING RELATIONSHIPS
    # ========================================================================

    def _load_existing_relationships(self) -> Dict[str, pd.DataFrame]:
        """Load existing knowledge_skill and skill_ability relationships that were already converted to UUIDs."""
        logger.info("\n" + "=" * 70)
        logger.info("LOADING EXISTING RELATIONSHIPS")
        logger.info("=" * 70)

        existing_relationships = {}

        # Load knowledge_skill
        ks_path = self.output_dir / "knowledge_skill.csv"
        if ks_path.exists():
            logger.info(
                f"Loading knowledge_skill relationships from {ks_path}...")
            ks_df = pd.read_csv(ks_path)
            existing_relationships['knowledge_skill'] = ks_df
            self._stats['conversion']['knowledge_skill_converted'] = len(ks_df)
            logger.info(
                f"Loaded {len(ks_df)} knowledge_skill relationships")
        else:
            logger.warning(f"knowledge_skill.csv not found at {ks_path}")

        # Load skill_ability
        sa_path = self.output_dir / "skill_ability.csv"
        if sa_path.exists():
            logger.info(
                f"Loading skill_ability relationships from {sa_path}...")
            sa_df = pd.read_csv(sa_path)
            existing_relationships['skill_ability'] = sa_df
            self._stats['conversion']['skill_ability_converted'] = len(sa_df)
            logger.info(
                f"Loaded {len(sa_df)} skill_ability relationships")
        else:
            logger.warning(f"skill_ability.csv not found at {sa_path}")

        logger.info("=" * 70)
        logger.info("EXISTING RELATIONSHIPS LOADED")
        logger.info("=" * 70)

        return existing_relationships

    # ========================================================================
    # SAVE RELATIONSHIPS
    # ========================================================================

    def _save_relationships(self, relationships: Dict[str, pd.DataFrame]):
        """Save all relationships to CSV files."""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING RELATIONSHIP DATA")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")

        for rel_type, rel_df in relationships.items():
            if rel_df.empty:
                logger.warning(f"No relationships to save for {rel_type}")
                continue

            filename = f"{rel_type}_inferred.csv"
            filepath = self.output_dir / filename

            # Sort by confidence score
            rel_df = rel_df.sort_values('confidence_score', ascending=False)
            rel_df.to_csv(filepath, index=False)

            self.metadata['relationship_counts'][rel_type] = len(rel_df)
            logger.info(
                f"Saved {len(rel_df)} {rel_type} relationships to {filepath}")

        logger.info("=" * 70)

    # ========================================================================
    # REPORTING
    # ========================================================================

    def _generate_relationship_summary(self, relationships: Dict[str, pd.DataFrame]) -> str:
        """Generate summary of relationships."""
        lines = [
            "RELATIONSHIP COUNTS:",
            "-" * 70
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
                    f"  Confidence range: {min_score:.3f} - {max_score:.3f}")
                lines.append(f"  Mean confidence: {mean_score:.3f}")

                calc_time = self._stats['calculation']['calculation_time_by_type'].get(
                    rel_type)
                if calc_time:
                    lines.append(
                        f"  Calculation time: {calc_time:.2f} seconds")

        lines.extend([
            "",
            "-" * 70,
            f"TOTAL RELATIONSHIPS: {total:,}",
        ])

        return "\n".join(lines)

    def _generate_pipeline_report(self, relationships: Dict[str, pd.DataFrame]) -> str:
        """Generate comprehensive pipeline report."""
        stats = self._stats['embedding']

        lines = [
            "=" * 70,
            "SKILLS FRAMEWORK RELATIONSHIP BUILDING PIPELINE REPORT",
            "=" * 70,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.embedding_model}",
            f"Batch Size: {self.batch_size}",
            "",
            "=" * 70,
            "EMBEDDING STATISTICS",
            "=" * 70,
            f"  Total Embeddings Generated: {stats['total_embeddings_generated']}",
            f"  Total API Calls: {stats['total_api_calls']}",
            f"  Cache Hits: {stats['cache_hits']}",
            f"  Cache Misses: {stats['cache_misses']}",
            "",
            "=" * 70,
            "EXISTING RELATIONSHIP STATISTICS",
            "=" * 70,
            f"  knowledge_skill loaded: {self._stats['conversion']['knowledge_skill_converted']}",
            f"  skill_ability loaded: {self._stats['conversion']['skill_ability_converted']}",
        ]

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
            "3. Run sf_loader.py to load into database",
            "4. Execute sf_validator.py to verify data integrity",
            "",
            "=" * 70
        ])

        return "\n".join(lines)

    def _save_reports(self, report: str):
        """Save pipeline report to file."""
        latest_file = self.reports_dir / "latest_sf_pipeline_report.txt"
        with open(latest_file, 'w') as f:
            f.write(report)
        logger.info(f"Pipeline report saved to {latest_file}")

    def _save_metadata(self):
        """Save pipeline metadata."""
        logger.info("Metadata included in pipeline report")

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def validate_inputs(self) -> bool:
        """Validate that all required input files exist."""
        logger.info("Validating input files...")

        required_files = [
            'sf_knowledge.csv',
            'sf_skill.csv',
            'sf_ability.csv',
            'sf_function.csv',
            'sf_task.csv'
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
        """Execute the complete relationship building pipeline."""
        try:
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

            # Step 3: Calculate NEW relationships
            new_relationships = self._calculate_all_relationships(
                entities, embeddings)

            # Step 4: Load existing relationships (already converted to UUIDs by mapper)
            # These are loaded for statistics/reporting only, not saved again
            existing_relationships = self._load_existing_relationships()

            # Step 5: Save only NEW relationships
            self._save_relationships(new_relationships)

            # Step 6: Combine all relationships for reporting
            all_relationships = {**new_relationships, **existing_relationships}

            # Step 7: Generate and save reports
            report = self._generate_pipeline_report(all_relationships)
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

            self._save_metadata()

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Duration: {duration}")
            logger.info("=" * 70)

            return all_relationships

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
            'calculation_stats': self._stats['calculation'],
            'conversion_stats': self._stats['conversion']
        }


def main():
    """Main execution function for standalone usage."""
    logger.info("=" * 70)
    logger.info("SKILLS FRAMEWORK RELATIONSHIP BUILDER")
    logger.info("=" * 70)

    # Initialize builder
    builder = SkillsFrameworkRelationshipBuilder(
        embedding_model="models/gemini-embedding-001",
        use_cache=True
    )

    # Validate inputs
    if not builder.validate_inputs():
        logger.error(
            "Input validation failed. Run skillsframework_mapper.py first.")
        return

    # Build relationships
    try:
        relationships = builder.build_relationships(cleanup_after=False)
        total = sum(len(df) for df in relationships.values())
        logger.info(f"\n✓ Successfully generated {total:,} relationships!")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
