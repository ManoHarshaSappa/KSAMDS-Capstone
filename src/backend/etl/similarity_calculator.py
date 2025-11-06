"""
Similarity Calculator for KSAMDS Project
Calculates cosine similarity between entity embeddings and infers relationships
Supports multiple relationship types with configurable thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Helper function for timed logging ---


def log_timed_event(message, start_time=None):
    """Logs a message with timestamp and optional duration using logger."""
    if start_time:
        duration = time.time() - start_time
        logger.info(f"✅ {message} (Duration: {duration:.2f} seconds)")
    else:
        logger.info(f"⏳ {message}")
        return time.time()


class SimilarityCalculator:
    """
    Calculates similarity between entity embeddings and infers relationships.
    Supports configurable thresholds for different relationship types.
    """

    def __init__(
        self,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        max_relationships: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the similarity calculator.

        Args:
            similarity_thresholds: Minimum similarity scores for each relationship type
            max_relationships: Maximum relationships per entity for each type
        """
        self.similarity_thresholds = similarity_thresholds or {
            'knowledge_skill': 0.70,
            'skill_ability': 0.65,
            'knowledge_function': 0.60,
            'ability_task': 0.65,
            'function_task': 0.75
        }
        self.max_relationships = max_relationships or {
            'knowledge_skill': 15,
            'skill_ability': 10,
            'knowledge_function': 20,
            'ability_task': 25,
            'function_task': 10
        }

        # Track statistics
        self._calculation_stats = {
            'total_relationships_inferred': 0,
            'relationships_by_type': {},
            'calculation_time_by_type': {}
        }

        logger.info("=" * 70)
        logger.info("INITIALIZING SIMILARITY CALCULATOR")
        logger.info("-" * 70)
        logger.info("Configuration:")
        logger.info(f"Similarity thresholds: {self.similarity_thresholds}")
        logger.info(f"Max relationships: {self.max_relationships}")
        logger.info("=" * 70)

    def calculate_relationships(
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
            relationship_type: Type of relationship being calculated

        Returns:
            DataFrame with inferred relationships
        """
        start_time = time.time()
        method_start_time = log_timed_event(
            f"Calculating '{relationship_type}' relationships ({len(source_df)} x {len(target_df)})..."
        )

        source_names = source_df['name'].tolist()
        target_names = target_df['name'].tolist()

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

        # Extract relationships based on thresholds
        relationships = []
        threshold = self.similarity_thresholds.get(relationship_type, 0.60)
        max_per_entity = self.max_relationships.get(relationship_type, 20)

        log_timed_event(
            f"  [FILTERING] Applying threshold {threshold} and max {max_per_entity} relationships per entity...")

        for i, source_name in enumerate(source_names):
            similarities = similarity_matrix[i]
            candidate_indices = np.where(similarities >= threshold)[0]

            if not len(candidate_indices):
                continue

            # Sort by similarity score
            candidate_similarities = similarities[candidate_indices]
            sorted_indices = np.argsort(-candidate_similarities)

            count = 0
            for j, sim in zip(candidate_indices[sorted_indices], candidate_similarities[sorted_indices]):
                if count >= max_per_entity:
                    break
                # Business rule: Don't create self-relationships
                if source_name != target_names[j]:
                    relationships.append({
                        'source_name': source_name,
                        'target_name': target_names[j],
                        'confidence_score': float(sim)
                    })
                    count += 1

        # Update statistics
        self._calculation_stats['relationships_by_type'][relationship_type] = len(
            relationships)
        self._calculation_stats['total_relationships_inferred'] += len(
            relationships)
        self._calculation_stats['calculation_time_by_type'][relationship_type] = time.time(
        ) - start_time

        log_timed_event(
            f"Finished calculating '{relationship_type}'. Found {len(relationships)} relationships.",
            method_start_time
        )
        return pd.DataFrame(relationships)

    def calculate_all_relationships(
        self,
        knowledge_df: pd.DataFrame,
        skills_df: pd.DataFrame,
        abilities_df: pd.DataFrame,
        functions_df: pd.DataFrame,
        tasks_df: pd.DataFrame,
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all relationship types. This signature is compatible with the
        EmbeddingRelationshipBuilder.

        Args:
            knowledge_df: Knowledge entities DataFrame
            skills_df: Skills entities DataFrame
            abilities_df: Abilities entities DataFrame
            functions_df: Functions entities DataFrame
            tasks_df: Tasks entities DataFrame
            embeddings: Dictionary of embeddings for each entity type

        Returns:
            Dictionary of relationship DataFrames
        """
        logger.info("=" * 70)
        logger.info("CALCULATING RELATIONSHIPS FOR ALL ENTITY TYPES")
        logger.info("-" * 70)
        logger.info(f"Knowledge Entities: {len(knowledge_df)}")
        logger.info(f"Skill Entities: {len(skills_df)}")
        logger.info(f"Ability Entities: {len(abilities_df)}")
        logger.info(f"Function Entities: {len(functions_df)}")
        logger.info(f"Task Entities: {len(tasks_df)}")
        logger.info("=" * 70)

        all_relationships = {}

        # Create a dictionary of dataframes for easier access
        entities = {
            'knowledge': knowledge_df,
            'skills': skills_df,
            'abilities': abilities_df,
            'functions': functions_df,
            'tasks': tasks_df
        }

        # Relationship configurations: (relationship_type, source_key, target_key)
        configs = [
            ('knowledge_skill', 'knowledge', 'skills'),
            ('skill_ability', 'skills', 'abilities'),
            ('knowledge_function', 'knowledge', 'functions'),
            ('ability_task', 'abilities', 'tasks'),
            ('function_task', 'functions', 'tasks')
        ]

        for rel_type, src_key, tgt_key in configs:
            all_relationships[rel_type] = self.calculate_relationships(
                source_df=entities[src_key],
                target_df=entities[tgt_key],
                source_embeddings=embeddings[src_key],
                target_embeddings=embeddings[tgt_key],
                relationship_type=rel_type
            )

        logger.info("=" * 70)
        logger.info("CALCULATION SUMMARY")
        logger.info("-" * 70)
        logger.info(
            f"Total Relationships Inferred: {self._calculation_stats['total_relationships_inferred']}")
        for rel_type, count in self._calculation_stats['relationships_by_type'].items():
            logger.info(f"{rel_type}: {count}")
        logger.info("=" * 70)

        return all_relationships

    def save_relationships(
        self,
        relationships: Dict[str, pd.DataFrame],
    ):
        """
        Save calculated relationships to CSV files.

        Args:
            relationships: Dictionary of relationship DataFrames
        """
        # Get project root directory (3 levels up from etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        output_path = project_root / "data/archive/relationships"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("SAVING RELATIONSHIP DATA")
        logger.info("-" * 70)
        logger.info(f"Output directory: {output_path}")

        for rel_type, rel_df in relationships.items():
            filename = f"{rel_type}_inferred.csv"
            filepath = output_path / filename

            # Sort by confidence score before saving
            if not rel_df.empty:
                rel_df = rel_df.sort_values(
                    'confidence_score', ascending=False)

            rel_df.to_csv(filepath, index=False)
            logger.info(
                f"Saved {len(rel_df)} {rel_type} relationships to {filepath}")

        # Save metadata about the calculation
        self._save_calculation_metadata(output_path)

    def _save_calculation_metadata(self, output_dir: Path):
        """Save metadata about the similarity calculation process."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'thresholds': self.similarity_thresholds,
            'max_relationships': self.max_relationships,
            'statistics': self._calculation_stats
        }

        metadata_path = output_dir / "similarity_calculation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("-" * 70)
        logger.info("SAVING CALCULATION METADATA")
        logger.info(f"Metadata file: {metadata_path}")
        logger.info("=" * 70)

    def get_calculation_stats(self) -> Dict[str, any]:
        """
        Get statistics about the similarity calculation.

        Returns:
            Dictionary with calculation statistics
        """
        return self._calculation_stats.copy()

    def generate_relationship_summary(self, relationships: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a summary report of the relationships.

        Args:
            relationships: Dictionary of relationship DataFrames

        Returns:
            Summary report as string
        """
        lines = [
            "="*70,
            "RELATIONSHIP INFERENCE SUMMARY",
            "="*70,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RELATIONSHIP COUNTS:",
            "-"*70
        ]

        total_relationships = 0
        for rel_type, rel_df in relationships.items():
            count = len(rel_df)
            total_relationships += count

            if count > 0:
                min_score = rel_df['confidence_score'].min()
                max_score = rel_df['confidence_score'].max()
                mean_score = rel_df['confidence_score'].mean()

                lines.append(f"\n{rel_type}:")
                lines.append(f"  Count: {count:,}")
                lines.append(
                    f"  Threshold: {self.similarity_thresholds.get(rel_type, 'N/A')}")
                lines.append(
                    f"  Max per entity: {self.max_relationships.get(rel_type, 'N/A')}")
                lines.append(
                    f"  Confidence range: {min_score:.3f} - {max_score:.3f}")
                lines.append(f"  Mean confidence: {mean_score:.3f}")

                # Calculate time if available
                calc_time = self._calculation_stats['calculation_time_by_type'].get(
                    rel_type)
                if calc_time:
                    lines.append(
                        f"  Calculation time: {calc_time:.2f} seconds")

        lines.extend([
            "",
            "-"*70,
            f"TOTAL RELATIONSHIPS: {total_relationships:,}",
            "="*70
        ])

        return "\n".join(lines)
