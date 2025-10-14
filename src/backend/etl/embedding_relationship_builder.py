"""
Embedding Relationship Builder for KSAMDS Project
Orchestrates the embedding generation and relationship inference process
Bridges the gap between mapped O*NET data and database loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime
import json

from embedding_generator import EmbeddingGenerator
from similarity_calculator import SimilarityCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingRelationshipBuilder:
    """
    Orchestrates the complete embedding-based relationship inference pipeline.
    Coordinates EmbeddingGenerator and SimilarityCalculator to produce
    relationship files ready for database loading.
    """

    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/processed",
        embedding_cache_dir: str = "data/embeddings",
        embedding_model: str = "models/embedding-001",
        similarity_thresholds: Optional[Dict[str, float]] = None,
        max_relationships: Optional[Dict[str, int]] = None,
        use_cache: bool = True
    ):
        """
        Initialize the relationship builder.

        Args:
            input_dir: Directory containing mapped entity CSV files
            output_dir: Directory to save relationship CSV files
            embedding_cache_dir: Directory for embedding cache
            embedding_model: Sentence-BERT model name
            similarity_thresholds: Custom thresholds for each relationship type
            max_relationships: Custom max relationships per entity
            use_cache: Whether to use cached embeddings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_model = embedding_model
        self.use_cache = use_cache

        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing EmbeddingGenerator...")
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            cache_dir=str(embedding_cache_dir),
            batch_size=32
        )

        logger.info("Initializing SimilarityCalculator...")
        self.similarity_calculator = SimilarityCalculator(
            similarity_thresholds=similarity_thresholds,
            max_relationships=max_relationships
        )

        # Track pipeline metadata
        self.metadata = {
            'pipeline_name': 'EmbeddingRelationshipBuilder',
            'embedding_model': embedding_model,
            'start_time': None,
            'end_time': None,
            'entity_counts': {},
            'relationship_counts': {},
            'status': 'initialized'
        }

        logger.info("EmbeddingRelationshipBuilder initialized successfully")

    def _load_mapped_entities(self) -> Dict[str, pd.DataFrame]:
        """
        Load all mapped entity DataFrames from CSV files.

        Returns:
            Dictionary mapping entity types to DataFrames
        """
        logger.info("="*70)
        logger.info("LOADING MAPPED ENTITIES")
        logger.info("="*70)

        entity_files = {
            'knowledge': 'knowledge_mapped.csv',
            'skills': 'skill_mapped.csv',
            'abilities': 'ability_mapped.csv',
            'functions': 'function_mapped.csv',
            'tasks': 'task_mapped.csv'
        }

        entities = {}

        for entity_type, filename in entity_files.items():
            filepath = self.input_dir / "ksamds" / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {filepath}\n"
                    f"Please ensure onet_mapper.py has been run successfully."
                )

            logger.info(f"Loading {entity_type} from {filepath}")
            df = pd.read_csv(filepath)
            entities[entity_type] = df

            # Track entity counts
            self.metadata['entity_counts'][entity_type] = len(df)
            logger.info(f"  Loaded {len(df)} {entity_type} entities")

        logger.info("="*70)
        logger.info(
            f"Total entities loaded: {sum(self.metadata['entity_counts'].values())}")
        logger.info("="*70)

        return entities

    def _generate_embeddings(
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
        logger.info("\n" + "="*70)
        logger.info("GENERATING EMBEDDINGS")
        logger.info("="*70)

        embeddings = self.embedding_generator.generate_all(
            knowledge_df=entities['knowledge'],
            skills_df=entities['skills'],
            abilities_df=entities['abilities'],
            functions_df=entities['functions'],
            tasks_df=entities['tasks'],
            use_cache=self.use_cache
        )

        # Log embedding statistics
        logger.info("\n" + "="*70)
        logger.info("EMBEDDING STATISTICS")
        logger.info("="*70)

        for entity_type, entity_embeddings in embeddings.items():
            count = len(entity_embeddings)
            if count > 0:
                sample_embedding = next(iter(entity_embeddings.values()))
                dim = len(sample_embedding)
                logger.info(
                    f"{entity_type:12s}: {count:5d} embeddings (dimension: {dim})")

        logger.info("="*70)

        return embeddings

    def _calculate_relationships(
        self,
        entities: Dict[str, pd.DataFrame],
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all relationship types using similarity.

        Args:
            entities: Dictionary of entity DataFrames
            embeddings: Dictionary of embeddings

        Returns:
            Dictionary of relationship DataFrames
        """
        logger.info("\n" + "="*70)
        logger.info("CALCULATING RELATIONSHIPS")
        logger.info("="*70)

        relationships = self.similarity_calculator.calculate_all_relationships(
            knowledge_df=entities['knowledge'],
            skills_df=entities['skills'],
            abilities_df=entities['abilities'],
            functions_df=entities['functions'],
            tasks_df=entities['tasks'],
            embeddings=embeddings
        )

        # Track relationship counts
        for rel_type, rel_df in relationships.items():
            self.metadata['relationship_counts'][rel_type] = len(rel_df)

        return relationships

    def _save_relationships(self, relationships: Dict[str, pd.DataFrame]):
        """
        Save relationship DataFrames to CSV files.

        Args:
            relationships: Dictionary of relationship DataFrames
        """
        logger.info("\n" + "="*70)
        logger.info("SAVING RELATIONSHIPS")
        logger.info("="*70)

        self.similarity_calculator.save_relationships(
            relationships=relationships,
            output_dir=str(self.output_dir)
        )

        logger.info("All relationship files saved successfully")

    def _generate_pipeline_report(
        self,
        relationships: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate comprehensive pipeline execution report.

        Args:
            relationships: Dictionary of relationship DataFrames

        Returns:
            Formatted report string
        """
        report_lines = [
            "="*70,
            "EMBEDDING RELATIONSHIP BUILDER - PIPELINE REPORT",
            "="*70,
            f"Pipeline: {self.metadata['pipeline_name']}",
            f"Embedding Model: {self.metadata['embedding_model']}",
            f"Start Time: {self.metadata['start_time']}",
            f"End Time: {self.metadata['end_time']}",
            f"Duration: {self.metadata.get('duration', 'N/A')}",
            f"Status: {self.metadata['status']}",
            "",
            "="*70,
            "ENTITY SUMMARY",
            "="*70,
        ]

        total_entities = 0
        for entity_type, count in self.metadata['entity_counts'].items():
            report_lines.append(f"  {entity_type:12s}: {count:5d} entities")
            total_entities += count

        report_lines.extend([
            "",
            f"  Total Entities: {total_entities}",
            "",
            "="*70,
            "RELATIONSHIP SUMMARY",
            "="*70,
        ])

        total_relationships = 0
        for rel_type, rel_df in relationships.items():
            count = len(rel_df)
            total_relationships += count

            report_lines.append(f"\n{rel_type}:")
            report_lines.append(f"  Count: {count}")

            if count > 0:
                report_lines.append(
                    f"  Confidence Range: {rel_df['confidence_score'].min():.3f} - {rel_df['confidence_score'].max():.3f}")
                report_lines.append(
                    f"  Mean Confidence: {rel_df['confidence_score'].mean():.3f}")
                report_lines.append(
                    f"  Median Confidence: {rel_df['confidence_score'].median():.3f}")

                # Calculate average relationships per source entity
                unique_sources = rel_df['source_name'].nunique()
                avg_per_source = count / unique_sources if unique_sources > 0 else 0
                report_lines.append(
                    f"  Avg Relationships per Source: {avg_per_source:.1f}")

        report_lines.extend([
            "",
            "="*70,
            f"TOTAL RELATIONSHIPS INFERRED: {total_relationships}",
            "="*70,
            "",
            "OUTPUT FILES:",
        ])

        for rel_type in relationships.keys():
            filename = f"{rel_type}_inferred.csv"
            report_lines.append(f"  - {self.output_dir / filename}")

        report_lines.extend([
            "",
            "="*70,
            "NEXT STEPS:",
            "="*70,
            "1. Review the relationship files in the output directory",
            "2. Validate relationship quality and confidence scores",
            "3. Run onet_loader.py to load entities and relationships into database",
            "4. Execute onet_validator.py to verify data integrity",
            "",
            "="*70
        ])

        return "\n".join(report_lines)

    def _save_metadata(self):
        """Save pipeline metadata to JSON file."""
        metadata_file = self.output_dir / "embedding_relationship_metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Pipeline metadata saved to {metadata_file}")

    def build_relationships(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete relationship building pipeline.

        Returns:
            Dictionary of relationship DataFrames
        """
        try:
            # Update metadata
            self.metadata['start_time'] = datetime.now().isoformat()
            self.metadata['status'] = 'running'

            logger.info("\n" + "="*70)
            logger.info("STARTING EMBEDDING RELATIONSHIP BUILDER PIPELINE")
            logger.info("="*70)
            logger.info(f"Input Directory: {self.input_dir}")
            logger.info(f"Output Directory: {self.output_dir}")
            logger.info(f"Embedding Model: {self.embedding_model}")
            logger.info(f"Use Cache: {self.use_cache}")
            logger.info("="*70)

            # Step 1: Load mapped entities
            entities = self._load_mapped_entities()

            # Step 2: Generate embeddings
            embeddings = self._generate_embeddings(entities)

            # Step 3: Calculate relationships
            relationships = self._calculate_relationships(entities, embeddings)

            # Step 4: Save relationships
            self._save_relationships(relationships)

            # Step 5: Generate and save report
            report = self._generate_pipeline_report(relationships)

            report_file = self.output_dir / "embedding_relationship_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)

            logger.info(f"\nPipeline report saved to {report_file}")

            # Print report to console
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

            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Duration: {duration}")
            logger.info("="*70)

            return relationships

        except Exception as e:
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['end_time'] = datetime.now().isoformat()

            logger.error(
                f"Pipeline failed with error: {str(e)}", exc_info=True)
            self._save_metadata()

            raise

    def validate_inputs(self) -> bool:
        """
        Validate that all required input files exist.

        Returns:
            True if all inputs are valid, False otherwise
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
            filepath = self.input_dir / "ksamds" / filename
            if filepath.exists():
                logger.info(f"  ✓ {filename} found")
            else:
                logger.error(f"  ✗ {filename} NOT FOUND")
                all_valid = False

        return all_valid

    def get_statistics(self) -> Dict:
        """
        Get current pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            'metadata': self.metadata,
            'embedding_stats': self.embedding_generator.get_embedding_stats()
        }


def main():
    """
    Main execution function for standalone usage.
    """
    logger.info("="*70)
    logger.info("EMBEDDING RELATIONSHIP BUILDER")
    logger.info("="*70)

    # Initialize builder with default settings
    builder = EmbeddingRelationshipBuilder(
        input_dir="data/processed",
        output_dir="data/processed",
        embedding_cache_dir="data/embeddings",
        embedding_model="models/embedding-001",
        use_cache=True
    )

    # Validate inputs
    if not builder.validate_inputs():
        logger.error(
            "Input validation failed. Please run onet_mapper.py first.")
        return

    # Build relationships
    try:
        relationships = builder.build_relationships()
        logger.info("\n✓ Relationship building completed successfully!")
        logger.info(
            f"✓ Generated {sum(len(df) for df in relationships.values())} total relationships")

    except Exception as e:
        logger.error(f"\n✗ Relationship building failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
