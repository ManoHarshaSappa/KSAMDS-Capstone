"""
SkillsFuture Skills Framework ETL Pipeline Orchestrator for KSAMDS Project

This module orchestrates the complete ETL pipeline:
1. Extract Skills Framework data
2. Generate synthetic attributes
3. Map to KSAMDS structure
4. Infer relationships using embeddings
5. Load into PostgreSQL database
6. Validate data quality and integrity

Provides comprehensive logging, error handling, and rollback capabilities.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
import argparse
import shutil

# Import pipeline components
from skillsframework_extractor import SkillsFrameworkExtractor
from skillsframework_datagen import SyntheticAttributeGenerator
from skillsframework_mapper import SkillsFrameworkMapper
from skillsframework_relations import SkillsFrameworkRelationshipBuilder
from skillsframework_loader import SkillsFrameworkLoader, DatabaseConfig
from skillsframework_validator import SkillsFrameworkValidator


def setup_logging():
    """
    Configure logging for the pipeline.

    Returns:
        tuple: (logger, log_file_path, latest_log_path)
    """
    # Get project root and setup log directory
    project_root = Path(__file__).parent.parent.parent.parent
    logs_dir = project_root / "data" / "skillsframework" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Get timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"sf_pipeline_{timestamp}.log"
    latest_log = logs_dir / "sf_pipeline_latest.log"

    # Create handlers with explicit flushing
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    latest_handler = logging.FileHandler(
        latest_log, mode='w', encoding='utf-8')
    latest_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    latest_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(latest_handler)

    # Get module logger
    logger = logging.getLogger(__name__)

    return logger, log_file, latest_log


@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""
    data_dir: str = "data"
    force_download: bool = False
    skip_extraction: bool = False
    skip_datagen: bool = False
    skip_mapping: bool = False
    skip_relationship_inference: bool = False
    skip_loading: bool = False
    skip_validation: bool = False
    include_statistics: bool = False
    cleanup_intermediate: bool = True
    db_config: Optional[DatabaseConfig] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    stage: str
    duration: float
    message: str
    details: Dict[str, Any]


class PipelineOrchestrator:
    """Orchestrate the complete Skills Framework ETL pipeline."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.results: Dict[str, PipelineResult] = {}
        self.start_time = None
        self.end_time = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.extractor = None
        self.datagen = None
        self.mapper = None
        self.relationship_builder = None
        self.loader = None
        self.validator = None

        # Get project root directory (3 levels up from skillsframework_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        # Setup directories
        self.data_dir = project_root / config.data_dir
        self.archive_dir = self.data_dir / "skillsframework" / "archive"
        self.reports_dir = self.data_dir / "skillsframework" / "reports"
        self.temp_dir = self.data_dir / "temp"

        self.logger.info(f"PipelineOrchestrator initialized")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Archive directory: {self.archive_dir}")
        self.logger.info(f"Reports directory: {self.reports_dir}")

    def log_stage_start(self, stage: str):
        """Log the start of a pipeline stage."""
        self.logger.info("=" * 80)
        self.logger.info(f"STAGE: {stage}")
        self.logger.info("=" * 80)

    def log_stage_complete(self, stage: str, duration: float, success: bool):
        """Log the completion of a pipeline stage."""
        status = "COMPLETED" if success else "FAILED"
        self.logger.info(f"{stage} {status} in {duration:.2f} seconds")
        self.logger.info("")

    def run_extraction(self) -> PipelineResult:
        """Run the extraction stage."""
        stage = "EXTRACTION"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.extractor = SkillsFrameworkExtractor(
                data_dir=self.config.data_dir
            )

            success = self.extractor.extract_all()

            if not success:
                raise Exception("Extraction failed")

            summary = self.extractor.get_data_summary()

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="Skills Framework data extracted successfully",
                details={"summary": summary}
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"Extraction failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Extraction failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_datagen(self) -> PipelineResult:
        """Run the synthetic attribute generation stage."""
        stage = "DATA_GENERATION"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.datagen = SyntheticAttributeGenerator(
                model_name="models/gemini-embedding-001",
                batch_size=100,
                confidence_threshold=0.5
            )

            success = self.datagen.process_all(save_to_csv=True)

            if not success:
                raise Exception("Synthetic attribute generation failed")

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="Generated synthetic attributes for tasks and competencies",
                details={
                    "total_processed": self.datagen.stats['total_processed'],
                    "api_calls": self.datagen.stats['api_calls'],
                    "cache_hits": self.datagen.stats['cache_hits'],
                    "cache_misses": self.datagen.stats['cache_misses']
                }
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"Data generation failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Data generation failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_mapping(self) -> PipelineResult:
        """Run the mapping stage."""
        stage = "MAPPING"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.mapper = SkillsFrameworkMapper()

            success = self.mapper.map_all_entities(
                cleanup_after=self.config.cleanup_intermediate
            )

            if not success:
                raise Exception("Mapping failed")

            summary = self.mapper.get_mapping_summary()

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="Mapped Skills Framework to KSAMDS structure",
                details={"summary": summary}
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"Mapping failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Mapping failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_relationship_inference(self) -> PipelineResult:
        """Run the relationship inference stage."""
        stage = "RELATIONSHIP_INFERENCE"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.relationship_builder = SkillsFrameworkRelationshipBuilder(
                embedding_model="models/gemini-embedding-001",
                use_cache=True
            )

            if not self.relationship_builder.validate_inputs():
                raise Exception(
                    "Relationship inference input validation failed")

            relationships = self.relationship_builder.build_relationships()

            if not relationships:
                raise Exception("No relationships generated")

            total_relationships = sum(len(df) for df in relationships.values())
            relationship_counts = {
                rel_type: len(df) for rel_type, df in relationships.items()
            }

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message=f"Inferred {total_relationships:,} relationships",
                details={
                    "total_relationships": total_relationships,
                    "relationship_counts": relationship_counts
                }
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(
                f"Relationship inference failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Relationship inference failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_loading(self) -> PipelineResult:
        """Run the database loading stage."""
        stage = "LOADING"
        self.log_stage_start(stage)
        start = time.time()

        try:
            if not self.config.db_config:
                raise ValueError("Database configuration not provided")

            self.loader = SkillsFrameworkLoader(self.config.db_config)

            success = self.loader.load_all_data()

            if not success:
                raise Exception("Database loading failed")

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="Loaded data into PostgreSQL database",
                details={"insertion_stats": self.loader.insertion_stats}
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"Loading failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Loading failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_validation(self) -> PipelineResult:
        """Run the validation stage."""
        stage = "VALIDATION"
        self.log_stage_start(stage)
        start = time.time()

        try:
            if not self.config.db_config:
                raise ValueError("Database configuration not provided")

            self.validator = SkillsFrameworkValidator(self.config.db_config)

            success = self.validator.validate_all()

            # Count failures by severity
            total_checks = len(self.validator.validation_results)
            passed = sum(
                1 for r in self.validator.validation_results if r.passed)
            failed = sum(
                1 for r in self.validator.validation_results if not r.passed)
            critical_failures = sum(
                1 for r in self.validator.validation_results
                if r.severity == "CRITICAL" and not r.passed
            )

            duration = time.time() - start
            result = PipelineResult(
                success=success,
                stage=stage,
                duration=duration,
                message=f"Validation: {passed}/{total_checks} checks passed",
                details={
                    "total_checks": total_checks,
                    "passed": passed,
                    "failed": failed,
                    "critical_failures": critical_failures
                }
            )

            self.log_stage_complete(stage, duration, success)
            return result

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"Validation failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Validation failed: {str(e)}",
                details={}
            )
            self.log_stage_complete(stage, duration, False)
            return result

    def run_pipeline(self) -> bool:
        """
        Execute the complete pipeline.

        Returns:
            bool: True if all stages successful
        """
        self.start_time = time.time()
        self.end_time = None  # Initialize to None, will be set at completion
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SKILLS FRAMEWORK ETL PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")

        # Stage 1: Extraction
        if not self.config.skip_extraction:
            self.results['extraction'] = self.run_extraction()
            if not self.results['extraction'].success:
                self.logger.error("Pipeline stopped due to extraction failure")
                return False
        else:
            self.logger.info("SKIPPING EXTRACTION (using existing data)")

        # Stage 2: Data Generation
        if not self.config.skip_datagen:
            self.results['datagen'] = self.run_datagen()
            if not self.results['datagen'].success:
                self.logger.error(
                    "Pipeline stopped due to data generation failure")
                return False
        else:
            self.logger.info(
                "SKIPPING DATA GENERATION (using existing synthetic attributes)")

        # Stage 3: Mapping
        if not self.config.skip_mapping:
            self.results['mapping'] = self.run_mapping()
            if not self.results['mapping'].success:
                self.logger.error("Pipeline stopped due to mapping failure")
                return False
        else:
            self.logger.info("SKIPPING MAPPING (using existing mapped data)")

        # Stage 4: Relationship Inference
        if not self.config.skip_relationship_inference:
            self.results['relationship_inference'] = self.run_relationship_inference()
            if not self.results['relationship_inference'].success:
                self.logger.error(
                    "Pipeline stopped due to relationship inference failure")
                return False
        else:
            self.logger.info(
                "SKIPPING RELATIONSHIP INFERENCE (using existing relationships)")

        # Stage 5: Loading
        if not self.config.skip_loading:
            self.results['loading'] = self.run_loading()
            if not self.results['loading'].success:
                self.logger.error("Pipeline stopped due to loading failure")
                return False
        else:
            self.logger.info("SKIPPING LOADING (database not modified)")

        # Stage 6: Validation
        if not self.config.skip_validation:
            self.results['validation'] = self.run_validation()
            # Validation failure is non-fatal - we log but continue
            if not self.results['validation'].success:
                self.logger.warning(
                    "Validation completed with failures - check reports")
        else:
            self.logger.info("SKIPPING VALIDATION")

        self.end_time = time.time()
        total_duration = self.end_time - self.start_time

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SKILLS FRAMEWORK ETL PIPELINE COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.logger.info("")

        return all(r.success for r in self.results.values())

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("SKILLS FRAMEWORK ETL PIPELINE - EXECUTION REPORT")
        lines.append("=" * 80)
        lines.append(
            f"Pipeline Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        # Handle case where end_time might not be set (if pipeline failed early)
        if self.end_time is None:
            self.end_time = time.time()

        lines.append(
            f"Pipeline Ended: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        total_duration = self.end_time - self.start_time
        lines.append(
            f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        lines.append("")

        # Stage results
        lines.append("STAGE RESULTS:")
        lines.append("-" * 80)
        for stage_name, result in self.results.items():
            status = "✓ PASSED" if result.success else "✗ FAILED"
            lines.append(
                f"{stage_name.upper():30s} {status:10s} ({result.duration:.2f}s)")
            if result.message:
                lines.append(f"  └─ {result.message}")
        lines.append("")

        # Detailed statistics
        lines.append("DETAILED STATISTICS:")
        lines.append("-" * 80)

        # Extraction details
        if 'extraction' in self.results and self.results['extraction'].success:
            summary = self.results['extraction'].details.get('summary', {})
            lines.append("\nExtraction:")
            for key, info in summary.items():
                if isinstance(info, dict):
                    count = info.get('rows', info.get('count', 0))
                    lines.append(f"  - {key}: {count:,} records")

        # Data generation details
        if 'datagen' in self.results and self.results['datagen'].success:
            details = self.results['datagen'].details
            lines.append("\nData Generation:")
            lines.append(
                f"  Total processed: {details.get('total_processed', 0):,}")
            lines.append(f"  API calls: {details.get('api_calls', 0):,}")
            lines.append(f"  Cache hits: {details.get('cache_hits', 0):,}")
            lines.append(f"  Cache misses: {details.get('cache_misses', 0):,}")

        # Mapping details
        if 'mapping' in self.results and self.results['mapping'].success:
            summary = self.results['mapping'].details.get('summary', {})
            lines.append("\nMapping:")
            entities = summary.get('entities', {})
            for entity, count in entities.items():
                lines.append(f"  - {entity}: {count:,} entities")

            relationships = summary.get('relationships', {})
            if relationships:
                lines.append("\n  Relationships:")
                for rel_type, count in relationships.items():
                    lines.append(f"    - {rel_type}: {count:,}")

        # Relationship Inference details
        if 'relationship_inference' in self.results and self.results['relationship_inference'].success:
            counts = self.results['relationship_inference'].details.get(
                'relationship_counts', {})
            total = self.results['relationship_inference'].details.get(
                'total_relationships', 0)
            lines.append("\nRelationship Inference:")
            lines.append(f"  Total inferred relationships: {total:,}")
            for rel_type, count in counts.items():
                lines.append(f"    - {rel_type}: {count:,}")

        # Loading details
        if 'loading' in self.results and self.results['loading'].success:
            stats = self.results['loading'].details.get('insertion_stats', {})
            lines.append("\nLoading:")
            for entity, count in stats.items():
                if entity != 'errors' and count > 0:
                    lines.append(f"  - {entity}: {count:,} inserted")

        # Validation details
        if 'validation' in self.results:
            details = self.results['validation'].details
            lines.append("\nValidation:")
            lines.append(f"  - Total Checks: {details.get('total_checks', 0)}")
            lines.append(f"  - Passed: {details.get('passed', 0)}")
            lines.append(f"  - Failed: {details.get('failed', 0)}")
            if details.get('critical_failures', 0) > 0:
                lines.append(
                    f"  - Critical Failures: {details.get('critical_failures', 0)}")

        lines.append("")

        # Directory structure summary
        lines.append("OUTPUT STRUCTURE:")
        lines.append("-" * 80)
        lines.append(f"Archive: {self.archive_dir}")
        lines.append(
            "  ├── raw/                 (Skills Framework JSON files)")
        lines.append("  ├── mapped/              (Mapped entity CSVs)")
        lines.append("  ├── relationships/       (Inferred relationships)")
        lines.append("  └── embeddings/          (Embedding cache)")
        lines.append("")
        lines.append(f"Reports: {self.reports_dir}")
        lines.append("  ├── latest_sf_pipeline_report.txt")
        lines.append("  └── latest_sf_validation_report.txt")
        lines.append("")

        lines.append("=" * 80)

        overall_success = all(r.success for r in self.results.values())
        if overall_success:
            lines.append("PIPELINE COMPLETED SUCCESSFULLY ✓")
        else:
            lines.append("PIPELINE COMPLETED WITH ERRORS ✗")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_execution_report(self):
        """Save execution report to file (only latest version)."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Text report only
        text_report = self.generate_summary_report()
        latest_text = self.reports_dir / "latest_sf_pipeline_report.txt"
        with open(latest_text, 'w') as f:
            f.write(text_report)
        self.logger.info(f"Pipeline report saved to {latest_text}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Skills Framework ETL Pipeline Orchestrator for KSAMDS"
    )

    parser.add_argument(
        '--data-dir',
        default='data',
        help='Base directory for data storage (default: data)'
    )

    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of Skills Framework data'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extraction stage (use existing data)'
    )

    parser.add_argument(
        '--skip-datagen',
        action='store_true',
        help='Skip data generation stage (use existing synthetic attributes)'
    )

    parser.add_argument(
        '--skip-mapping',
        action='store_true',
        help='Skip mapping stage (use existing mapped data)'
    )

    parser.add_argument(
        '--skip-relationship-inference',
        action='store_true',
        help='Skip relationship inference stage'
    )

    parser.add_argument(
        '--skip-loading',
        action='store_true',
        help='Skip loading stage'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation stage'
    )

    parser.add_argument(
        '--include-statistics',
        action='store_true',
        help='Include detailed statistics in validation (may be slow)'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Disable cleanup of intermediate files'
    )

    parser.add_argument(
        '--db-host',
        default='localhost',
        help='Database host (default: localhost)'
    )

    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='Database port (default: 5432)'
    )

    parser.add_argument(
        '--db-name',
        default='ksamds',
        help='Database name (default: ksamds)'
    )

    parser.add_argument(
        '--db-user',
        default='postgres',
        help='Database user (default: postgres)'
    )

    parser.add_argument(
        '--db-password',
        help='Database password (uses DB_PASSWORD env var if not provided)'
    )

    parser.add_argument(
        '--db-schema',
        default='ksamds',
        help='Database schema (default: ksamds)'
    )

    return parser.parse_args()


def main():
    """Main function to run the orchestrator."""
    # Setup logging first
    logger, log_file, latest_log = setup_logging()

    args = parse_arguments()

    # Get database password from env or args
    import os
    db_password = args.db_password or os.getenv('DB_PASSWORD')

    if not db_password:
        raise ValueError(
            "Database password must be provided via --db-password argument or DB_PASSWORD environment variable")

    # Create database config
    db_config = DatabaseConfig(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        username=args.db_user,
        password=db_password,
        schema=args.db_schema
    )

    # Create pipeline config
    config = PipelineConfig(
        data_dir=args.data_dir,
        force_download=args.force_download,
        skip_extraction=args.skip_extraction,
        skip_datagen=args.skip_datagen,
        skip_mapping=args.skip_mapping,
        skip_relationship_inference=args.skip_relationship_inference,
        skip_loading=args.skip_loading,
        skip_validation=args.skip_validation,
        include_statistics=args.include_statistics,
        cleanup_intermediate=not args.no_cleanup,
        db_config=db_config
    )

    # Create orchestrator and run pipeline
    orchestrator = PipelineOrchestrator(config)

    try:
        success = orchestrator.run_pipeline()

        # Generate and print summary
        summary = orchestrator.generate_summary_report()
        print(summary)

        # Save reports
        orchestrator.save_execution_report()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(
            f"Pipeline failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
