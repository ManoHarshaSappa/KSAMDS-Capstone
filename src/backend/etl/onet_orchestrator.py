"""
O*NET ETL Pipeline Orchestrator for KSAMDS Project

This module orchestrates the complete ETL pipeline:
1. Extract O*NET data
2. Map to KSAMDS structure
3. Generate embeddings and infer relationships
4. Load into PostgreSQL database
5. Validate data quality and integrity

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
from onet_extractor import ONetExtractor
from onet_datagen import ONetSyntheticAttributeGenerator
from onet_mapper import ONetMapper
from onet_loader import ONetLoader, DatabaseConfig
from onet_validator import ONetValidator
from onet_relations import ONetRelationshipBuilder


def setup_logging():
    """
    Configure logging for the pipeline.

    Returns:
        tuple: (logger, log_file_path, latest_log_path)
    """
    # Get project root and setup log directory
    project_root = Path(__file__).parent.parent.parent.parent
    logs_dir = project_root / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Get timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    latest_log = logs_dir / "pipeline_latest.log"

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
    """Orchestrate the complete O*NET ETL pipeline."""

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
        self.loader = None
        self.validator = None
        self.relationship_builder = None

        # Get project root directory (3 levels up from etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        # Setup directories
        self.data_dir = project_root / config.data_dir
        self.archive_dir = self.data_dir / "archive"
        self.reports_dir = self.data_dir / "reports"
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
            self.extractor = ONetExtractor(
                data_dir=self.config.data_dir
            )

            success = self.extractor.extract_all(
                force_download=self.config.force_download,
                cleanup_after=self.config.cleanup_intermediate
            )

            if not success:
                raise Exception("Extraction failed")

            summary = self.extractor.get_data_summary()

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="O*NET data extracted successfully",
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
            self.datagen = ONetSyntheticAttributeGenerator(
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
                message="Generated synthetic attributes for tasks and functions",
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
            self.mapper = ONetMapper()

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
                message="O*NET data mapped to KSAMDS structure",
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
        """Run the relationship inference stage using embeddings."""
        stage = "RELATIONSHIP_INFERENCE"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.relationship_builder = ONetRelationshipBuilder(
                embedding_model="models/gemini-embedding-001",
                use_cache=True
            )

            # Validate inputs
            if not self.relationship_builder.validate_inputs():
                raise Exception(
                    "Input validation failed - ensure onet_mapper.py completed successfully")

            # Build relationships
            relationships = self.relationship_builder.build_relationships()

            if not relationships:
                raise Exception("Relationship inference failed")

            # Get statistics
            stats = self.relationship_builder.get_statistics()
            total_relationships = sum(len(df) for df in relationships.values())
            relationship_counts = {k: len(v) for k, v in relationships.items()}

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message=f"Inferred {total_relationships:,} relationships using embeddings",
                details={
                    "relationship_counts": relationship_counts,
                    "total_relationships": total_relationships,
                    "statistics": stats
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
        """Run the loading stage."""
        stage = "LOADING"
        self.log_stage_start(stage)
        start = time.time()

        try:
            if self.config.db_config is None:
                raise ValueError(
                    "Database configuration is required for loading")

            self.loader = ONetLoader(
                db_config=self.config.db_config
            )

            success = self.loader.load_all_data()

            if not success:
                raise Exception("Loading failed")

            stats = self.loader.insertion_stats

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message="Data loaded into KSAMDS database",
                details={"insertion_stats": stats}
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
            if self.config.db_config is None:
                raise ValueError(
                    "Database configuration is required for validation")

            self.validator = ONetValidator(
                db_config=self.config.db_config
            )

            # Run validation
            success = self.validator.validate_all()

            # Get summary
            total_checks = len(self.validator.validation_results)
            passed = sum(
                1 for r in self.validator.validation_results if r.passed)
            failed = total_checks - passed
            critical = sum(1 for r in self.validator.validation_results
                           if r.severity == "CRITICAL" and not r.passed)

            duration = time.time() - start
            result = PipelineResult(
                success=success and critical == 0,
                stage=stage,
                duration=duration,
                message=f"Validation complete: {passed}/{total_checks} checks passed",
                details={
                    "total_checks": total_checks,
                    "passed": passed,
                    "failed": failed,
                    "critical_failures": critical
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

    def cleanup_pipeline_artifacts(self):
        """Clean up temporary and intermediate files after pipeline completion."""
        if not self.config.cleanup_intermediate:
            self.logger.info("Cleanup disabled - skipping artifact cleanup")
            return

        self.logger.info("="*70)
        self.logger.info("CLEANING UP PIPELINE ARTIFACTS")
        self.logger.info("="*70)

        # Clean up temp directory
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(
                    f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temp directory: {e}")

        # Clean up old embedding cache if relationship builder was used
        if self.relationship_builder:
            try:
                # Get datagen cache files to preserve them
                datagen_cache_files = set()
                if self.datagen and hasattr(self.datagen, '_session_cache_files'):
                    datagen_cache_files = self.datagen._session_cache_files
                    self.logger.info(
                        f"Preserving {len(datagen_cache_files)} datagen embedding files")

                # Add datagen cache files to relationship builder's session files
                if datagen_cache_files:
                    self.relationship_builder._session_cache_files.update(
                        datagen_cache_files)

                self.relationship_builder.cleanup_intermediate_files(
                    keep_embeddings=False
                )
                self.logger.info(
                    "Cleaned up old embedding cache files (preserved datagen embeddings)")
            except Exception as e:
                self.logger.warning(f"Failed to clean up embeddings: {e}")

        self.logger.info("Pipeline artifact cleanup completed")

    def run_pipeline(self) -> bool:
        """
        Run the complete ETL pipeline.

        Returns:
            bool: True if all stages completed successfully
        """
        self.logger.info("=" * 80)
        self.logger.info("KSAMDS ETL PIPELINE START")
        self.logger.info(
            f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"Cleanup Intermediate Files: {self.config.cleanup_intermediate}")
        self.logger.info("=" * 80)
        self.logger.info("")

        self.start_time = time.time()
        overall_success = True

        # Stage 1: Extraction
        if not self.config.skip_extraction:
            result = self.run_extraction()
            self.results['extraction'] = result
            if not result.success:
                overall_success = False
                self.logger.error("Pipeline halted due to extraction failure")
                return False
        else:
            self.logger.info("Skipping extraction stage")

        # Stage 2: Data Generation (Synthetic Attributes)
        if not self.config.skip_datagen:
            result = self.run_datagen()
            self.results['data_generation'] = result
            if not result.success:
                overall_success = False
                self.logger.error(
                    "Pipeline halted due to data generation failure")
                return False
        else:
            self.logger.info("Skipping data generation stage")

        # Stage 3: Mapping
        if not self.config.skip_mapping:
            result = self.run_mapping()
            self.results['mapping'] = result
            if not result.success:
                overall_success = False
                self.logger.error("Pipeline halted due to mapping failure")
                return False
        else:
            self.logger.info("Skipping mapping stage")

        # Stage 4: Relationship Inference
        if not self.config.skip_relationship_inference:
            result = self.run_relationship_inference()
            self.results['relationship_inference'] = result
            if not result.success:
                overall_success = False
                self.logger.error(
                    "Pipeline halted due to relationship inference failure")
                return False
        else:
            self.logger.info("Skipping relationship inference stage")

        # Stage 5: Loading
        if not self.config.skip_loading:
            result = self.run_loading()
            self.results['loading'] = result
            if not result.success:
                overall_success = False
                self.logger.error("Pipeline halted due to loading failure")
                return False
        else:
            self.logger.info("Skipping loading stage")

        # Stage 6: Validation
        if not self.config.skip_validation:
            result = self.run_validation()
            self.results['validation'] = result
            if not result.success:
                overall_success = False
                self.logger.warning("Validation stage reported issues")
                # Don't halt pipeline - validation failures are warnings
        else:
            self.logger.info("Skipping validation stage")

        # Cleanup artifacts if enabled
        if self.config.cleanup_intermediate:
            self.cleanup_pipeline_artifacts()

        self.end_time = time.time()
        return overall_success

    def generate_summary_report(self) -> str:
        """Generate a summary report of the pipeline execution."""
        if self.end_time:
            total_duration = self.end_time - self.start_time
        else:
            total_duration = time.time() - self.start_time if self.start_time else 0

        lines = [
            "",
            "=" * 80,
            "PIPELINE EXECUTION SUMMARY",
            "=" * 80,
            f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}",
            f"Completed: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}",
            f"Total Duration: {f'{total_duration:.2f} seconds ({total_duration/60:.2f} minutes)' if total_duration else 'N/A'}",

            f"Cleanup Intermediate Files: {self.config.cleanup_intermediate}",
            "",
            "STAGE RESULTS:",
            "-" * 80
        ]

        for stage, result in self.results.items():
            status = "SUCCESS" if result.success else "FAILED"
            lines.append(
                f"{stage.upper():25} {status:10} {result.duration:8.2f}s - {result.message}")

        lines.append("")
        lines.append("DETAILS:")
        lines.append("-" * 80)

        # Extraction details
        if 'extraction' in self.results and self.results['extraction'].success:
            summary = self.results['extraction'].details.get('summary', {})
            lines.append("Extraction:")
            for key, info in summary.items():
                if isinstance(info, dict) and 'rows' in info:
                    lines.append(f"  - {key}: {info['rows']:,} rows")

        # Data Generation details
        if 'data_generation' in self.results and self.results['data_generation'].success:
            details = self.results['data_generation'].details
            lines.append("\nData Generation:")
            lines.append(
                f"  - Total records processed: {details.get('total_processed', 0):,}")
            lines.append(f"  - API calls: {details.get('api_calls', 0):,}")
            lines.append(f"  - Cache hits: {details.get('cache_hits', 0):,}")
            lines.append(
                f"  - Cache misses: {details.get('cache_misses', 0):,}")

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
        lines.append("  ├── raw/                 (O*NET ZIP file)")
        lines.append("  ├── mapped/              (Mapped entity CSVs)")
        lines.append("  ├── relationships/       (Inferred relationships)")
        lines.append("  └── embeddings/          (Embedding cache)")
        lines.append("")
        lines.append(f"Reports: {self.reports_dir}")
        lines.append("  ├── latest_pipeline_report.txt")
        lines.append("  └── latest_validation_report.txt")
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
        latest_text = self.reports_dir / "latest_pipeline_report.txt"
        with open(latest_text, 'w') as f:
            f.write(text_report)
        self.logger.info(f"Pipeline report saved to {latest_text}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="O*NET ETL Pipeline Orchestrator for KSAMDS"
    )

    parser.add_argument(
        '--data-dir',
        default='data',
        help='Base directory for data storage (default: data)'
    )

    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of O*NET database'
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
