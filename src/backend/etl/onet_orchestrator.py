"""
O*NET ETL Pipeline Orchestrator for KSAMDS Project

This module orchestrates the complete ETL pipeline:
1. Extract O*NET data
2. Map to KSAMDS structure
3. Load into PostgreSQL database
4. Validate data quality and integrity

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

# Import pipeline components
from onet_extractor import ONetExtractor
from onet_mapper import ONetMapper
from onet_loader import ONetLoader, DatabaseConfig
from onet_validator import ONetValidator
from embedding_relationship_builder import EmbeddingRelationshipBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""
    data_dir: str = "data"
    force_download: bool = False
    skip_extraction: bool = False
    skip_mapping: bool = False
    skip_relationship_inference: bool = False  # NEW!
    skip_loading: bool = False
    skip_validation: bool = False
    include_statistics: bool = False
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

        # Initialize components
        self.extractor = None
        self.mapper = None
        self.loader = None
        self.validator = None

    def log_stage_start(self, stage: str):
        """Log the start of a pipeline stage."""
        logger.info("=" * 80)
        logger.info(f"STAGE: {stage}")
        logger.info("=" * 80)

    def log_stage_complete(self, stage: str, duration: float, success: bool):
        """Log the completion of a pipeline stage."""
        status = "COMPLETED" if success else "FAILED"
        logger.info(f"{stage} {status} in {duration:.2f} seconds")
        logger.info("")

    def run_extraction(self) -> PipelineResult:
        """Run the extraction stage."""
        stage = "EXTRACTION"
        self.log_stage_start(stage)
        start = time.time()

        try:
            self.extractor = ONetExtractor(data_dir=self.config.data_dir)

            success = self.extractor.extract_all(
                force_download=self.config.force_download
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
            logger.error(f"Extraction failed: {e}", exc_info=True)
            result = PipelineResult(
                success=False,
                stage=stage,
                duration=duration,
                message=f"Extraction failed: {str(e)}",
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
            self.mapper = ONetMapper(data_dir=self.config.data_dir)

            success = self.mapper.map_all_entities()

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
            logger.error(f"Mapping failed: {e}", exc_info=True)
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
            builder = EmbeddingRelationshipBuilder(
                input_dir=str(Path(self.config.data_dir) / "processed"),
                output_dir=str(Path(self.config.data_dir) / "processed"),
                embedding_cache_dir=str(
                    Path(self.config.data_dir) / "embeddings"),
                embedding_model="models/embedding-001",  # Google AI model
                use_cache=True
            )

            # Validate inputs
            if not builder.validate_inputs():
                raise Exception(
                    "Input validation failed - ensure onet_mapper.py completed successfully")

            # Build relationships
            relationships = builder.build_relationships()

            if not relationships:
                raise Exception("Relationship inference failed")

            # Get statistics
            total_relationships = sum(len(df) for df in relationships.values())
            relationship_counts = {k: len(v) for k, v in relationships.items()}

            duration = time.time() - start
            result = PipelineResult(
                success=True,
                stage=stage,
                duration=duration,
                message=f"Inferred {total_relationships} relationships using embeddings",
                details={
                    "relationship_counts": relationship_counts,
                    "total_relationships": total_relationships
                }
            )

            self.log_stage_complete(stage, duration, True)
            return result

        except Exception as e:
            duration = time.time() - start
            logger.error(f"Relationship inference failed: {e}", exc_info=True)
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
                db_config=self.config.db_config,
                data_dir=self.config.data_dir
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
            logger.error(f"Loading failed: {e}", exc_info=True)
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
                db_config=self.config.db_config,
                output_dir=str(Path(self.config.data_dir) / "validation")
            )

            # Run validation with optional statistics
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
            logger.error(f"Validation failed: {e}", exc_info=True)
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
        Run the complete ETL pipeline.

        Returns:
            bool: True if all stages completed successfully
        """
        logger.info("=" * 80)
        logger.info("KSAMDS ETL PIPELINE START")
        logger.info(
            f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("")

        self.start_time = time.time()
        overall_success = True

        # Stage 1: Extraction
        if not self.config.skip_extraction:
            result = self.run_extraction()
            self.results['extraction'] = result
            if not result.success:
                overall_success = False
                logger.error("Pipeline halted due to extraction failure")
                return False
        else:
            logger.info("Skipping extraction stage")

        # Stage 2: Mapping
        if not self.config.skip_mapping:
            result = self.run_mapping()
            self.results['mapping'] = result
            if not result.success:
                overall_success = False
                logger.error("Pipeline halted due to mapping failure")
                return False
        else:
            logger.info("Skipping mapping stage")

        # Stage 3: Relationship Inference (NEW!)
        if not self.config.skip_relationship_inference:
            result = self.run_relationship_inference()
            self.results['relationship_inference'] = result
            if not result.success:
                overall_success = False
                logger.error(
                    "Pipeline halted due to relationship inference failure")
                return False
        else:
            logger.info("Skipping relationship inference stage")

        # Stage 4: Loading
        if not self.config.skip_loading:
            result = self.run_loading()
            self.results['loading'] = result
            if not result.success:
                overall_success = False
                logger.error("Pipeline halted due to loading failure")
                return False
        else:
            logger.info("Skipping loading stage")

        # Stage 5: Validation
        if not self.config.skip_validation:
            result = self.run_validation()
            self.results['validation'] = result
            if not result.success:
                overall_success = False
                logger.warning("Validation stage reported issues")
                # Don't halt pipeline - validation failures are warnings
        else:
            logger.info("Skipping validation stage")

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
            f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Completed: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)",
            "",
            "STAGE RESULTS:",
            "-" * 80
        ]

        for stage, result in self.results.items():
            status = "SUCCESS" if result.success else "FAILED"
            lines.append(
                f"{stage.upper():15} {status:10} {result.duration:8.2f}s - {result.message}")

        lines.append("")
        lines.append("DETAILS:")
        lines.append("-" * 80)

        # Extraction details
        if 'extraction' in self.results and self.results['extraction'].success:
            summary = self.results['extraction'].details.get('summary', {})
            lines.append("Extraction:")
            for key, info in summary.items():
                if isinstance(info, dict) and 'rows' in info:
                    lines.append(f"  - {key}: {info['rows']} rows")

        # Mapping details
        if 'mapping' in self.results and self.results['mapping'].success:
            summary = self.results['mapping'].details.get('summary', {})
            lines.append("\nMapping:")
            entities = summary.get('entities', {})
            for entity, count in entities.items():
                lines.append(f"  - {entity}: {count} entities")

            relationships = summary.get('relationships', {})
            if relationships:
                lines.append("\n  Relationships:")
                for rel_type, count in relationships.items():
                    lines.append(f"    - {rel_type}: {count}")

        # Relationship Inference details
        if 'relationship_inference' in self.results and self.results['relationship_inference'].success:
            counts = self.results['relationship_inference'].details.get(
                'relationship_counts', {})
            total = self.results['relationship_inference'].details.get(
                'total_relationships', 0)
            lines.append("\nRelationship Inference:")
            lines.append(f"  Total inferred relationships: {total}")
            for rel_type, count in counts.items():
                lines.append(f"    - {rel_type}: {count}")

        # Loading details
        if 'loading' in self.results and self.results['loading'].success:
            stats = self.results['loading'].details.get('insertion_stats', {})
            lines.append("\nLoading:")
            for entity, count in stats.items():
                if entity != 'errors' and count > 0:
                    lines.append(f"  - {entity}: {count} inserted")

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
        lines.append("=" * 80)

        overall_success = all(r.success for r in self.results.values())
        if overall_success:
            lines.append("PIPELINE COMPLETED SUCCESSFULLY")
        else:
            lines.append("PIPELINE COMPLETED WITH ERRORS")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_execution_report(self):
        """Save execution report to file."""
        report_dir = Path(self.config.data_dir) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Text report
        text_report = self.generate_summary_report()
        text_path = report_dir / f"pipeline_execution_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)
        logger.info(f"Execution report saved to {text_path}")

        # JSON report
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.end_time - self.start_time if self.end_time else 0,
            "success": all(r.success for r in self.results.values()),
            "stages": {
                stage: {
                    "success": result.success,
                    "duration": result.duration,
                    "message": result.message,
                    "details": result.details
                }
                for stage, result in self.results.items()
            }
        }

        json_path = report_dir / f"pipeline_execution_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")


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
    args = parse_arguments()

    # Get database password from env or args
    import os
    db_password = args.db_password or os.getenv('DB_PASSWORD', '8086')

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
        skip_mapping=args.skip_mapping,
        skip_loading=args.skip_loading,
        skip_validation=args.skip_validation,
        include_statistics=args.include_statistics,
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
