"""
O*NET Data Validator for KSAMDS Project

This module validates the mapped and loaded O*NET data in the KSAMDS database.
It checks data integrity, relationship validity, dimension assignments,
and generates comprehensive quality reports.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "ksamds"


class ONetValidator:
    """Validate KSAMDS database integrity and quality."""

    def __init__(self, db_config: DatabaseConfig, output_dir: str = "data/validation"):
        """
        Initialize the validator.

        Args:
            db_config: Database connection configuration
            output_dir: Directory for validation reports
        """
        self.db_config = db_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validation_results: List[ValidationResult] = []
        self.stats = {}

    def get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.username,
            password=self.db_config.password
        )

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute query and return results as list of dicts."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def execute_count_query(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute query and return count."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SET search_path TO {self.db_config.schema}, public")
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Count query failed: {e}")
            return 0

    def add_result(self, check_name: str, passed: bool, message: str,
                   details: Optional[Dict] = None, severity: str = "INFO"):
        """Add validation result."""
        result = ValidationResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details=details or {},
            severity=severity
        )
        self.validation_results.append(result)

        # Log based on severity
        log_message = f"{check_name}: {message}"
        if severity == "CRITICAL":
            logger.critical(log_message)
        elif severity == "ERROR":
            logger.error(log_message)
        elif severity == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    # ==========================================
    # Core Entity Validation
    # ==========================================

    def validate_entity_counts(self) -> bool:
        """Validate that entities were loaded."""
        logger.info("Validating entity counts...")

        entities = ['knowledge', 'skill', 'ability',
                    'occupation', 'task', 'function']
        all_valid = True

        for entity in entities:
            count = self.execute_count_query(f"SELECT COUNT(*) FROM {entity}")
            self.stats[f'{entity}_count'] = count

            if count == 0:
                self.add_result(
                    f"{entity}_count",
                    False,
                    f"No {entity} entities found in database",
                    {"count": count},
                    "CRITICAL"
                )
                all_valid = False
            else:
                self.add_result(
                    f"{entity}_count",
                    True,
                    f"Found {count} {entity} entities",
                    {"count": count},
                    "INFO"
                )

        return all_valid

    def validate_unique_names(self) -> bool:
        """Validate that entity names are unique."""
        logger.info("Validating unique names...")

        entities = ['knowledge', 'skill', 'ability',
                    'occupation', 'task', 'function']
        all_valid = True

        for entity in entities:
            name_field = 'title' if entity == 'occupation' else 'name'

            # Check for duplicate names
            query = f"""
                SELECT {name_field}, COUNT(*) as count
                FROM {entity}
                GROUP BY {name_field}
                HAVING COUNT(*) > 1
            """
            duplicates = self.execute_query(query)

            if duplicates:
                self.add_result(
                    f"{entity}_unique_names",
                    False,
                    f"Found {len(duplicates)} duplicate {entity} names",
                    {"duplicates": duplicates[:5]},  # Show first 5
                    "ERROR"
                )
                all_valid = False
            else:
                self.add_result(
                    f"{entity}_unique_names",
                    True,
                    f"All {entity} names are unique",
                    {},
                    "INFO"
                )

        return all_valid

    def validate_null_fields(self) -> bool:
        """Validate that critical fields are not null."""
        logger.info("Validating null fields...")

        checks = [
            ('knowledge', 'name', 'CRITICAL'),
            ('knowledge', 'definition', 'WARNING'),
            ('skill', 'name', 'CRITICAL'),
            ('skill', 'definition', 'WARNING'),
            ('ability', 'name', 'CRITICAL'),
            ('ability', 'definition', 'WARNING'),
            ('occupation', 'title', 'CRITICAL'),
            ('task', 'name', 'CRITICAL'),
            ('function', 'name', 'CRITICAL'),
        ]

        all_valid = True

        for table, field, severity in checks:
            query = f"""
                SELECT COUNT(*) 
                FROM {table} 
                WHERE {field} IS NULL OR {field} = ''
            """
            null_count = self.execute_count_query(query)

            if null_count > 0:
                self.add_result(
                    f"{table}_{field}_null",
                    False,
                    f"Found {null_count} {table} records with null/empty {field}",
                    {"null_count": null_count},
                    severity
                )
                if severity in ['CRITICAL', 'ERROR']:
                    all_valid = False
            else:
                self.add_result(
                    f"{table}_{field}_null",
                    True,
                    f"All {table} records have valid {field}",
                    {},
                    "INFO"
                )

        return all_valid

    # ==========================================
    # Dimension Validation
    # ==========================================

    def validate_dimension_tables(self) -> bool:
        """Validate that dimension tables are populated."""
        logger.info("Validating dimension tables...")

        dimensions = [
            'type_dim',
            'level_dim',
            'basis_dim',
            'environment_dim',
            'mode_dim',
            'physicality_dim',
            'cognitive_dim'
        ]

        all_valid = True

        for dim_table in dimensions:
            count = self.execute_count_query(
                f"SELECT COUNT(*) FROM {dim_table}")
            self.stats[f'{dim_table}_count'] = count

            if count == 0:
                self.add_result(
                    f"{dim_table}_populated",
                    False,
                    f"Dimension table {dim_table} is empty",
                    {"count": count},
                    "CRITICAL"
                )
                all_valid = False
            else:
                self.add_result(
                    f"{dim_table}_populated",
                    True,
                    f"Dimension table {dim_table} has {count} entries",
                    {"count": count},
                    "INFO"
                )

        return all_valid

    def validate_dimension_assignments(self) -> bool:
        """Validate that entities have dimension assignments."""
        logger.info("Validating dimension assignments...")

        checks = [
            ('knowledge', ['knowledge_type',
             'knowledge_level', 'knowledge_basis']),
            ('skill', ['skill_type', 'skill_level', 'skill_basis']),
            ('ability', ['ability_type', 'ability_level', 'ability_basis']),
            ('task', ['task_type', 'task_env', 'task_mode']),
            ('function', ['function_env',
             'function_physicality', 'function_cognitive'])
        ]

        all_valid = True

        for entity, junction_tables in checks:
            entity_id_field = f"{entity}_id"

            for junction_table in junction_tables:
                # Count entities with this dimension
                query = f"""
                    SELECT COUNT(DISTINCT {entity_id_field})
                    FROM {junction_table}
                """
                entities_with_dim = self.execute_count_query(query)

                # Count total entities
                total_entities = self.stats.get(f'{entity}_count', 0)

                if total_entities > 0:
                    coverage = (entities_with_dim / total_entities) * 100

                    if coverage < 50:
                        self.add_result(
                            f"{junction_table}_coverage",
                            False,
                            f"Only {coverage:.1f}% of {entity} have {junction_table} assignments",
                            {"coverage": coverage, "count": entities_with_dim,
                                "total": total_entities},
                            "WARNING"
                        )
                    else:
                        self.add_result(
                            f"{junction_table}_coverage",
                            True,
                            f"{coverage:.1f}% of {entity} have {junction_table} assignments",
                            {"coverage": coverage, "count": entities_with_dim,
                                "total": total_entities},
                            "INFO"
                        )

        return all_valid

    # ==========================================
    # Relationship Validation
    # ==========================================

    def validate_occupation_relationships(self) -> bool:
        """Validate occupation relationships."""
        logger.info("Validating occupation relationships...")

        relationship_tables = [
            ('occupation_knowledge', 'knowledge'),
            ('occupation_skill', 'skill'),
            ('occupation_ability', 'ability'),
            ('occupation_task', 'task'),
            ('occupation_function', 'function')
        ]

        all_valid = True

        for rel_table, entity_table in relationship_tables:
            # Count relationships
            count = self.execute_count_query(
                f"SELECT COUNT(*) FROM {rel_table}")
            self.stats[f'{rel_table}_count'] = count

            if count == 0:
                self.add_result(
                    f"{rel_table}_count",
                    False,
                    f"No relationships found in {rel_table}",
                    {"count": count},
                    "ERROR"
                )
                all_valid = False
            else:
                self.add_result(
                    f"{rel_table}_count",
                    True,
                    f"Found {count} relationships in {rel_table}",
                    {"count": count},
                    "INFO"
                )

            # Check for orphaned relationships (referencing non-existent entities)
            entity_id_field = f"{entity_table}_id"
            query = f"""
                SELECT COUNT(*)
                FROM {rel_table} r
                LEFT JOIN {entity_table} e ON r.{entity_id_field} = e.id
                WHERE e.id IS NULL
            """
            orphaned = self.execute_count_query(query)

            if orphaned > 0:
                self.add_result(
                    f"{rel_table}_orphaned",
                    False,
                    f"Found {orphaned} orphaned relationships in {rel_table}",
                    {"orphaned_count": orphaned},
                    "ERROR"
                )
                all_valid = False

        return all_valid

    def validate_relationship_reciprocity(self) -> bool:
        """Validate that relationships make sense."""
        logger.info("Validating relationship reciprocity...")

        # Check if occupations have a reasonable distribution of KSAs
        query = """
            SELECT 
                o.title,
                COUNT(DISTINCT ok.knowledge_id) as knowledge_count,
                COUNT(DISTINCT os.skill_id) as skill_count,
                COUNT(DISTINCT oa.ability_id) as ability_count
            FROM occupation o
            LEFT JOIN occupation_knowledge ok ON o.id = ok.occupation_id
            LEFT JOIN occupation_skill os ON o.id = os.occupation_id
            LEFT JOIN occupation_ability oa ON o.id = oa.occupation_id
            GROUP BY o.id, o.title
            HAVING COUNT(DISTINCT ok.knowledge_id) = 0 
                OR COUNT(DISTINCT os.skill_id) = 0 
                OR COUNT(DISTINCT oa.ability_id) = 0
        """
        incomplete_occupations = self.execute_query(query)

        if incomplete_occupations:
            self.add_result(
                "occupation_incomplete_ksa",
                False,
                f"Found {len(incomplete_occupations)} occupations missing K, S, or A",
                {"sample": incomplete_occupations[:5]},
                "WARNING"
            )
        else:
            self.add_result(
                "occupation_complete_ksa",
                True,
                "All occupations have knowledge, skills, and abilities",
                {},
                "INFO"
            )

        return True

    # ==========================================
    # Data Quality Validation
    # ==========================================

    def validate_definition_quality(self) -> bool:
        """Validate definition field quality."""
        logger.info("Validating definition quality...")

        entities = ['knowledge', 'skill', 'ability', 'task', 'function']

        for entity in entities:
            # Check for very short definitions (likely poor quality)
            query = f"""
                SELECT COUNT(*)
                FROM {entity}
                WHERE LENGTH(definition) < 20 AND definition IS NOT NULL
            """
            short_def_count = self.execute_count_query(query)

            total = self.stats.get(f'{entity}_count', 0)
            if total > 0 and short_def_count > 0:
                percentage = (short_def_count / total) * 100
                self.add_result(
                    f"{entity}_short_definitions",
                    percentage < 10,
                    f"{percentage:.1f}% of {entity} have very short definitions (<20 chars)",
                    {"count": short_def_count, "total": total},
                    "WARNING" if percentage > 10 else "INFO"
                )

        return True

    def validate_source_references(self) -> bool:
        """Validate that source references are present."""
        logger.info("Validating source references...")

        entities = ['knowledge', 'skill', 'ability']

        for entity in entities:
            query = f"""
                SELECT COUNT(*)
                FROM {entity}
                WHERE source_ref IS NULL OR source_ref = ''
            """
            missing_ref = self.execute_count_query(query)

            total = self.stats.get(f'{entity}_count', 0)
            if total > 0:
                percentage = (missing_ref / total) * 100
                self.add_result(
                    f"{entity}_source_ref",
                    percentage < 5,
                    f"{percentage:.1f}% of {entity} missing source references",
                    {"count": missing_ref, "total": total},
                    "WARNING" if percentage > 5 else "INFO"
                )

        return True

    # ==========================================
    # Statistical Analysis
    # ==========================================

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive database statistics."""
        logger.info("Generating database statistics...")

        stats = {}

        # Entity counts (already collected)
        stats['entity_counts'] = {
            k: v for k, v in self.stats.items() if k.endswith('_count')
        }

        # Calculate averages with separate simple queries (much faster)
        occupation_count = self.stats.get('occupation_count', 1)

        if occupation_count > 0:
            stats['avg_relationships_per_occupation'] = {
                'avg_knowledge': self.execute_count_query(
                    "SELECT COUNT(*) FROM occupation_knowledge"
                ) / occupation_count,
                'avg_skills': self.execute_count_query(
                    "SELECT COUNT(*) FROM occupation_skill"
                ) / occupation_count,
                'avg_abilities': self.execute_count_query(
                    "SELECT COUNT(*) FROM occupation_ability"
                ) / occupation_count,
                'avg_tasks': self.execute_count_query(
                    "SELECT COUNT(*) FROM occupation_task"
                ) / occupation_count,
            }

        # Dimension distribution (keep these, they're fast)
        for dim_type in ['type_dim', 'level_dim', 'basis_dim']:
            query = f"""
                SELECT name, COUNT(*) as usage_count
                FROM {dim_type}
                GROUP BY name
                ORDER BY usage_count DESC
                LIMIT 10
            """
            stats[f'{dim_type}_distribution'] = self.execute_query(query)

        return stats

    # ==========================================
    # Report Generation
    # ==========================================

    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        report_lines = [
            "=" * 80,
            "KSAMDS DATABASE VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]

        # Summary counts
        total_checks = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.passed)
        failed = total_checks - passed

        critical = sum(
            1 for r in self.validation_results if r.severity == "CRITICAL" and not r.passed)
        errors = sum(
            1 for r in self.validation_results if r.severity == "ERROR" and not r.passed)
        warnings = sum(
            1 for r in self.validation_results if r.severity == "WARNING" and not r.passed)

        report_lines.extend([
            "SUMMARY",
            "-" * 80,
            f"Total Checks: {total_checks}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"  - Critical: {critical}",
            f"  - Errors: {errors}",
            f"  - Warnings: {warnings}",
            ""
        ])

        # Entity counts
        report_lines.extend([
            "ENTITY COUNTS",
            "-" * 80
        ])
        for key, value in self.stats.items():
            if key.endswith('_count') and not key.endswith('_relationships_count'):
                entity_name = key.replace(
                    '_count', '').replace('_', ' ').title()
                report_lines.append(f"{entity_name}: {value:,}")
        report_lines.append("")

        # Group results by severity
        for severity in ['CRITICAL', 'ERROR', 'WARNING', 'INFO']:
            severity_results = [
                r for r in self.validation_results if r.severity == severity]
            if severity_results:
                report_lines.extend([
                    f"{severity} CHECKS",
                    "-" * 80
                ])
                for result in severity_results:
                    status = "✓" if result.passed else "✗"
                    report_lines.append(
                        f"{status} {result.check_name}: {result.message}")
                report_lines.append("")

        return "\n".join(report_lines)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate machine-readable JSON report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.validation_results),
                "passed": sum(1 for r in self.validation_results if r.passed),
                "failed": sum(1 for r in self.validation_results if not r.passed),
                "critical": sum(1 for r in self.validation_results if r.severity == "CRITICAL" and not r.passed),
                "errors": sum(1 for r in self.validation_results if r.severity == "ERROR" and not r.passed),
                "warnings": sum(1 for r in self.validation_results if r.severity == "WARNING" and not r.passed),
            },
            "statistics": self.stats,
            "validation_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in self.validation_results
            ]
        }

    def save_reports(self):
        """Save validation reports to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Text report
        text_report = self.generate_text_report()
        text_path = self.output_dir / f"validation_report_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)
        logger.info(f"Text report saved to {text_path}")

        # JSON report
        json_report = self.generate_json_report()
        json_path = self.output_dir / f"validation_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")

        # Also save latest report (overwrite)
        latest_text = self.output_dir / "validation_report_latest.txt"
        with open(latest_text, 'w') as f:
            f.write(text_report)

        latest_json = self.output_dir / "validation_report_latest.json"
        with open(latest_json, 'w') as f:
            json.dump(json_report, f, indent=2)

    # ==========================================
    # Main Validation Pipeline
    # ==========================================

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting comprehensive validation...")

        all_valid = True

        # Core entity validation
        all_valid &= self.validate_entity_counts()
        all_valid &= self.validate_unique_names()
        all_valid &= self.validate_null_fields()

        # Dimension validation
        all_valid &= self.validate_dimension_tables()
        all_valid &= self.validate_dimension_assignments()

        # Relationship validation
        all_valid &= self.validate_occupation_relationships()
        all_valid &= self.validate_relationship_reciprocity()

        # Data quality validation
        all_valid &= self.validate_definition_quality()
        all_valid &= self.validate_source_references()

        # Generate statistics
        self.stats.update(self.generate_statistics())

        # Save reports
        self.save_reports()

        logger.info("Validation completed!")
        return all_valid

    @classmethod
    def from_environment(cls, output_dir: str = "data/validation") -> 'ONetValidator':
        """Create validator instance from environment variables."""
        db_config = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'ksamds'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '8086'),
            schema=os.getenv('DB_SCHEMA', 'ksamds')
        )
        return cls(db_config, output_dir)


def main():
    """Main function to run the validator."""
    validator = ONetValidator.from_environment()

    success = validator.validate_all()

    # Print summary to console
    print("\n" + "=" * 80)
    if success:
        print("✓ VALIDATION PASSED - All critical checks passed")
    else:
        print("✗ VALIDATION FAILED - Some checks did not pass")
    print("=" * 80)

    # Print summary statistics
    print("\nValidation Summary:")
    print(f"Total Checks: {len(validator.validation_results)}")
    print(
        f"Passed: {sum(1 for r in validator.validation_results if r.passed)}")
    print(
        f"Failed: {sum(1 for r in validator.validation_results if not r.passed)}")

    critical_failures = sum(1 for r in validator.validation_results
                            if r.severity == "CRITICAL" and not r.passed)
    if critical_failures > 0:
        print(f"\n⚠️  CRITICAL FAILURES: {critical_failures}")
        print("Review the validation report for details.")

    print(f"\nDetailed reports saved to: {validator.output_dir}")
    print("- validation_report_latest.txt")
    print("- validation_report_latest.json")

    # Exit with error code if validation failed
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
