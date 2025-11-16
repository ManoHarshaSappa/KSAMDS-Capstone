"""
Skills Framework Data Extractor for KSAMDS Project

This module handles downloading and extracting Skills Framework dataset from
Singapore's SkillsFuture portal. Downloads the Excel file with multiple sheets,
extracts relevant data, and loads them into pandas DataFrames for further processing.

Data Source: https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks
File: Skills-Framework-Dataset-Q3-2025.xlsx
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkillsFrameworkExtractor:
    """Extract Skills Framework dataset and convert to pandas DataFrames."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Skills Framework extractor.

        Args:
            data_dir: Base directory for data storage (relative to project root)
        """
        # Get project root directory (3 levels up from skillsframework_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / data_dir

        # Setup directory structure
        self.raw_dir = self.data_dir / "skillsframework" / "archive" / "raw"
        self.processed_dir = self.data_dir / \
            "skillsframework" / "archive" / "intermediate"
        self.relationships_dir = self.data_dir / \
            "skillsframework" / "archive" / "relationships"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.relationships_dir.mkdir(parents=True, exist_ok=True)

        # Skills Framework file info
        # Direct download URL for the Skills Framework dataset
        self.download_url = "https://file.go.gov.sg/jobsandskills-skillsfuture-skills-framework-dataset.xlsx"
        self.excel_path = self.raw_dir / "skills_framework_q3_2025.xlsx"

        # Sheet names and required columns
        self.sheet_config = {
            'job_role_description': {
                'sheet_name': 'Job Role_Description',
                'required_columns': ['Job Role', 'Job Role Description'],
                'rename_map': {
                    'Job Role': 'job_role',
                    'Job Role Description': 'job_role_description'
                }
            },
            'job_role_cwf_kt': {
                'sheet_name': 'Job Role_CWF_KT',
                'required_columns': ['Job Role', 'Critical Work Function', 'Key Tasks'],
                'rename_map': {
                    'Job Role': 'job_role',
                    'Critical Work Function': 'critical_work_function',
                    'Key Tasks': 'key_tasks'
                }
            },
            'job_role_tcs_ccs': {
                'sheet_name': 'Job Role_TCS_CCS',
                'required_columns': ['Job Role', 'skill_sector_title', 'Proficiency Level', 'TSC_CCS Code'],
                'rename_map': {
                    'Job Role': 'job_role',
                    'skill_sector_title': 'skill_sector_title',
                    'Proficiency Level': 'proficiency_level',
                    'TSC_CCS Code': 'tsc_ccs_code'
                }
            },
            'tsc_ccs_knowledge_ability': {
                'sheet_name': 'TSC_CCS_K&A',
                'required_columns': ['TSC_CCS Code', 'Knowledge / Ability Items', 'Knowledge / Ability Classification'],
                'rename_map': {
                    'TSC_CCS Code': 'tsc_ccs_code',
                    'Knowledge / Ability Items': 'knowledge_ability_items',
                    'Knowledge / Ability Classification': 'knowledge_ability_classification'
                }
            }
        }

        self.dataframes: Dict[str, pd.DataFrame] = {}

    def download_excel_file(self, url: Optional[str] = None) -> bool:
        """
        Download the Skills Framework Excel file.

        Args:
            url: Direct download URL for the Excel file (uses instance URL if not provided)

        Returns:
            bool: True if file exists (downloaded or already present), False otherwise
        """
        if self.excel_path.exists():
            logger.info(
                f"Skills Framework file already exists at {self.excel_path}")
            return True

        # Use instance URL if none provided
        if url is None:
            url = self.download_url

        try:
            logger.info("Downloading Skills Framework dataset...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(self.excel_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            logger.info(f"Download progress: {percent:.1f}%")

            logger.info(
                f"Successfully downloaded Skills Framework dataset to {self.excel_path}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download Skills Framework dataset: {e}")
            logger.warning(
                "=" * 70 +
                "\nAutomatic download failed. Please manually download the file:\n"
                "1. Visit: https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks\n"
                "2. Click on 'Skills-Framework-Dataset-Q3-2025' button to download\n"
                f"3. Save the file to: {self.excel_path}\n"
                "=" * 70
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False

    def load_excel_sheet(self, sheet_name: str, required_columns: list) -> Optional[pd.DataFrame]:
        """
        Load a specific sheet from the Skills Framework Excel file.

        Args:
            sheet_name: Name of the sheet to load
            required_columns: List of column names that must be present

        Returns:
            pd.DataFrame or None if sheet not found/load failed
        """
        if not self.excel_path.exists():
            logger.error(
                f"Skills Framework Excel file not found at {self.excel_path}. "
                "Run download_excel_file() first or place the file manually."
            )
            return None

        try:
            logger.info(f"Loading sheet '{sheet_name}'...")
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)

            # Verify required columns exist
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(
                    f"Sheet '{sheet_name}' is missing required columns: {missing_columns}\n"
                    f"Available columns: {list(df.columns)}"
                )
                return None

            # Basic data cleaning
            df = self._clean_dataframe(df)

            logger.info(
                f"Successfully loaded sheet '{sheet_name}' with {len(df)} rows")
            return df

        except ValueError as e:
            logger.error(f"Sheet '{sheet_name}' not found in Excel file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load sheet '{sheet_name}': {e}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic cleaning to a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Strip whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Replace empty strings with NaN
        df = df.replace('', pd.NA)
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        return df

    def _pluralize_occupation(self, occupation: str) -> str:
        """
        Convert occupation name to plural form.

        Args:
            occupation: Singular occupation name

        Returns:
            Pluralized occupation name
        """
        if pd.isna(occupation):
            return occupation

        occupation = str(occupation).strip()

        # Common exceptions and irregular plurals
        exceptions = {
            'Chief': 'Chiefs',
            'Staff': 'Staff',
            'Personnel': 'Personnel',
            'Police': 'Police',
            'Clergy': 'Clergy'
        }

        # Check if it's already plural or an exception
        if occupation in exceptions:
            return exceptions[occupation]

        # Check if already ends with 's' (likely already plural)
        if occupation.endswith('s') or occupation.endswith('S'):
            return occupation

        # Apply standard pluralization rules
        if occupation.endswith(('ch', 'sh', 'ss', 'x', 'z', 'o')):
            return occupation + 'es'
        elif occupation.endswith('y') and len(occupation) > 1 and occupation[-2] not in 'aeiou':
            return occupation[:-1] + 'ies'
        elif occupation.endswith('f'):
            return occupation[:-1] + 'ves'
        elif occupation.endswith('fe'):
            return occupation[:-2] + 'ves'
        else:
            return occupation + 's'

    def _select_and_rename_columns(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Select required columns and rename them according to the config.

        Args:
            df: Input DataFrame
            config: Configuration dictionary with 'required_columns' and 'rename_map'

        Returns:
            pd.DataFrame: DataFrame with selected and renamed columns
        """
        # Select only required columns
        df_selected = df[config['required_columns']].copy()

        # Rename columns
        df_renamed = df_selected.rename(columns=config['rename_map'])

        return df_renamed

    def load_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all configured sheets from the Skills Framework Excel file.

        Returns:
            dict: Dictionary of DataFrames keyed by sheet identifier
        """
        logger.info("Loading all Skills Framework sheets...")

        temp_dataframes = {}

        for key, config in self.sheet_config.items():
            df = self.load_excel_sheet(
                sheet_name=config['sheet_name'],
                required_columns=config['required_columns']
            )

            if df is not None:
                # Select and rename columns
                df_processed = self._select_and_rename_columns(df, config)
                temp_dataframes[key] = df_processed
                logger.info(
                    f"Processed '{key}': {len(df_processed)} rows, "
                    f"{len(df_processed.columns)} columns"
                )
            else:
                logger.warning(
                    f"Failed to load sheet: {key} ({config['sheet_name']})")

        # Store temporarily
        self.dataframes = temp_dataframes

        # Create final 6 DataFrames with correct structure
        logger.info("Creating final DataFrames...")
        self._create_final_dataframes()

        logger.info(
            f"Successfully created {len(self.dataframes)} final datasets"
        )

        return self.dataframes

    def _create_final_dataframes(self):
        """
        Create the final 6 DataFrames with the specified structure:
        1. occupations.csv: occupation, description
        2. knowledge.csv: occupation, knowledge, level
        3. skills.csv: occupation, skill, level
        4. abilities.csv: occupation, ability, level
        5. functions.csv: occupation, function
        6. tasks.csv: occupation, task

        Also creates relationship files:
        - knowledge_skill.csv: skill, knowledge
        - skill_ability.csv: skill, ability
        """
        final_dfs = {}
        relationship_dfs = {}

        # 1. occupations.csv
        if 'job_role_description' in self.dataframes:
            occupations = self.dataframes['job_role_description'].copy()
            occupations = occupations.rename(columns={
                'job_role': 'occupation',
                'job_role_description': 'description'
            })
            # Pluralize occupation names
            occupations['occupation'] = occupations['occupation'].apply(
                self._pluralize_occupation)
            final_dfs['occupations'] = occupations
            logger.info(f"Created occupations: {len(occupations)} rows")

        # 2. skills.csv (from job_role_tcs_ccs)
        if 'job_role_tcs_ccs' in self.dataframes:
            skills = self.dataframes['job_role_tcs_ccs'][[
                'job_role', 'skill_sector_title', 'proficiency_level', 'tsc_ccs_code']].copy()
            skills = skills.rename(columns={
                'job_role': 'occupation',
                'skill_sector_title': 'skill',
                'proficiency_level': 'level'
            })
            # Pluralize occupation names
            skills['occupation'] = skills['occupation'].apply(
                self._pluralize_occupation)
            skills_final = skills[['occupation',
                                   'skill', 'level']].drop_duplicates()
            final_dfs['skills'] = skills_final
            logger.info(f"Created skills: {len(skills_final)} rows")

        # 3. knowledge.csv and 4. abilities.csv (from tsc_ccs_knowledge_ability linked to job_role_tcs_ccs)
        if 'tsc_ccs_knowledge_ability' in self.dataframes and 'job_role_tcs_ccs' in self.dataframes:
            ka_df = self.dataframes['tsc_ccs_knowledge_ability'].copy()
            job_role_tcs = self.dataframes['job_role_tcs_ccs'].copy()

            # Clean and normalize the classification column
            ka_df['knowledge_ability_classification'] = ka_df['knowledge_ability_classification'].str.strip(
            ).str.lower()

            # Separate knowledge and abilities
            knowledge_df = ka_df[ka_df['knowledge_ability_classification'] == 'knowledge'][[
                'tsc_ccs_code', 'knowledge_ability_items']].copy()
            ability_df = ka_df[ka_df['knowledge_ability_classification'] == 'ability'][[
                'tsc_ccs_code', 'knowledge_ability_items']].copy()

            logger.info(
                f"Found {len(knowledge_df)} knowledge items and {len(ability_df)} ability items")

            # Link to job roles and skills via tsc_ccs_code (keep tsc_ccs_code, skill name, and level)
            knowledge = job_role_tcs[['job_role', 'tsc_ccs_code', 'skill_sector_title', 'proficiency_level']].merge(
                knowledge_df,
                on='tsc_ccs_code',
                how='inner'
            )

            abilities = job_role_tcs[['job_role', 'tsc_ccs_code', 'skill_sector_title', 'proficiency_level']].merge(
                ability_df,
                on='tsc_ccs_code',
                how='inner'
            )

            # Create relationship files (skill to knowledge/ability) - save to intermediate folder
            knowledge_skill_rel = knowledge[[
                'skill_sector_title', 'knowledge_ability_items']].drop_duplicates()
            knowledge_skill_rel = knowledge_skill_rel.rename(columns={
                'skill_sector_title': 'skill',
                'knowledge_ability_items': 'knowledge'
            })
            final_dfs['knowledge_skill'] = knowledge_skill_rel
            logger.info(
                f"Created knowledge_skill relationship: {len(knowledge_skill_rel)} rows")

            skill_ability_rel = abilities[[
                'skill_sector_title', 'knowledge_ability_items']].drop_duplicates()
            skill_ability_rel = skill_ability_rel.rename(columns={
                'skill_sector_title': 'skill',
                'knowledge_ability_items': 'ability'
            })
            final_dfs['skill_ability'] = skill_ability_rel
            logger.info(
                f"Created skill_ability relationship: {len(skill_ability_rel)} rows")

            # Create final knowledge and abilities files with level
            knowledge_final = knowledge[[
                'job_role', 'knowledge_ability_items', 'proficiency_level']].drop_duplicates()
            knowledge_final = knowledge_final.rename(columns={
                'job_role': 'occupation',
                'knowledge_ability_items': 'knowledge',
                'proficiency_level': 'level'
            })
            # Pluralize occupation names
            knowledge_final['occupation'] = knowledge_final['occupation'].apply(
                self._pluralize_occupation)

            abilities_final = abilities[[
                'job_role', 'knowledge_ability_items', 'proficiency_level']].drop_duplicates()
            abilities_final = abilities_final.rename(columns={
                'job_role': 'occupation',
                'knowledge_ability_items': 'ability',
                'proficiency_level': 'level'
            })
            # Pluralize occupation names
            abilities_final['occupation'] = abilities_final['occupation'].apply(
                self._pluralize_occupation)

            final_dfs['knowledge'] = knowledge_final
            final_dfs['abilities'] = abilities_final
            logger.info(f"Created knowledge: {len(knowledge_final)} rows")
            logger.info(f"Created abilities: {len(abilities_final)} rows")

        # 5. functions.csv and 6. tasks.csv (from job_role_cwf_kt)
        if 'job_role_cwf_kt' in self.dataframes:
            cwf_kt = self.dataframes['job_role_cwf_kt'].copy()

            # Functions
            functions = cwf_kt[['job_role', 'critical_work_function']].copy()
            functions = functions.rename(columns={
                'job_role': 'occupation',
                'critical_work_function': 'function'
            })
            # Pluralize occupation names
            functions['occupation'] = functions['occupation'].apply(
                self._pluralize_occupation)
            functions = functions.drop_duplicates()
            final_dfs['functions'] = functions
            logger.info(f"Created functions: {len(functions)} rows")

            # Tasks
            tasks = cwf_kt[['job_role', 'key_tasks']].copy()
            tasks = tasks.rename(columns={
                'job_role': 'occupation',
                'key_tasks': 'task'
            })
            # Pluralize occupation names
            tasks['occupation'] = tasks['occupation'].apply(
                self._pluralize_occupation)
            tasks = tasks.drop_duplicates()
            final_dfs['tasks'] = tasks
            logger.info(f"Created tasks: {len(tasks)} rows")

        # Replace the dataframes dictionary with final DataFrames only
        self.dataframes = final_dfs

    def save_dataframes_to_csv(self, output_dir: Optional[str] = None) -> bool:
        """
        Save all loaded DataFrames to CSV files.

        Args:
            output_dir: Directory to save CSV files (defaults to processed dir)

        Returns:
            bool: True if successful
        """
        if not self.dataframes:
            logger.error("No DataFrames loaded. Run load_all_sheets() first.")
            return False

        if output_dir is None:
            output_dir = self.processed_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save all dataframes to intermediate directory
            for key, df in self.dataframes.items():
                csv_path = output_dir / f"{key}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {csv_path} ({len(df)} rows)")

            logger.info(f"All CSV files saved to {output_dir}")

            return True

        except Exception as e:
            logger.error(f"Failed to save DataFrames: {e}")
            return False

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all loaded DataFrames.

        Returns:
            dict: Summary information for each DataFrame
        """
        summary = {}

        for key, df in self.dataframes.items():
            summary[key] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'missing_data': df.isnull().sum().to_dict(),
            }

            # Add specific insights based on data type
            if key == 'occupations':
                summary[key]['unique_occupations'] = df['occupation'].nunique()

            elif key == 'knowledge':
                summary[key]['unique_occupations'] = df['occupation'].nunique()
                summary[key]['unique_knowledge_items'] = df['knowledge'].nunique()

            elif key == 'skills':
                summary[key]['unique_occupations'] = df['occupation'].nunique()
                summary[key]['unique_skills'] = df['skill'].nunique()
                if 'level' in df.columns:
                    summary[key]['proficiency_levels'] = sorted(
                        df['level'].dropna().unique().tolist())

            elif key == 'abilities':
                summary[key]['unique_occupations'] = df['occupation'].nunique()
                summary[key]['unique_ability_items'] = df['ability'].nunique()

            elif key == 'functions':
                summary[key]['unique_occupations'] = df['occupation'].nunique()
                summary[key]['unique_functions'] = df['function'].nunique()

            elif key == 'tasks':
                summary[key]['unique_occupations'] = df['occupation'].nunique()
                summary[key]['unique_tasks'] = df['task'].nunique()

        return summary

    def validate_data_integrity(self) -> Dict[str, list]:
        """
        Validate data integrity across sheets.

        Returns:
            dict: Dictionary of validation issues found
        """
        issues = {
            'data_quality': []
        }

        if not self.dataframes:
            issues['data_quality'].append("No data loaded")
            return issues

        # Check for null values in key columns
        for key, df in self.dataframes.items():
            for col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    if null_pct > 10:  # Report if more than 10% missing
                        issues['data_quality'].append(
                            f"{key}.{col}: {null_pct:.1f}% missing ({null_count}/{len(df)} rows)"
                        )

        return issues

    def extract_all(self, download_url: Optional[str] = None) -> bool:
        """
        Complete extraction pipeline: download/check, load, and save all data.

        Args:
            download_url: Optional direct download URL (uses instance URL if not provided)

        Returns:
            bool: True if successful
        """
        logger.info("Starting Skills Framework data extraction pipeline...")

        # Step 1: Download or check for file (uses instance URL by default)
        if not self.download_excel_file(download_url):
            if not self.excel_path.exists():
                logger.error(
                    "Excel file not available. Please download manually.")
                return False

        # Step 2: Load all sheets
        dataframes = self.load_all_sheets()
        if not dataframes:
            logger.error("No sheets were successfully loaded")
            return False

        # Step 3: Validate data integrity
        logger.info("Validating data integrity...")
        issues = self.validate_data_integrity()

        if any(issues.values()):
            logger.warning("Data integrity issues found:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    logger.warning(
                        f"  {issue_type.replace('_', ' ').title()}:")
                    for issue in issue_list:
                        logger.warning(f"    - {issue}")
        else:
            logger.info("Data integrity validation passed!")

        # Step 4: Save to CSV
        if not self.save_dataframes_to_csv():
            logger.warning(
                "Failed to save CSV files, but extraction was successful")

        logger.info(
            "Skills Framework extraction pipeline completed successfully!")
        return True


def main():
    """Main function to run the Skills Framework extractor."""
    extractor = SkillsFrameworkExtractor()

    # Run complete extraction (will attempt automatic download from the configured URL)
    success = extractor.extract_all()

    if success:
        # Log summary
        summary = extractor.get_data_summary()
        logger.info("=" * 70)
        logger.info("SKILLS FRAMEWORK DATA SUMMARY")
        logger.info("=" * 70)

        for key, info in summary.items():
            logger.info(f"\n{key.upper()}:")
            logger.info(f"  Rows: {info['rows']}")
            logger.info(f"  Columns: {', '.join(info['column_names'])}")

            # Log unique value counts
            if 'unique_occupations' in info:
                logger.info(
                    f"  Unique occupations: {info['unique_occupations']}")
            if 'unique_knowledge_items' in info:
                logger.info(
                    f"  Unique knowledge items: {info['unique_knowledge_items']}")
            if 'unique_skills' in info:
                logger.info(f"  Unique skills: {info['unique_skills']}")
            if 'unique_ability_items' in info:
                logger.info(
                    f"  Unique abilities: {info['unique_ability_items']}")
            if 'unique_functions' in info:
                logger.info(f"  Unique functions: {info['unique_functions']}")
            if 'unique_tasks' in info:
                logger.info(f"  Unique tasks: {info['unique_tasks']}")
            if 'proficiency_levels' in info:
                logger.info(
                    f"  Proficiency levels: {info['proficiency_levels']}")

            # Log missing data
            missing = {k: v for k, v in info['missing_data'].items() if v > 0}
            if missing:
                logger.info(f"  Missing data: {missing}")

        logger.info("\n" + "=" * 70)
        logger.info(f"Raw data location: {extractor.raw_dir}")
        logger.info(f"CSV files saved to: {extractor.processed_dir}")
        logger.info("=" * 70)
    else:
        logger.error("Extraction failed. Check logs for details.")


if __name__ == "__main__":
    main()
