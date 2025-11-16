"""
O*NET Data Extractor for KSAMDS Project

This module handles downloading and extracting O*NET database files.
Downloads the latest O*NET database, extracts relevant Excel files,
and loads them into pandas DataFrames for further processing.

UPDATED: Now extracts level (LV) scale values from O*NET data for Knowledge, Skills, and Abilities
"""

import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, Set
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONetExtractor:
    """Extract O*NET database files and convert to pandas DataFrames."""

    def __init__(self, data_dir: str = "data/onet"):
        """
        Initialize the O*NET extractor.

        Args:
            data_dir: Base directory for data storage (relative to project root)
        """
        # Get project root directory (3 levels up from onet_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / data_dir

        # Setup directory structure
        self.raw_dir = self.data_dir / "archive" / "raw"
        self.temp_extract_dir = project_root / "data" / "temp" / "extracted"
        self.processed_dir = self.data_dir / "archive" / "intermediate"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # O*NET database URL (check for latest version)
        self.onet_url = "https://www.onetcenter.org/dl_files/database/db_30_0_excel.zip"
        self.zip_path = self.raw_dir / "onet_database.zip"

        # Key O*NET files we need for KSAMDS
        self.key_files = {
            'knowledge': 'Knowledge.xlsx',
            'skills': 'Skills.xlsx',
            'abilities': 'Abilities.xlsx',
            'occupation_data': 'Occupation Data.xlsx',
            'task_ratings': 'Task Ratings.xlsx',
            'task_statements': 'Task Statements.xlsx',
            'work_activities': 'Work Activities.xlsx',
            'content_model_reference': 'Content Model Reference.xlsx',
            'scales_reference': 'Scales Reference.xlsx',
            'job_zones': 'Job Zones.xlsx',
            'education_training_experience': 'Education, Training, and Experience.xlsx',
            'work_context': 'Work Context.xlsx',
            'interests': 'Interests.xlsx',
            'work_values': 'Work Values.xlsx'
        }

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self._cleanup_on_extract = True  # Clean up Excel files after CSV conversion

    def download_onet_database(self, force_download: bool = False) -> bool:
        """
        Download the latest O*NET database zip file.

        Args:
            force_download: Whether to re-download if file already exists

        Returns:
            bool: True if successful, False otherwise
        """
        if self.zip_path.exists() and not force_download:
            logger.info(f"O*NET database already exists at {self.zip_path}")
            return True

        try:
            logger.info("Downloading O*NET database...")
            response = requests.get(self.onet_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(self.zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            logger.info(
                                f"Download progress: {percent:.1f}%")
            logger.info(
                f"Successfully downloaded O*NET database to {self.zip_path}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download O*NET database: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False

    def extract_database(self) -> bool:
        """
        Extract the O*NET database zip file to a temporary directory.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.zip_path.exists():
            logger.error(
                "O*NET database zip file not found. Run download_onet_database() first.")
            return False

        try:
            logger.info("Extracting O*NET database...")

            # Use temp directory for extraction
            if self.temp_extract_dir.exists():
                shutil.rmtree(self.temp_extract_dir)

            self.temp_extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_extract_dir)

            logger.info(
                f"Successfully extracted O*NET database to {self.temp_extract_dir}")
            return True

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to extract database: {e}")
            return False

    def load_excel_file(self, file_name: str, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load a specific Excel file from the extracted O*NET database.

        Args:
            file_name: Name of the Excel file to load
            sheet_name: Specific sheet to load (None for first sheet)

        Returns:
            pd.DataFrame or None if file not found/load failed
        """
        file_path = self.temp_extract_dir / "db_30_0_excel" / file_name

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            logger.info(f"Loading {file_name}...")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Basic data cleaning
            if isinstance(df, dict):  # Multiple sheets
                for sheet, data in df.items():
                    df[sheet] = self._clean_dataframe(data)
            else:  # Single sheet
                df = self._clean_dataframe(df)

            logger.info(
                f"Successfully loaded {file_name} with {len(df) if not isinstance(df, dict) else sum(len(d) for d in df.values())} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_name}: {e}")
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
        df[string_columns] = df[string_columns].apply(
            lambda x: x.str.strip() if x.dtype == 'object' else x)

        # Replace empty strings with NaN
        df = df.replace('', pd.NA)

        # Remove completely empty rows
        df = df.dropna(how='all')

        return df

    def _extract_scale_value(self, df: pd.DataFrame, scale_name: str) -> pd.Series:
        """
        Extract a specific scale value from the Data Value column.

        O*NET stores multiple scale values in the 'Data Value' column, with the
        'Scale ID' column indicating which scale the row represents.

        Args:
            df: DataFrame with 'Scale ID' and 'Data Value' columns
            scale_name: Scale to extract (e.g., 'LV' for Level, 'IM' for Importance)

        Returns:
            pd.Series: Series indexed by O*NET-SOC Code and Element ID with scale values
        """
        if 'Scale ID' not in df.columns or 'Data Value' not in df.columns:
            logger.warning(f"Missing required columns for scale extraction")
            return pd.Series(dtype=float)

        # Filter for the specific scale
        scale_df = df[df['Scale ID'] == scale_name].copy()

        if scale_df.empty:
            logger.warning(f"No data found for scale '{scale_name}'")
            return pd.Series(dtype=float)

        # Convert Data Value to numeric
        scale_df['Data Value'] = pd.to_numeric(
            scale_df['Data Value'], errors='coerce')

        # Create multi-index series: (O*NET-SOC Code, Element ID) -> Data Value
        if 'O*NET-SOC Code' in scale_df.columns and 'Element ID' in scale_df.columns:
            scale_df = scale_df.set_index(['O*NET-SOC Code', 'Element ID'])
            return scale_df['Data Value']
        else:
            logger.warning(f"Missing index columns for scale extraction")
            return pd.Series(dtype=float)

    def _enrich_with_levels(self, df, entity_type: str):
        """
        Enrich Knowledge/Skills/Abilities DataFrame with level (LV) scale values.

        O*NET stores level data separately from the element names. This function
        extracts the LV (Level) scale values and joins them to the main data.

        Args:
            df: DataFrame or dict of DataFrames from Knowledge.xlsx, Skills.xlsx, or Abilities.xlsx
            entity_type: Type of entity ('knowledge', 'skills', 'abilities')

        Returns:
            DataFrame or dict: Enriched data with 'Data Value LV' column
        """
        # Handle multi-sheet Excel files (returned as dict)
        if isinstance(df, dict):
            logger.info(f"Processing multi-sheet file for {entity_type}")
            # For multi-sheet files, find the main data sheet and enrich only that
            enriched_dict = {}
            main_sheet_found = False

            for sheet_name, sheet_df in df.items():
                # Look for the sheet that contains actual data (has required columns)
                if isinstance(sheet_df, pd.DataFrame) and \
                   'O*NET-SOC Code' in sheet_df.columns and \
                   'Element ID' in sheet_df.columns and \
                   'Scale ID' in sheet_df.columns:
                    # This is the main data sheet, enrich it
                    enriched_dict[sheet_name] = self._enrich_single_dataframe(
                        sheet_df, entity_type)
                    main_sheet_found = True
                else:
                    # Keep other sheets as-is
                    enriched_dict[sheet_name] = sheet_df

            if not main_sheet_found:
                logger.warning(
                    f"No main data sheet found for {entity_type}, returning original dict")

            return enriched_dict

        # Handle single DataFrame
        elif isinstance(df, pd.DataFrame):
            return self._enrich_single_dataframe(df, entity_type)

        # Handle None or other types
        else:
            logger.warning(
                f"Unexpected data type for {entity_type}: {type(df)}")
            return df

    def _enrich_single_dataframe(self, df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """
        Enrich a single DataFrame with level data.

        Args:
            df: Single DataFrame to enrich
            entity_type: Type of entity

        Returns:
            Enriched DataFrame
        """
        if df is None or df.empty:
            return df

        # Check if this DataFrame has the required columns
        if 'O*NET-SOC Code' not in df.columns or 'Element ID' not in df.columns or 'Scale ID' not in df.columns:
            logger.warning(
                f"DataFrame for {entity_type} missing required columns, skipping enrichment")
            if 'Data Value LV' not in df.columns:
                df['Data Value LV'] = None
            return df

        # Extract LV scale values
        lv_values = self._extract_scale_value(df, 'LV')

        if lv_values.empty:
            logger.warning(f"No level (LV) data found for {entity_type}")
            # Add empty LV column for consistency
            df['Data Value LV'] = None
            return df

        # Create temporary index for joining
        df = df.set_index(['O*NET-SOC Code', 'Element ID'])
        df['Data Value LV'] = lv_values
        df = df.reset_index()

        logger.info(
            f"Enriched {entity_type} with {lv_values.notna().sum()} level values")

        return df

    def load_all_key_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all key O*NET files needed for KSAMDS.

        UPDATED: Now enriches Knowledge, Skills, and Abilities with level data.

        Returns:
            dict: Dictionary of DataFrames keyed by file type
        """
        self.dataframes.clear()

        # Required files for KSAMDS core entities
        required_files = ['knowledge', 'skills', 'abilities', 'occupation_data',
                          'task_ratings', 'task_statements', 'work_activities']

        # Optional supplementary files
        optional_files = ['content_model_reference', 'scales_reference', 'job_zones',
                          'education_training_experience', 'work_context', 'interests', 'work_values']

        # Load required files first
        for key in required_files:
            filename = self.key_files[key]
            df = self.load_excel_file(filename)
            if df is not None:
                # Enrich KSA files with level data
                if key in ['knowledge', 'skills', 'abilities']:
                    df = self._enrich_with_levels(df, key)
                self.dataframes[key] = df
            else:
                logger.warning(
                    f"Failed to load required file {key} ({filename})")

        # Load optional files
        for key in optional_files:
            filename = self.key_files[key]
            df = self.load_excel_file(filename)
            if df is not None:
                self.dataframes[key] = df
            else:
                logger.info(f"Optional file not loaded: {key} ({filename})")

        logger.info(
            f"Successfully loaded {len(self.dataframes)}/{len(self.key_files)} key files")
        logger.info(
            f"Required files loaded: {len([k for k in required_files if k in self.dataframes])}/{len(required_files)}")

        return self.dataframes

    def _filter_excluded_occupations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out occupations with O*NET-SOC Code starting with '55-' 
        or Title ending with 'All Other' or 'All Other Specialists'.

        Args:
            df: DataFrame with 'O*NET-SOC Code' and optionally 'Title' columns

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if 'O*NET-SOC Code' not in df.columns:
            return df

        initial_count = len(df)

        # Filter out occupations starting with '55-'
        mask = ~df['O*NET-SOC Code'].str.startswith('55-', na=False)

        # Filter out occupations with Title ending in 'All Other' or 'All Other Specialists' if Title column exists
        if 'Title' in df.columns:
            mask = mask & ~df['Title'].str.endswith('All Other', na=False)
            mask = mask & ~df['Title'].str.endswith(
                'All Other Specialists', na=False)

        df_filtered = df[mask].copy()

        excluded_count = initial_count - len(df_filtered)
        if excluded_count > 0:
            logger.info(
                f"Filtered out {excluded_count} occupation records (55-*, 'All Other', or 'All Other Specialists')")

        return df_filtered

    def _get_valid_occupation_codes(self) -> Set[str]:
        """
        Get the set of valid occupation codes after filtering.

        Returns:
            Set of valid O*NET-SOC codes
        """
        if 'occupation_data' not in self.dataframes:
            return set()

        occupation_df = self.dataframes['occupation_data']
        if isinstance(occupation_df, dict):
            # Handle multi-sheet case
            for sheet_df in occupation_df.values():
                if 'O*NET-SOC Code' in sheet_df.columns:
                    occupation_df = sheet_df
                    break

        if isinstance(occupation_df, pd.DataFrame):
            filtered_df = self._filter_excluded_occupations(occupation_df)
            return set(filtered_df['O*NET-SOC Code'].unique())

        return set()

    def save_dataframes_to_csv(self, output_dir: Optional[str] = None) -> bool:
        """
        Save all loaded DataFrames to CSV files for inspection.
        Filters out occupations with O*NET-SOC Code starting with '55-' 
        or Title ending with 'All Other', and removes related data.

        Args:
            output_dir: Directory to save CSV files (defaults to processed dir)

        Returns:
            bool: True if successful
        """
        if not self.dataframes:
            logger.error(
                "No DataFrames loaded. Run load_all_key_files() first.")
            return False

        if output_dir is None:
            output_dir = self.processed_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get valid occupation codes after filtering
        valid_occupation_codes = self._get_valid_occupation_codes()
        if valid_occupation_codes:
            logger.info(
                f"Retaining {len(valid_occupation_codes)} valid occupations after filtering")

        try:
            for key, df in self.dataframes.items():
                if isinstance(df, dict):
                    # Find the main sheet for multi-sheet Excel files
                    sheet_keys = list(df.keys())
                    main_df = None
                    main_sheet_name = None

                    # Look for a sheet name that matches the key
                    for sheet_name, sheet_df in df.items():
                        if sheet_name.lower().replace(' ', '_') == key.lower():
                            main_df = sheet_df
                            main_sheet_name = sheet_name
                            break

                    # If no matching name, use the first sheet
                    if main_df is None and sheet_keys:
                        main_sheet_name = sheet_keys[0]
                        main_df = df[main_sheet_name]

                    if main_df is not None:
                        # Apply filtering for occupation-related data
                        if key == 'occupation_data':
                            main_df = self._filter_excluded_occupations(
                                main_df)
                        elif 'O*NET-SOC Code' in main_df.columns and valid_occupation_codes:
                            initial_count = len(main_df)
                            main_df = main_df[main_df['O*NET-SOC Code'].isin(
                                valid_occupation_codes)].copy()
                            excluded_count = initial_count - len(main_df)
                            if excluded_count > 0:
                                logger.info(
                                    f"Filtered {excluded_count} records from {key} related to excluded occupations")

                        csv_path = output_dir / f"{key}.csv"
                        main_df.to_csv(csv_path, index=False)
                        logger.info(
                            f"Saved primary data for {key} from sheet '{main_sheet_name}' to {csv_path}")

                else:
                    # Single DataFrame case
                    df_to_save = df.copy()

                    # Apply filtering for occupation-related data
                    if key == 'occupation_data':
                        df_to_save = self._filter_excluded_occupations(
                            df_to_save)
                    elif 'O*NET-SOC Code' in df_to_save.columns and valid_occupation_codes:
                        initial_count = len(df_to_save)
                        df_to_save = df_to_save[df_to_save['O*NET-SOC Code'].isin(
                            valid_occupation_codes)].copy()
                        excluded_count = initial_count - len(df_to_save)
                        if excluded_count > 0:
                            logger.info(
                                f"Filtered {excluded_count} records from {key} related to excluded occupations")

                    csv_path = output_dir / f"{key}.csv"
                    df_to_save.to_csv(csv_path, index=False)
                    logger.info(f"Saved {csv_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save DataFrames: {e}")
            return False

    def cleanup_temp_files(self):
        """Clean up temporary extraction directory."""
        if self.temp_extract_dir.exists():
            logger.info(
                f"Cleaning up temporary files in {self.temp_extract_dir}")
            try:
                shutil.rmtree(self.temp_extract_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")

    def cleanup_intermediate_files(self):
        """
        Clean up intermediate files after pipeline completion.
        Keeps only the original ZIP file and removes extracted/intermediate files.
        """
        logger.info("Cleaning up intermediate files...")

        # Clean up temp extraction directory
        self.cleanup_temp_files()

        # We might want to clean up intermediate CSVs after they've been processed by the mapper
        if self.processed_dir.exists():
            logger.info(
                f"Note: Intermediate CSVs in {self.processed_dir} can be removed after mapping completes")

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all loaded DataFrames.

        Returns:
            dict: Summary information for each DataFrame
        """
        summary = {}

        for key, df in self.dataframes.items():
            if isinstance(df, dict):  # Multiple sheets
                summary[key] = {}
                for sheet_name, sheet_df in df.items():
                    summary[key][sheet_name] = {
                        'rows': len(sheet_df),
                        'columns': len(sheet_df.columns),
                        'column_names': list(sheet_df.columns)
                    }
            else:
                summary[key] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'missing_data': df.isnull().sum().to_dict()
                }

                # Add level data summary for KSA entities
                if key in ['knowledge', 'skills', 'abilities'] and 'Data Value LV' in df.columns:
                    lv_count = df['Data Value LV'].notna().sum()
                    lv_range = (df['Data Value LV'].min(
                    ), df['Data Value LV'].max()) if lv_count > 0 else (None, None)
                    summary[key]['level_data'] = {
                        'records_with_levels': lv_count,
                        'level_range': lv_range
                    }

        return summary

    def extract_all(self, force_download: bool = False, cleanup_after: bool = True) -> bool:
        """
        Complete extraction pipeline: download, extract, and load all data.

        Args:
            force_download: Whether to re-download existing files
            cleanup_after: Whether to clean up temp files after extraction

        Returns:
            bool: True if successful
        """
        logger.info("Starting O*NET data extraction pipeline...")

        # Step 1: Download
        if not self.download_onet_database(force_download):
            return False

        # Step 2: Extract
        if not self.extract_database():
            return False

        # Step 3: Load key files
        dataframes = self.load_all_key_files()
        if not dataframes:
            logger.error("No files were successfully loaded")
            return False

        # Step 4: Save to CSV for inspection
        if not self.save_dataframes_to_csv():
            logger.warning(
                "Failed to save CSV files, but extraction was successful")

        # Step 5: Clean up if requested
        if cleanup_after and self._cleanup_on_extract:
            self.cleanup_temp_files()

        logger.info("O*NET extraction pipeline completed successfully!")
        return True


def main():
    """Main function to run the O*NET extractor."""
    extractor = ONetExtractor()

    # Run complete extraction
    success = extractor.extract_all(force_download=False, cleanup_after=True)

    if success:
        # Log summary
        summary = extractor.get_data_summary()
        logger.info("=" * 70)
        logger.info("O*NET DATA SUMMARY")
        logger.info("=" * 70)

        for key, info in summary.items():
            if isinstance(info, dict) and 'rows' not in info:  # Multiple sheets
                logger.info(f"\n{key.upper()}:")
                for sheet, sheet_info in info.items():
                    logger.info(
                        f"  {sheet}: {sheet_info['rows']} rows, {sheet_info['columns']} columns")
            else:
                logger.info(
                    f"{key.upper()}: {info['rows']} rows, {info['columns']} columns")
                # Log level data if available
                if 'level_data' in info:
                    level_info = info['level_data']
                    logger.info(
                        f"  Level data: {level_info['records_with_levels']} records, "
                        f"range: {level_info['level_range']}")

        logger.info("=" * 70)
        logger.info(f"Raw data saved to: {extractor.raw_dir}")
        logger.info(f"CSV files saved to: {extractor.processed_dir}")
        logger.info("=" * 70)
    else:
        logger.error("Extraction failed. Check logs for details.")


if __name__ == "__main__":
    main()
