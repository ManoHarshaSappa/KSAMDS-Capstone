"""
O*NET Data Extractor for KSAMDS Project

This module handles downloading and extracting O*NET database files.
Downloads the latest O*NET database, extracts relevant Excel files,
and loads them into pandas DataFrames for further processing.
"""

# import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import shutil
# from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONetExtractor:
    """Extract O*NET database files and convert to pandas DataFrames."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the O*NET extractor.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "onet"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # O*NET database URL (check for latest version)
        self.onet_url = "https://www.onetcenter.org/dl_files/database/db_30_0_excel.zip"
        self.zip_path = self.raw_dir / "onet_database.zip"
        self.extract_dir = self.raw_dir / "extracted"

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
                            print(
                                f"\rDownload progress: {percent:.1f}%", end="")

            print()  # New line after progress
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
        Extract the O*NET database zip file.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.zip_path.exists():
            logger.error(
                "O*NET database zip file not found. Run download_onet_database() first.")
            return False

        try:
            logger.info("Extracting O*NET database...")

            # Remove existing extraction directory
            if self.extract_dir.exists():
                shutil.rmtree(self.extract_dir)

            self.extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)

            logger.info(
                f"Successfully extracted O*NET database to {self.extract_dir}")
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
        file_path = self.extract_dir / "db_30_0_excel" / file_name

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

    def load_all_key_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all key O*NET files needed for KSAMDS.

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

    def save_dataframes_to_csv(self, output_dir: Optional[str] = None) -> bool:
        """
        Save all loaded DataFrames to CSV files for inspection.

        Args:
            output_dir: Directory to save CSV files (defaults to processed/onet/)

        Returns:
            bool: True if successful
        """
        if not self.dataframes:
            logger.error(
                "No DataFrames loaded. Run load_all_key_files() first.")
            return False

        if output_dir is None:
            output_dir = self.data_dir / "processed" / "onet"

        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for key, df in self.dataframes.items():
                if isinstance(df, dict):

                    # Attempt to find a sheet whose name matches the key (case-insensitive)
                    # or just take the first sheet if a match isn't found.
                    sheet_keys = list(df.keys())

                    # Find the sheet that most likely holds the main data
                    main_df = None
                    main_sheet_name = None

                    # 1. Look for a sheet name that matches the key or filename (e.g., 'Abilities')
                    for sheet_name, sheet_df in df.items():
                        if sheet_name.lower().replace(' ', '_') == key.lower():
                            main_df = sheet_df
                            main_sheet_name = sheet_name
                            break

                    # 2. If no matching name is found, just use the first sheet
                    if main_df is None and sheet_keys:
                        main_sheet_name = sheet_keys[0]
                        main_df = df[main_sheet_name]

                    if main_df is not None:
                        csv_path = output_dir / f"{key}.csv"
                        main_df.to_csv(csv_path, index=False)
                        logger.info(
                            f"Saved primary data for {key} from sheet '{main_sheet_name}' to {csv_path}")

                else:
                    # Single DataFrame case (the original 'else' logic)
                    csv_path = output_dir / f"{key}.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {csv_path}")

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

        return summary

    def extract_all(self, force_download: bool = False) -> bool:
        """
        Complete extraction pipeline: download, extract, and load all data.

        Args:
            force_download: Whether to re-download existing files

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

        logger.info("O*NET extraction pipeline completed successfully!")
        return True


def main():
    """Main function to run the O*NET extractor."""
    extractor = ONetExtractor()

    # Run complete extraction
    success = extractor.extract_all(force_download=False)

    if success:
        # Print summary
        summary = extractor.get_data_summary()
        print("\n=== O*NET Data Summary ===")
        for key, info in summary.items():
            if isinstance(info, dict) and 'rows' not in info:  # Multiple sheets
                print(f"\n{key.upper()}:")
                for sheet, sheet_info in info.items():
                    print(
                        f"  {sheet}: {sheet_info['rows']} rows, {sheet_info['columns']} columns")
            else:
                print(
                    f"{key.upper()}: {info['rows']} rows, {info['columns']} columns")

        print(f"\nRaw data saved to: {extractor.raw_dir}")
        print(
            f"CSV files saved to: {extractor.data_dir / 'processed' / 'onet'}")
    else:
        print("Extraction failed. Check logs for details.")


if __name__ == "__main__":
    main()
