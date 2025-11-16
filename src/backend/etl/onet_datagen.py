"""
O*NET Synthetic Attribute Generator for KSAMDS Project

Generates synthetic attributes for Tasks and Functions using semantic embeddings
and cosine similarity matching against reference categories.

Tasks need: type, mode, environment
Functions need: physicality, cognitive_load, environment

Uses Google Gemini Embedding Model to embed occupation+entity pairs and matches them
against predefined reference category embeddings.
"""

import os
import time
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONetSyntheticAttributeGenerator:
    """
    Generates synthetic attributes for tasks and functions
    using semantic embeddings and cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-embedding-001",
        batch_size: int = 100,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the synthetic attribute generator.

        Args:
            model_name: Google AI embedding model to use
            batch_size: Number of texts to process in one API call
            confidence_threshold: Minimum similarity score to assign a category
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Get project root directory (3 levels up from etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / "data" / "archive" / "intermediate"
        self.cache_dir = project_root / "data" / "archive" / "embeddings"

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define reference category descriptions
        self.reference_categories = self._define_reference_categories()

        # Track cache files created in this session
        self._session_cache_files = set()

        # Initialize statistics
        self.stats = {
            'total_processed': 0,
            'low_confidence_assignments': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"Using embedding model: {self.model_name}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

        # Configure Google AI API
        try:
            if 'GOOGLE_API_KEY' in os.environ:
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                logger.info("Configured Google AI with environment variable.")
            else:
                # Try Colab userdata if environment variable not available
                try:
                    from google.colab import userdata
                    api_key = userdata.get('GOOGLE_API_KEY')
                    genai.configure(api_key=api_key)
                    logger.info("Configured Google AI with Colab userdata.")
                except ImportError:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable not set and not running in Colab")
        except Exception as e:
            logger.error(
                "Could not configure Google AI. Please set GOOGLE_API_KEY environment variable.")
            raise e

    def _define_reference_categories(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Define reference descriptions for each category combination.
        These will be embedded and used as reference points for similarity matching.
        """
        return {
            'function': {
                'physicality': {
                    'Light': 'Minimal physical effort, primarily sedentary work, desk-based activities, light manipulation of objects, little to no lifting or physical strain',
                    'Moderate': 'Some physical activity required, occasional standing or walking, moderate lifting or carrying, mixed physical and sedentary tasks',
                    'Heavy': 'Significant physical demands, continuous standing or movement, heavy lifting, strenuous physical labor, high endurance requirements'
                },
                'cognitive_load': {
                    'Light': 'Routine and repetitive tasks, minimal decision-making, simple processes, clear procedures, low mental complexity',
                    'Moderate': 'Some problem-solving required, occasional complex decisions, varied tasks, moderate attention and concentration needed',
                    'Heavy': 'Complex problem-solving, strategic thinking, high-stakes decisions, sustained concentration, managing multiple priorities simultaneously'
                },
                'environment': {
                    'Indoors (Office/Lab)': 'Work performed in controlled indoor environments such as offices, laboratories, studios, or climate-controlled facilities',
                    'Outdoors': 'Work conducted in outdoor settings, exposed to weather conditions, natural environments, construction sites, or field locations',
                    'Virtual/Remote': 'Work performed remotely via digital platforms, online collaboration, virtual meetings, distributed teams, work-from-home settings',
                    'Public-Facing': 'Direct interaction with customers, clients, or the general public, service-oriented roles, front-line positions',
                    'Team-Oriented': 'Collaborative work with colleagues, team-based projects, interdependent tasks, group coordination and cooperation',
                    'Independent': 'Autonomous work with minimal supervision, self-directed tasks, individual responsibility, working alone or with minimal collaboration'
                }
            },
            'task': {
                'type': {
                    'Core': 'Essential and fundamental tasks central to the occupation, primary job responsibilities, mission-critical activities, defining characteristics of the role',
                    'Supplemental': 'Supporting or auxiliary tasks, administrative duties, secondary responsibilities, tasks that facilitate core work but are not primary job functions'
                },
                'mode': {
                    'Physical Tool': 'Tasks involving manual tools, equipment, machinery, instruments, or physical devices requiring hands-on operation',
                    'Software/Technology': 'Tasks using computer software, digital platforms, applications, programming, or technology-based systems',
                    'Process/Method': 'Tasks following systematic procedures, methodologies, protocols, standardized processes, or established workflows',
                    'Theory/Concept': 'Tasks involving theoretical knowledge, conceptual understanding, abstract reasoning, principles, or academic concepts',
                    'Physical Action': 'Tasks requiring bodily movement, manual labor, physical manipulation, kinesthetic activities, or athletic performance',
                    'Communication': 'Tasks centered on verbal or written communication, presentations, meetings, negotiations, teaching, or interpersonal interaction'
                },
                'environment': {
                    'Indoors (Office/Lab)': 'Tasks performed in controlled indoor environments such as offices, laboratories, studios, or climate-controlled facilities',
                    'Outdoors': 'Tasks conducted in outdoor settings, exposed to weather conditions, natural environments, construction sites, or field locations',
                    'Virtual/Remote': 'Tasks performed remotely via digital platforms, online collaboration, virtual meetings, distributed teams, work-from-home settings',
                    'Public-Facing': 'Tasks involving direct interaction with customers, clients, or the general public, service-oriented activities, front-line work',
                    'Team-Oriented': 'Tasks requiring collaboration with colleagues, team-based activities, interdependent work, group coordination and cooperation',
                    'Independent': 'Tasks performed autonomously with minimal supervision, self-directed work, individual responsibility, working alone or with minimal collaboration'
                }
            }
        }

    def _create_embedding_text(
        self,
        occupation: str,
        entity: str,
        entity_type: str
    ) -> str:
        """
        Create combined text for embedding generation.

        Args:
            occupation: The occupation name
            entity: The entity name (task/function)
            entity_type: Type of entity ('task' or 'function')

        Returns:
            Combined text string
        """
        return f"Occupation: {occupation}. {entity_type.capitalize()}: {entity}"

    def _generate_cache_key(self, texts: List[str]) -> str:
        """Generate a unique hash for a list of texts."""
        content = f"{self.model_name}_{'_'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[np.ndarray]]:
        """Load embeddings from cache if available."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            logger.info(f"Loading embeddings from cache: {cache_path.name}")
            # Track this cache file as part of current session
            self._session_cache_files.add(cache_path)
            with open(cache_path, 'rb') as f:
                self.stats['cache_hits'] += 1
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key: str, embeddings: List[np.ndarray]):
        """Save embeddings to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        # Track this cache file as part of current session
        self._session_cache_files.add(cache_path)
        logger.info(f"Saved embeddings to cache: {cache_path.name}")

    def _generate_embeddings_batch(
        self,
        texts: List[str],
        delay: float = 1.0,
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts with retry logic and caching.

        Args:
            texts: List of texts to embed
            delay: Delay between batches in seconds
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding arrays
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(texts)
            cached_embeddings = self._load_from_cache(cache_key)
            if cached_embeddings is not None:
                return cached_embeddings

        self.stats['cache_misses'] += 1

        all_embeddings = []
        max_retries = 3
        base_delay = 2

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    all_embeddings.extend(result['embedding'])
                    self.stats['api_calls'] += 1

                    if len(texts) > self.batch_size:
                        logger.info(
                            f"Processed batch {batch_num}/{(len(texts)-1)//self.batch_size + 1}")
                        time.sleep(delay)

                    break  # Success

                except Exception as e:
                    wait_time = base_delay * (2 ** attempt)
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Batch {batch_num} failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Batch {batch_num} failed after {max_retries} attempts")
                        raise

        embeddings = [np.array(emb) for emb in all_embeddings]

        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, embeddings)

        return embeddings

    def _generate_reference_embeddings(self, entity_type: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate embeddings for all reference categories.

        Args:
            entity_type: Type of entity ('task' or 'function')

        Returns:
            Dictionary of reference embeddings for all attributes
        """
        logger.info(f"Generating reference embeddings for {entity_type}...")

        reference_embeddings = {}

        # Get all attribute types for this entity
        attribute_types = self.reference_categories[entity_type].keys()

        for attr_type in attribute_types:
            categories = self.reference_categories[entity_type][attr_type]
            category_texts = list(categories.values())
            category_names = list(categories.keys())

            logger.info(
                f"  Embedding {len(category_texts)} {attr_type} categories...")
            embeddings = self._generate_embeddings_batch(
                category_texts, delay=0.5)
            reference_embeddings[attr_type] = {
                name: emb for name, emb in zip(category_names, embeddings)}

        return reference_embeddings

    def _assign_category(
        self,
        entity_embedding: np.ndarray,
        reference_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[str, float]:
        """
        Assign category based on highest cosine similarity.

        Args:
            entity_embedding: Embedding of the entity
            reference_embeddings: Dictionary of reference category embeddings

        Returns:
            Tuple of (category_name, confidence_score)
        """
        best_category = None
        best_score = -1.0

        for category_name, ref_embedding in reference_embeddings.items():
            similarity = cosine_similarity(
                entity_embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]

            if similarity > best_score:
                best_score = similarity
                best_category = category_name

        return best_category, float(best_score)

    def _check_columns_exist_and_filled(
        self,
        df: pd.DataFrame,
        entity_type: str
    ) -> bool:
        """
        Check if all attribute columns exist and are fully populated.

        Args:
            df: DataFrame to check
            entity_type: Type of entity ('task' or 'function')

        Returns:
            True if all columns exist and are fully filled, False otherwise
        """
        # Get all attribute types for this entity
        attribute_types = self.reference_categories[entity_type].keys()

        # Check each attribute column
        for attr_type in attribute_types:
            col_name = f'{entity_type}_{attr_type}'

            # Check if column exists
            if col_name not in df.columns:
                return False

            # Check if all rows are filled (no NaN values)
            if not df[col_name].notna().all():
                return False

        logger.info(
            f"✓ {entity_type.capitalize()} already has all attribute columns fully populated")
        return True

    def process_entity_dataframe(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_col: str,
        occupation_col: str = 'Title'
    ) -> pd.DataFrame:
        """
        Process a dataframe to add attribute columns.

        Args:
            df: Input dataframe with occupation and entity columns
            entity_type: Type of entity ('task' or 'function')
            entity_col: Name of the entity column
            occupation_col: Name of the occupation column (default 'Title')

        Returns:
            DataFrame with added attribute columns
        """
        logger.info("=" * 70)
        logger.info(f"PROCESSING {entity_type.upper()} DATAFRAME")
        logger.info("-" * 70)
        logger.info(f"Total records: {len(df)}")

        # Check if columns already exist and are fully populated
        if self._check_columns_exist_and_filled(df, entity_type):
            logger.info(f"Skipping {entity_type} - already processed")
            logger.info("=" * 70)
            return df

        # Check if required columns exist
        if entity_col not in df.columns:
            logger.error(
                f"Entity column '{entity_col}' not found in dataframe")
            return df

        # For O*NET data, we need to get occupation titles
        # The intermediate files have 'O*NET-SOC Code' but we need to join with occupation titles
        if occupation_col not in df.columns and 'O*NET-SOC Code' in df.columns:
            # Load occupation data to get titles
            occupation_path = self.data_dir / "occupation_data.csv"
            if occupation_path.exists():
                occupation_df = pd.read_csv(occupation_path)
                # Create a mapping from O*NET-SOC Code to Title
                occupation_lookup = dict(
                    zip(occupation_df['O*NET-SOC Code'], occupation_df['Title']))
                # Add occupation title to the dataframe
                df['Title'] = df['O*NET-SOC Code'].map(occupation_lookup)
                occupation_col = 'Title'
                logger.info(
                    f"Mapped {df['Title'].notna().sum()} occupation titles from O*NET-SOC codes")
            else:
                logger.error(
                    f"Occupation data file not found at {occupation_path}")
                return df

        if occupation_col not in df.columns:
            logger.error(
                f"Occupation column '{occupation_col}' not found in dataframe")
            return df

        logger.info("=" * 70)

        # Generate reference embeddings
        reference_embeddings = self._generate_reference_embeddings(entity_type)

        # Create embedding texts
        logger.info(
            f"Creating embedding texts for {len(df)} {entity_type} items...")
        embedding_texts = []
        for _, row in df.iterrows():
            occupation = row.get(occupation_col, 'Unknown')
            entity_text = row.get(entity_col, '')
            if pd.notna(occupation) and pd.notna(entity_text):
                embedding_texts.append(
                    self._create_embedding_text(occupation, entity_text, entity_type))
            else:
                # Handle missing values
                embedding_texts.append(f"{entity_type}: {entity_text}")

        # Generate entity embeddings
        logger.info(
            f"Generating embeddings for {len(embedding_texts)} items...")
        start_time = time.time()
        entity_embeddings = self._generate_embeddings_batch(embedding_texts)
        duration = time.time() - start_time
        logger.info(
            f"Generated {len(entity_embeddings)} embeddings in {duration:.2f} seconds")

        # Assign all attributes dynamically
        df = df.copy()
        attribute_types = self.reference_categories[entity_type].keys()

        for attr_type in attribute_types:
            logger.info(f"Assigning {attr_type} categories...")
            assignments = []
            confidences = []

            for emb in entity_embeddings:
                category, confidence = self._assign_category(
                    emb, reference_embeddings[attr_type])
                assignments.append(category)
                confidences.append(confidence)

                if confidence < self.confidence_threshold:
                    self.stats['low_confidence_assignments'] += 1

            # Add columns to dataframe
            df[f'{entity_type}_{attr_type}'] = assignments
            df[f'{entity_type}_{attr_type}_confidence'] = confidences

        self.stats['total_processed'] += len(df)

        # Log statistics
        logger.info("-" * 70)
        logger.info("ASSIGNMENT SUMMARY:")

        for attr_type in attribute_types:
            col_name = f'{entity_type}_{attr_type}'
            conf_col = f'{entity_type}_{attr_type}_confidence'

            logger.info(f"{attr_type.capitalize()} distribution:")
            for cat, count in df[col_name].value_counts().items():
                logger.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")

            logger.info(
                f"Average {attr_type} confidence: {df[conf_col].mean():.3f}")

        logger.info("=" * 70)

        return df

    def process_tasks(self, save_to_csv: bool = True) -> bool:
        """
        Process task statements and add synthetic attributes.

        Args:
            save_to_csv: Whether to save results back to CSV file

        Returns:
            bool: True if successful
        """
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING TASKS")
        logger.info("=" * 80)

        task_path = self.data_dir / "task_statements.csv"
        if not task_path.exists():
            logger.error(f"Task statements file not found: {task_path}")
            return False

        try:
            task_df = pd.read_csv(task_path)
            logger.info(f"Loaded {len(task_df)} task records from {task_path}")

            # Process task dataframe
            task_df = self.process_entity_dataframe(
                df=task_df,
                entity_type='task',
                entity_col='Task',
                occupation_col='Title'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in task_df.columns if '_confidence' not in col]
                task_df[final_cols].to_csv(task_path, index=False)
                logger.info(
                    f"✓ Saved updated task_statements.csv to {task_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to process tasks: {e}", exc_info=True)
            return False

    def process_functions(self, save_to_csv: bool = True) -> bool:
        """
        Process work activities (functions) and add synthetic attributes.

        Only processes functions with Scale ID = 'IM' (Importance) and Data Value >= 3.0
        to match the filtering logic in onet_mapper.py

        Args:
            save_to_csv: Whether to save results back to CSV file

        Returns:
            bool: True if successful
        """
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING FUNCTIONS (Work Activities)")
        logger.info("=" * 80)

        function_path = self.data_dir / "work_activities.csv"
        if not function_path.exists():
            logger.error(f"Work activities file not found: {function_path}")
            return False

        try:
            function_df = pd.read_csv(function_path)
            logger.info(
                f"Loaded {len(function_df)} function records from {function_path}")

            # Filter for importance (IM) >= 3.0 (same as mapper logic)
            original_count = len(function_df)
            function_df_filtered = function_df[
                (function_df['Scale ID'] == 'IM') &
                (function_df['Data Value'] >= 3.0)
            ].copy()

            logger.info(
                f"Filtered to {len(function_df_filtered)} function records (IM >= 3.0) "
                f"from {original_count} total (skipped {original_count - len(function_df_filtered)})"
            )

            # Process only the filtered function dataframe
            # Work activities use 'Element Name' as the entity column
            function_df_processed = self.process_entity_dataframe(
                df=function_df_filtered,
                entity_type='function',
                entity_col='Element Name',
                occupation_col='Title'
            )

            if save_to_csv:
                # Merge processed data back with original dataframe
                # Keep all original records but only the filtered ones have synthetic attributes

                # Create a mapping key to identify which records were processed
                if 'O*NET-SOC Code' in function_df_processed.columns and 'Element ID' in function_df_processed.columns:
                    function_df_processed['_merge_key'] = (
                        function_df_processed['O*NET-SOC Code'].astype(str) + '_' +
                        function_df_processed['Element ID'].astype(str)
                    )
                    function_df['_merge_key'] = (
                        function_df['O*NET-SOC Code'].astype(str) + '_' +
                        function_df['Element ID'].astype(str)
                    )

                    # Get only the new synthetic attribute columns
                    synthetic_cols = [col for col in function_df_processed.columns
                                      if col.startswith('function_') and col not in function_df.columns]

                    # Merge back: add synthetic attributes to matching rows in original df
                    merge_cols = ['_merge_key'] + synthetic_cols
                    function_df = function_df.merge(
                        function_df_processed[merge_cols],
                        on='_merge_key',
                        how='left'
                    )

                    # Drop the temporary merge key
                    function_df = function_df.drop(columns=['_merge_key'])

                    logger.info(
                        f"Merged synthetic attributes: {len(synthetic_cols)} new columns added")
                else:
                    # Fallback: if merge keys not available, just save the processed subset
                    logger.warning(
                        "Could not merge - missing merge key columns. Saving only filtered records.")
                    function_df = function_df_processed

                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in function_df.columns if '_confidence' not in col]
                function_df[final_cols].to_csv(function_path, index=False)
                logger.info(
                    f"✓ Saved updated work_activities.csv to {function_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to process functions: {e}", exc_info=True)
            return False

    def process_all(self, save_to_csv: bool = True) -> bool:
        """
        Process both tasks and functions to add synthetic attributes.

        Args:
            save_to_csv: Whether to save results back to CSV files

        Returns:
            bool: True if all processing successful
        """
        logger.info("=" * 80)
        logger.info("SYNTHETIC ATTRIBUTE GENERATION - TASKS AND FUNCTIONS")
        logger.info("=" * 80)

        tasks_success = self.process_tasks(save_to_csv=save_to_csv)
        functions_success = self.process_functions(save_to_csv=save_to_csv)

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("SYNTHETIC ATTRIBUTE GENERATION COMPLETED")
        logger.info("=" * 80)
        logger.info(
            f"Total records processed: {self.stats['total_processed']}")
        logger.info(f"Total API calls: {self.stats['api_calls']}")
        logger.info(f"Cache hits: {self.stats['cache_hits']}")
        logger.info(f"Cache misses: {self.stats['cache_misses']}")
        logger.info(
            f"Low confidence assignments (<{self.confidence_threshold}): {self.stats['low_confidence_assignments']}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info("=" * 80)

        return tasks_success and functions_success


def main():
    """Main function to run the synthetic attribute generator."""
    generator = ONetSyntheticAttributeGenerator(
        model_name="models/gemini-embedding-001",
        batch_size=100,
        confidence_threshold=0.5
    )

    success = generator.process_all(save_to_csv=True)

    if success:
        logger.info("✅ All synthetic attributes generated successfully!")
    else:
        logger.error("❌ Synthetic attribute generation failed!")
        exit(1)


if __name__ == "__main__":
    main()
