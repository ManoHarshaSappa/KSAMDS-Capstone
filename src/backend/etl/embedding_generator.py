"""
Embedding Generator for KSAMDS Project
Generates embeddings for Knowledge, Skills, Abilities, Functions, and Tasks
Uses Google AI API (models/embedding-001) for embedding generation
Supports caching and batch processing for efficiency
"""

import os
import pickle
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Helper function for timed logging ---


def log_timed_event(message, start_time=None):
    """Prints a message with a timestamp and optional duration."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if start_time:
        duration = time.time() - start_time
        print(f"[{timestamp}] ✅ {message} (Duration: {duration:.2f} seconds)")
    else:
        print(f"[{timestamp}] ⏳ {message}")
        return time.time()


class EmbeddingGenerator:
    """
    Generates embeddings using the Google AI API and caches the results.
    Includes caching, batch processing, and progress tracking.
    """

    def __init__(
        self,
        model_name: str = "models/embedding-001",
        cache_dir: str = "data/embeddings",
        batch_size: int = 100,  # Used for API calls
        # Included for compatibility, not used by Google API
        device: Optional[str] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Google AI model to use
            cache_dir: Directory to store cached embeddings
            batch_size: Number of texts to process in one API call
            device: Not used by Google AI, included for compatibility
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using Google AI model: {self.model_name}")

        try:
            # Attempt to configure the API key.
            # In a non-Colab environment, you might use os.environ.get('GOOGLE_API_KEY')
            if 'GOOGLE_API_KEY' in os.environ:
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                logger.info("Configured Google AI with environment variable.")
            else:
                # This part is for Google Colab.
                from google.colab import userdata
                api_key = userdata.get('GOOGLE_API_KEY')
                genai.configure(api_key=api_key)
                logger.info("Configured Google AI with Colab userdata.")
        except Exception as e:
            logger.error(
                "Could not configure Google AI. Please set the 'GOOGLE_API_KEY' in your environment or Colab Secrets.")
            raise e

    def _create_entity_text(
        self,
        name: str,
        entity_type: str
    ) -> str:
        """
        Create rich text representation for embedding generation.
        """
        return f"{entity_type}: {name}"

    def _generate_cache_key(self, df: pd.DataFrame, entity_type: str) -> str:
        """Generates a unique hash based on the dataframe content and model."""
        content = f"{entity_type}_{self.model_name}_{df.to_json()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, entity_type: str, cache_key: str) -> Path:
        """Gets the full path for a cache file."""
        return self.cache_dir / f"{entity_type}_{cache_key}.pkl"

    def generate_for_entity_type(
        self,
        df: pd.DataFrame,
        entity_type: str,
        name_col: str = 'name',
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a specific entity type using Google AI.
        """
        method_start_time = log_timed_event(
            f"Processing embeddings for {len(df)} '{entity_type}' entities...")

        # --- CACHING LOGIC: CHECK FOR EXISTING FILE ---
        cache_key = self._generate_cache_key(df, entity_type)
        cache_path = self._get_cache_path(entity_type, cache_key)

        if use_cache and cache_path.exists():
            log_timed_event(
                f"Found cache file. Loading embeddings from '{cache_path}'.", method_start_time)
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        log_timed_event(
            f"No cache found. Generating new embeddings via Google AI for '{entity_type}'...")

        texts, names = [], []
        for _, row in df.iterrows():
            texts.append(self._create_entity_text(row[name_col], entity_type))
            names.append(row[name_col])

        encode_start_time = log_timed_event(
            f"  [API CALL] Sending {len(texts)} texts to Google '{self.model_name}'...")
        all_embeddings = []

        # IMPROVED ERROR HANDLING AND RATE LIMITING
        max_retries = 3
        base_delay = 2  # Increased from 1 second

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1

            # Retry logic for this batch
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    all_embeddings.extend(result['embedding'])

                    if len(texts) > self.batch_size:
                        log_timed_event(
                            f"    Batch {batch_num}/{(len(texts)-1)//self.batch_size + 1} complete. Waiting {base_delay} seconds...")
                        time.sleep(base_delay)  # Increased delay

                    break  # Success, exit retry loop

                except Exception as e:
                    wait_time = base_delay * \
                        (2 ** attempt)  # Exponential backoff

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"    Batch {batch_num} failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        logger.info(f"    Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"    Batch {batch_num} failed after {max_retries} attempts")
                        # Save partial results before failing
                        if all_embeddings:
                            partial_embeddings = {name: np.array(embedding)
                                                  for name, embedding in zip(names[:len(all_embeddings)], all_embeddings)}
                            partial_cache_path = self.cache_dir / \
                                f"{entity_type}_{cache_key}_partial.pkl"
                            with open(partial_cache_path, 'wb') as f:
                                pickle.dump(partial_embeddings, f)
                            logger.info(
                                f"    Saved {len(partial_embeddings)} partial embeddings to {partial_cache_path}")
                        raise

        log_timed_event(
            f"  [API CALL] Received all {len(all_embeddings)} embeddings from Google.", encode_start_time)

        embeddings = {name: np.array(embedding)
                      for name, embedding in zip(names, all_embeddings)}

        # --- CACHING LOGIC: SAVE NEWLY FETCHED EMBEDDINGS ---
        if use_cache:
            log_timed_event(
                f"Saving newly generated embeddings to cache file: '{cache_path}'")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)

        log_timed_event(
            f"Finished processing embeddings for '{entity_type}'.", method_start_time)
        return embeddings

    def generate_all(
        self,
        knowledge_df: pd.DataFrame,
        skills_df: pd.DataFrame,
        abilities_df: pd.DataFrame,
        functions_df: pd.DataFrame,
        tasks_df: pd.DataFrame,
        use_cache: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate embeddings for all entity types. This signature is compatible
        with the EmbeddingRelationshipBuilder.
        """
        logger.info("="*70)
        logger.info(
            "Starting embedding generation for all entity types (Google AI)")
        logger.info(f"Model: {self.model_name}")
        logger.info("="*70)

        all_embeddings = {}

        entity_configs = [
            ('knowledge', knowledge_df, 'Knowledge'),
            ('skills', skills_df, 'Skill'),
            ('abilities', abilities_df, 'Ability'),
            ('functions', functions_df, 'Function'),
            ('tasks', tasks_df, 'Task')
        ]

        for key, df, entity_type in entity_configs:
            logger.info(f"\n--- Processing {entity_type} ---")
            all_embeddings[key] = self.generate_for_entity_type(
                df=df,
                entity_type=entity_type,
                use_cache=use_cache
            )

        logger.info("\n" + "="*70)
        logger.info("Completed embedding generation for all entity types")
        logger.info("="*70)

        return all_embeddings
