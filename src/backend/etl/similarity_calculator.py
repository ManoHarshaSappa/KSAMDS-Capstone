"""
Similarity Calculator for KSAMDS Project
Calculates cosine similarity between entity embeddings and infers relationships
Supports multiple relationship types with configurable thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

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

class SimilarityCalculator:
    """
    Calculates similarity between entity embeddings and infers relationships.
    Supports configurable thresholds for different relationship types.
    """
    def __init__(
        self,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        max_relationships: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the similarity calculator.
        """
        self.similarity_thresholds = similarity_thresholds or {
            'knowledge_skill': 0.70,
            'skill_ability': 0.65,
            'knowledge_function': 0.60,
            'ability_task': 0.65,
            'function_task': 0.75
        }
        self.max_relationships = max_relationships or {
            'knowledge_skill': 15,
            'skill_ability': 10,
            'knowledge_function': 20,
            'ability_task': 25,
            'function_task': 10
        }
        logger.info("Initialized SimilarityCalculator")
        logger.info(f"Similarity thresholds: {self.similarity_thresholds}")
        logger.info(f"Max relationships: {self.max_relationships}")

    def calculate_relationships(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_embeddings: Dict[str, np.ndarray],
        target_embeddings: Dict[str, np.ndarray],
        relationship_type: str
    ) -> pd.DataFrame:
        """
        Calculate relationships between source and target entities.
        """
        method_start_time = log_timed_event(f"Calculating '{relationship_type}' relationships ({len(source_df)} x {len(target_df)})...")
        
        source_names = source_df['name'].tolist()
        target_names = target_df['name'].tolist()
        
        source_embedding_matrix = np.array([source_embeddings.get(name) for name in source_names])
        target_embedding_matrix = np.array([target_embeddings.get(name) for name in target_names])
        
        sim_start_time = log_timed_event(f"  [COMPUTATION] Calculating cosine similarity matrix for '{relationship_type}'...")
        similarity_matrix = cosine_similarity(source_embedding_matrix, target_embedding_matrix)
        log_timed_event(f"  [COMPUTATION] Similarity matrix calculation finished.", sim_start_time)
        
        relationships = []
        threshold = self.similarity_thresholds.get(relationship_type, 0.60)
        max_per_entity = self.max_relationships.get(relationship_type, 20)
        
        for i, source_name in enumerate(source_names):
            similarities = similarity_matrix[i]
            candidate_indices = np.where(similarities >= threshold)[0]
            
            if not len(candidate_indices):
                continue
            
            candidate_similarities = similarities[candidate_indices]
            sorted_indices = np.argsort(-candidate_similarities)
            
            count = 0
            for j, sim in zip(candidate_indices[sorted_indices], candidate_similarities[sorted_indices]):
                if count >= max_per_entity:
                    break
                # Business rule: Don't create self-relationships
                if source_name != target_names[j]:
                    relationships.append({
                        'source_name': source_name,
                        'target_name': target_names[j],
                        'confidence_score': float(sim)
                    })
                    count += 1
                    
        log_timed_event(f"Finished calculating '{relationship_type}'. Found {len(relationships)} relationships.", method_start_time)
        return pd.DataFrame(relationships)

    def calculate_all_relationships(
        self,
        knowledge_df: pd.DataFrame,
        skills_df: pd.DataFrame,
        abilities_df: pd.DataFrame,
        functions_df: pd.DataFrame,
        tasks_df: pd.DataFrame,
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all relationship types. This signature is compatible with the
        EmbeddingRelationshipBuilder.
        """
        all_relationships = {}
        
        # Create a dictionary of dataframes for easier access
        entities = {
            'knowledge': knowledge_df,
            'skills': skills_df,
            'abilities': abilities_df,
            'functions': functions_df,
            'tasks': tasks_df
        }
        
        configs = [
            ('knowledge_skill', 'knowledge', 'skills'),
            ('skill_ability', 'skills', 'abilities'),
            ('knowledge_function', 'knowledge', 'functions'),
            ('ability_task', 'abilities', 'tasks'),
            ('function_task', 'functions', 'tasks')
        ]
        
        for rel_type, src_key, tgt_key in configs:
            all_relationships[rel_type] = self.calculate_relationships(
                source_df=entities[src_key],
                target_df=entities[tgt_key],
                source_embeddings=embeddings[src_key],
                target_embeddings=embeddings[tgt_key],
                relationship_type=rel_type
            )
            
        return all_relationships

    def save_relationships(
        self,
        relationships: Dict[str, pd.DataFrame],
        output_dir: str = "data/processed"
    ):
        """
        Save calculated relationships to CSV files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving relationships to {output_dir}")
        
        for rel_type, rel_df in relationships.items():
            filename = f"{rel_type}_inferred.csv"
            filepath = output_path / filename
            rel_df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(rel_df)} {rel_type} relationships to {filepath}")