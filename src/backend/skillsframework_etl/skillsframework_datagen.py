"""
Synthetic Attribute Generator for KSAMDS Project

Generates synthetic 'type' and 'basis' attributes for Knowledge, Skills, and Abilities
using semantic embeddings and cosine similarity matching against reference categories.

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


class SyntheticAttributeGenerator:
    """
    Generates type and basis attributes for knowledge, skills, and abilities
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

        # Get project root directory (3 levels up from skillsframework_etl folder)
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / "data" / \
            "skillsframework" / "archive" / "intermediate"
        self.cache_dir = project_root / "data" / \
            "skillsframework" / "archive" / "embeddings"

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define reference category descriptions
        self.reference_categories = self._define_reference_categories()

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
                from google.colab import userdata
                api_key = userdata.get('GOOGLE_API_KEY')
                genai.configure(api_key=api_key)
                logger.info("Configured Google AI with Colab userdata.")
        except Exception as e:
            logger.error(
                "Could not configure Google AI. Please set GOOGLE_API_KEY.")
            raise e

    def _define_reference_categories(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Define reference descriptions for each category combination.
        These will be embedded and used as reference points for similarity matching.
        """
        return {
            'knowledge': {
                'type': {
                    'Analytical': 'Theoretical knowledge requiring analytical reasoning, data analysis, statistical methods, research methodologies, and problem-solving through systematic investigation',
                    'Business': 'Commercial and organizational knowledge including business operations, economics, finance, accounting, marketing, management principles, and entrepreneurship',
                    'General': 'Foundational and broad knowledge applicable across multiple domains, including literacy, numeracy, communication, and basic concepts',
                    'Management': 'Leadership and organizational management knowledge including planning, resource allocation, team coordination, strategic thinking, and supervisory principles',
                    'Safety': 'Knowledge of safety protocols, risk assessment, hazard identification, emergency procedures, regulatory compliance, and workplace safety standards',
                    'Scientific': 'Scientific principles, natural sciences, laboratory methods, scientific theory, experimentation, and technical scientific concepts',
                    'Social': 'Understanding of human behavior, social dynamics, psychology, sociology, cultural awareness, interpersonal relationships, and community dynamics',
                    'Technical': 'Specialized technical knowledge of tools, equipment, procedures, systems, software, machinery, and domain-specific technical expertise'
                },
                'basis': {
                    'Academic': 'Theoretical knowledge typically acquired through formal education, university studies, textbooks, lectures, and academic research',
                    'On-the-Job Training': 'Practical knowledge gained through direct workplace experience, mentorship, hands-on practice, and learning by doing',
                    'Professional Development': 'Knowledge obtained through continuing education, professional certifications, workshops, seminars, and career advancement programs',
                    'Vocational Training': 'Trade-specific knowledge acquired through vocational schools, apprenticeships, technical institutes, and specialized training programs'
                }
            },
            'skill': {
                'type': {
                    'Analytical': 'Skills in analyzing data, identifying patterns, conducting research, solving complex problems, and making data-driven decisions',
                    'Business': 'Skills in business operations, customer service, sales, negotiation, financial management, and commercial activities',
                    'General': 'Broadly applicable skills such as communication, time management, adaptability, teamwork, and basic computer literacy',
                    'Management': 'Skills in leading teams, delegating tasks, strategic planning, decision-making, conflict resolution, and organizational coordination',
                    'Safety': 'Skills in maintaining safety standards, conducting safety inspections, responding to emergencies, and implementing safety protocols',
                    'Scientific': 'Skills in conducting experiments, using scientific instruments, applying scientific methods, and performing laboratory procedures',
                    'Social': 'Interpersonal skills including empathy, active listening, counseling, collaboration, teaching, and building relationships',
                    'Technical': 'Hands-on technical skills with tools, equipment, software, machinery, and specialized technical procedures'
                },
                'basis': {
                    'Academic': 'Skills learned through formal education, theoretical study, academic coursework, and structured learning environments',
                    'On-the-Job Training': 'Skills developed through workplace practice, real-world application, mentorship, and direct experience',
                    'Professional Development': 'Skills enhanced through professional training, certifications, continuing education, and career development activities',
                    'Vocational Training': 'Skills acquired through trade schools, apprenticeships, vocational programs, and hands-on technical training'
                }
            },
            'ability': {
                'type': {
                    'Cognitive': 'Mental abilities including reasoning, memory, attention, problem-solving, learning, comprehension, and information processing',
                    'Physical': 'Physical capabilities including strength, endurance, coordination, dexterity, flexibility, and motor skills',
                    'Sensory': 'Perceptual abilities including vision, hearing, touch, spatial awareness, and sensory discrimination'
                },
                'basis': {
                    'Academic': 'Abilities developed through formal education, structured learning, academic exercises, and theoretical training',
                    'On-the-Job Training': 'Abilities honed through workplace practice, repetitive tasks, real-world application, and experiential learning',
                    'Professional Development': 'Abilities refined through professional training, skill enhancement programs, and career advancement activities',
                    'Vocational Training': 'Abilities cultivated through vocational education, apprenticeships, trade-specific training, and hands-on practice'
                }
            },
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
            entity: The entity name (knowledge/skill/ability/function/task)
            entity_type: Type of entity ('knowledge', 'skill', 'ability', 'function', 'task')

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
            with open(cache_path, 'rb') as f:
                self.stats['cache_hits'] += 1
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key: str, embeddings: List[np.ndarray]):
        """Save embeddings to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
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
            entity_type: Type of entity ('knowledge', 'skill', 'ability', 'function', 'task')

        Returns:
            Dictionary of reference embeddings for all attributes
        """
        logger.info(f"Generating reference embeddings for {entity_type}...")

        reference_embeddings = {}

        # Get all attribute types for this entity (e.g., 'type', 'basis' for knowledge/skill/ability)
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
            entity_type: Type of entity ('knowledge', 'skill', 'ability', 'function', 'task')

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
        entity_col: str
    ) -> pd.DataFrame:
        """
        Process a dataframe to add attribute columns (type, basis, physicality, etc.).

        Args:
            df: Input dataframe with occupation and entity columns
            entity_type: Type of entity ('knowledge', 'skill', 'ability', 'function', 'task')
            entity_col: Name of the entity column

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

        logger.info("=" * 70)

        # Generate reference embeddings
        reference_embeddings = self._generate_reference_embeddings(entity_type)

        # Create embedding texts
        logger.info(
            f"Creating embedding texts for {len(df)} {entity_type} items...")
        embedding_texts = [
            self._create_embedding_text(
                row['occupation'], row[entity_col], entity_type)
            for _, row in df.iterrows()
        ]

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

    def process_all(self, save_to_csv: bool = True) -> bool:
        """
        Process all entity types (knowledge, skills, abilities, functions, tasks) and add synthetic attributes.

        Args:
            save_to_csv: Whether to save results back to CSV files

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("SYNTHETIC ATTRIBUTE GENERATION - ALL ENTITY TYPES")
        logger.info("=" * 80)

        success = True  # Track overall success

        # Process knowledge
        logger.info("\n" + "=" * 80)
        logger.info("1. KNOWLEDGE")
        logger.info("=" * 80)
        knowledge_path = self.data_dir / "knowledge.csv"
        if knowledge_path.exists():
            knowledge_df = pd.read_csv(knowledge_path)
            knowledge_df = self.process_entity_dataframe(
                df=knowledge_df,
                entity_type='knowledge',
                entity_col='knowledge'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in knowledge_df.columns if '_confidence' not in col]
                knowledge_df[final_cols].to_csv(knowledge_path, index=False)
                logger.info(
                    f"✓ Saved updated knowledge.csv to {knowledge_path}")
        else:
            logger.error(f"Knowledge file not found: {knowledge_path}")
            success = False

        # Process skills
        logger.info("\n" + "=" * 80)
        logger.info("2. SKILLS")
        logger.info("=" * 80)
        skills_path = self.data_dir / "skills.csv"
        if skills_path.exists():
            skills_df = pd.read_csv(skills_path)
            skills_df = self.process_entity_dataframe(
                df=skills_df,
                entity_type='skill',
                entity_col='skill'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in skills_df.columns if '_confidence' not in col]
                skills_df[final_cols].to_csv(skills_path, index=False)
                logger.info(f"✓ Saved updated skills.csv to {skills_path}")
        else:
            logger.error(f"Skills file not found: {skills_path}")
            success = False

        # Process abilities
        logger.info("\n" + "=" * 80)
        logger.info("3. ABILITIES")
        logger.info("=" * 80)
        abilities_path = self.data_dir / "abilities.csv"
        if abilities_path.exists():
            abilities_df = pd.read_csv(abilities_path)
            abilities_df = self.process_entity_dataframe(
                df=abilities_df,
                entity_type='ability',
                entity_col='ability'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in abilities_df.columns if '_confidence' not in col]
                abilities_df[final_cols].to_csv(abilities_path, index=False)
                logger.info(
                    f"✓ Saved updated abilities.csv to {abilities_path}")
        else:
            logger.error(f"Abilities file not found: {abilities_path}")
            success = False

        # Process functions
        logger.info("\n" + "=" * 80)
        logger.info("4. FUNCTIONS")
        logger.info("=" * 80)
        functions_path = self.data_dir / "functions.csv"
        if functions_path.exists():
            functions_df = pd.read_csv(functions_path)
            functions_df = self.process_entity_dataframe(
                df=functions_df,
                entity_type='function',
                entity_col='function'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in functions_df.columns if '_confidence' not in col]
                functions_df[final_cols].to_csv(functions_path, index=False)
                logger.info(
                    f"✓ Saved updated functions.csv to {functions_path}")
        else:
            logger.error(f"Functions file not found: {functions_path}")
            success = False

        # Process tasks
        logger.info("\n" + "=" * 80)
        logger.info("5. TASKS")
        logger.info("=" * 80)
        tasks_path = self.data_dir / "tasks.csv"
        if tasks_path.exists():
            tasks_df = pd.read_csv(tasks_path)
            tasks_df = self.process_entity_dataframe(
                df=tasks_df,
                entity_type='task',
                entity_col='task'
            )

            if save_to_csv:
                # Keep only the final columns without confidence scores
                final_cols = [
                    col for col in tasks_df.columns if '_confidence' not in col]
                tasks_df[final_cols].to_csv(tasks_path, index=False)
                logger.info(f"✓ Saved updated tasks.csv to {tasks_path}")
        else:
            logger.error(f"Tasks file not found: {tasks_path}")
            success = False

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

        return success


def main():
    """Main function to run the synthetic attribute generator."""
    generator = SyntheticAttributeGenerator(
        model_name="models/gemini-embedding-001",
        batch_size=100,
        confidence_threshold=0.5
    )

    generator.process_all(save_to_csv=True)


if __name__ == "__main__":
    main()
