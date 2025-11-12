import pandas as pd
import numpy as np
import json
import itertools
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# --- Conditional Import for API ---
import google.generativeai as genai
from dotenv import load_dotenv # Used for loading GOOGLE_API_KEY from .env

# --- 0. Taxonomy Definition (Copied from Source) ---
occupational_taxonomy = {
    'knowledge': {
        'basis': {
            'Academic': 'Theoretical knowledge typically acquired through formal education, university studies, textbooks, lectures, and academic research',
            'On-the-Job Training': 'Practical knowledge gained through direct workplace experience, mentorship, hands-on practice, and learning by doing',
            'Professional Development': 'Knowledge obtained through continuing education, professional certifications, workshops, seminars, and career advancement programs',
            'Vocational Training': 'Trade-specific knowledge acquired through vocational schools, apprenticeships, technical institutes, and specialized training programs'
        }
    },
    'skill': {
        'basis': {
            'Academic': 'Skills learned through formal education, theoretical study, academic coursework, and structured learning environments',
            'On-the-Job Training': 'Skills developed through workplace practice, real-world application, mentorship, and direct experience',
            'Professional Development': 'Skills enhanced through professional training, certifications, continuing education, and career development activities',
            'Vocational Training': 'Skills acquired through trade schools, apprenticeships, vocational programs, and hands-on technical training'
        }
    },
    'ability': {
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
        }
        ,
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

# --- 1. Environment & API Configuration ---

# Calculate project root path (assuming script is in src/backend/etl)
SCRIPT_PATH = Path(__file__).resolve()
# Go up 3 levels to reach the root: src/backend/etl -> backend/etl -> etl -> KSAMDS-Capstone/
PROJECT_ROOT = SCRIPT_PATH.parents[3]
# Directory where onet_extractor saves the intermediate CSVs
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "archive" / "intermediate"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "archive" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project Root: {PROJECT_ROOT}")
print(f"O*NET CSV Dir: {PROCESSED_DATA_DIR}")
print(f"Embeddings Cache Dir: {EMBEDDINGS_DIR}")

# Define cache file names based on the model used in the source code
MODEL_NAME = "models/embedding-001"
TAXONOMY_CACHE_FILE = EMBEDDINGS_DIR / f"taxonomy_embeddings_{MODEL_NAME.split('/')[-1]}.pkl"

def configure_api_and_model():
    """Configures the Gemini API using the GOOGLE_API_KEY from the .env file."""
    if genai is None:
        print("API dependency not found. Skipping API configuration.")
        return None

    # Load environment variables from the root .env file
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        print("FATAL ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        return None

    try:
        genai.configure(api_key=api_key)
        print(f"Google API configured successfully. Using embedding model: {MODEL_NAME}")
        return MODEL_NAME
    except Exception as e:
        print(f"FATAL ERROR: Could not configure Google API: {e}")
        return None

# --- 2. Embedding Helper Functions (Copied/Adapted from Source) ---

def embed_in_batches(texts: List[str], task_type: str, batch_size: int = 100) -> List[Optional[List[float]]]:
    """Embeds a list of texts in batches to avoid timeouts and rate limits."""
    if genai is None or not MODEL_NAME: return [None] * len(texts)

    all_embeddings = []
    num_batches = int(np.ceil(len(texts) / batch_size))

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(f"    Embedding batch {i // batch_size + 1} / {num_batches}...")
        try:
            result = genai.embed_content(
                model=MODEL_NAME, content=batch_texts, task_type=task_type
            )
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            print(f"    ERROR in batch {i // batch_size + 1}: {e}")
            all_embeddings.extend([None] * len(batch_texts))
        time.sleep(1.5) # Pause to avoid rate limits
    return all_embeddings

def embed_single_batch(texts: List[str], task_type: str) -> List[Optional[List[float]]]:
    """Embeds a single, small list of texts."""
    if genai is None or not MODEL_NAME: return [None] * len(texts)
    try:
        result = genai.embed_content(
            model=MODEL_NAME, content=texts, task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"  Error during embedding: {e}")
        return [None] * len(texts)

def pre_embed_taxonomy_feature(feature_dict: Dict[str, str]) -> Tuple[Optional[np.ndarray], List[str]]:
    """Helper to embed taxonomy definitions."""
    texts_to_embed = [f"{name}: {desc}" for name, desc in feature_dict.items()]
    names = list(feature_dict.keys())
    # Taxonomy items are the "documents" to be searched
    embeddings = embed_single_batch(texts_to_embed, "RETRIEVAL_DOCUMENT")

    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if not valid_indices: return (None, [])

    valid_embeddings = np.stack([embeddings[i] for i in valid_indices])
    valid_names = [names[i] for i in valid_indices]
    print(f"  Successfully embedded {len(valid_names)} items.")
    return (valid_embeddings, valid_names)

# --- 3. Main Embedding/Caching Logic for Taxonomy ---

def generate_and_cache_taxonomy_embeddings(model_is_configured: bool):
    """
    Checks cache, loads if present. If not present and API is configured,
    generates and caches the taxonomy embeddings.
    """
    global MODEL_NAME
    all_embeddings_ready = False
    results = {} # To hold the (embeds, names) tuples

    print("\n--- 3. Checking for Cached Taxonomy Embeddings ---")

    required_keys = ['k_basis', 's_basis', 'a_basis', 'f_phys', 'f_cog', 'f_env', 't_type', 't_mode', 't_env']

    if TAXONOMY_CACHE_FILE.exists():
        print(f"Loading cached taxonomy embeddings from {TAXONOMY_CACHE_FILE}...")
        try:
            with open(TAXONOMY_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)

            # Verification: Check if the cache contains all keys
            if all(key in cache_data for key in required_keys):
                for key in required_keys:
                    results[key] = cache_data[key]
                all_embeddings_ready = True
                print("Successfully loaded complete taxonomy embeddings from cache.")
            else:
                print("Cache file is incomplete. Will re-embed.")
                cache_data = None
        except Exception as e:
            print(f"Error loading cache: {e}. Will re-embed.")
            cache_data = None

    if not all_embeddings_ready:
        if not model_is_configured:
            print("Skipping embedding generation: API not configured or no model specified.")
            # Return empty but structured data to prevent downstream errors
            for key in required_keys:
                results[key] = (None, list(occupational_taxonomy.get(key[0:1], {}).get(key[2:], {}).keys())) # Fallback names
            return results, False

        try:
            print("Embedding Taxonomy Features...")
            # Structure for embedding knowledge, skill, ability basis
            for key in ['knowledge', 'skill', 'ability']:
                embeds, names = pre_embed_taxonomy_feature(occupational_taxonomy[key]['basis'])
                results[f'{key[0]}_basis'] = (embeds, names)

            # Structure for embedding function features
            for key in ['physicality', 'cognitive_load', 'environment']:
                embeds, names = pre_embed_taxonomy_feature(occupational_taxonomy['function'][key])
                results[f'f_{key.split("_")[0]}'] = (embeds, names)

            # Structure for embedding task features
            for key in ['mode', 'environment']:
                embeds, names = pre_embed_taxonomy_feature(occupational_taxonomy['task'][key])
                results[f't_{key}'] = (embeds, names)

            # Type is directly read, but included here for consistency if needed later
            results['t_type'] = (None, list(occupational_taxonomy['task']['type'].keys()))


            # Verification check for all required elements having non-None embeddings
            if not all(results[key][0] is not None for key in results if key != 't_type'):
                 raise Exception("Embedding failed for one or more critical taxonomy types.")

            # Save to cache
            print("\nCaching taxonomy embeddings to file...")
            cache_data = results
            with open(TAXONOMY_CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)

            all_embeddings_ready = True
            print("All taxonomy embeddings created and cached.")
        except Exception as e:
            print(f"FATAL ERROR during pre-embedding: {e}")
            for key in results:
                results[key] = (None, []) # Clear any partial results
            all_embeddings_ready = False

    return results, all_embeddings_ready

# --- 4. O*NET Data Loader ---

def load_onet_data(required_files: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads required O*NET data from CSVs generated by onet_extractor.py.
    """
    dataframes = {}
    print("\n--- 4. Loading O*NET Data from CSVs ---")

    for file_key in required_files:
        file_path = PROCESSED_DATA_DIR / f"{file_key}.csv"
        if file_path.exists():
            try:
                # Use low_memory=False for large files
                df = pd.read_csv(file_path, low_memory=False)
                dataframes[file_key] = df
                print(f"  Loaded {file_key}.csv ({len(df)} rows)")
            except Exception as e:
                print(f"  ERROR loading {file_key}.csv: {e}")
        else:
            print(f"  WARNING: Required file not found: {file_key}.csv at {file_path}")

    return dataframes

# --- 5. Main O*NET Data Query Embedding Logic ---

def generate_and_cache_onet_query_embeddings(model_is_configured: bool, dataframes: Dict[str, pd.DataFrame]):
    """
    Generates and caches the embeddings for all KSA, Function, and Task elements
    that will be used as queries in the downstream mapping process.
    """
    if not model_is_configured:
        print("Skipping O*NET query embedding generation: API not configured.")
        return False

    print("\n--- 5. Generating and Caching O*NET Query Embeddings ---")

    # Required files for lookups
    req_files = ['occupation_data', 'content_model_reference', 'task_statements', 'task_ratings',
                 'knowledge', 'skills', 'abilities', 'work_activities']
    if not all(f in dataframes for f in req_files):
        print("FATAL: Missing one or more required O*NET DataFrames to generate queries.")
        print(f"Missing keys: {set(req_files) - set(dataframes.keys())}")
        return False

    try:
        # Helper Lookups
        occupation_df = dataframes['occupation_data']
        code_to_title = pd.Series(occupation_df.Title.values, index=occupation_df['O*NET-SOC Code']).to_dict()
        code_to_desc = pd.Series(occupation_df.Description.values, index=occupation_df['O*NET-SOC Code']).to_dict()
        ref_df = dataframes['content_model_reference']
        el_name_to_desc = pd.Series(ref_df.Description.values, index=ref_df['Element Name']).to_dict()

        # Task Importance Lookup (used for filtering tasks)
        task_ratings_df = dataframes['task_ratings']
        task_im_lookup = task_ratings_df[task_ratings_df['Scale ID'] == 'IM'].set_index(
            ['O*NET-SOC Code', 'Task ID']
        )['Data Value'].to_dict()

    except KeyError as e:
        print(f"FATAL: Missing critical column for lookup: {e}")
        return False

    # --- KSA and Function Query Generation ---

    def _process_feature(df_key: str, entity_name: str, cache_prefix: str, feature_df: pd.DataFrame) -> bool:
        cache_file = EMBEDDINGS_DIR / f"{cache_prefix}_query_data_{MODEL_NAME.split('/')[-1]}.pkl"

        if cache_file.exists():
            print(f"  Skipping {entity_name}: Cache file already exists at {cache_file}")
            return True

        print(f"\n  Generating {entity_name} queries...")

        # Pivot to get Importance (IM) and Data Value LV columns side-by-side
        feature_pivot = feature_df.pivot_table(
            index=['O*NET-SOC Code', 'Element ID', 'Element Name'],
            columns='Scale ID', values='Data Value'
        ).reset_index().fillna(value=np.nan)

        query_texts, valid_rows_data = [], []
        # Use an index range to handle large DataFrames efficiently
        for index in range(len(feature_pivot)):
            row = feature_pivot.iloc[index]
            # Filter by Importance >= 3.0
            if row.get('IM', 0.0) >= 3.0:
                soc_code = row['O*NET-SOC Code']
                el_name = row['Element Name']
                title = code_to_title.get(soc_code)

                if title:
                    job_desc = code_to_desc.get(soc_code, '')
                    el_desc = el_name_to_desc.get(el_name, '')
                    # Create the query text as done in onet_occupation_features.py
                    query_texts.append(f"Job: {title}. {job_desc}. {entity_name}: {el_name}. {el_desc}")

                    # Store the relevant row data (which includes LV/IM) for the downstream mapping script
                    valid_rows_data.append(row.to_dict())

        print(f"  Embedding {len(query_texts)} {entity_name} queries in batches...")
        query_embeddings = embed_in_batches(query_texts, "RETRIEVAL_QUERY", batch_size=100)

        # Filter out rows where embedding failed
        final_embeddings = [emb for emb in query_embeddings if emb is not None]
        final_rows_data = [valid_rows_data[i] for i, emb in enumerate(query_embeddings) if emb is not None]

        if not final_embeddings:
            print(f"  WARNING: No valid embeddings generated for {entity_name}.")
            return False

        print(f"  Caching {len(final_embeddings)} valid {entity_name} queries and embeddings...")
        with open(cache_file, 'wb') as f:
            pickle.dump({'embeddings': final_embeddings, 'valid_rows_data': final_rows_data}, f)

        return True

    # Run for KSA and Functions (Work Activities)
    k_success = _process_feature('knowledge', 'Feature', 'knowledge', dataframes['knowledge'])
    s_success = _process_feature('skills', 'Feature', 'skills', dataframes['skills'])
    a_success = _process_feature('abilities', 'Feature', 'abilities', dataframes['abilities'])
    f_success = _process_feature('work_activities', 'Function', 'function', dataframes['work_activities'])

    # --- Task Query Generation (Uses different source DF) ---

    def _process_tasks() -> bool:
        entity_name = 'Task'
        cache_prefix = 'task'
        cache_file = EMBEDDINGS_DIR / f"{cache_prefix}_query_data_{MODEL_NAME.split('/')[-1]}.pkl"
        task_df = dataframes['task_statements']

        if cache_file.exists():
            print(f"  Skipping {entity_name}: Cache file already exists at {cache_file}")
            return True

        print(f"\n  Generating {entity_name} queries...")

        query_texts, valid_tasks_info = [], []
        # Iterate over Task Statements
        for index in range(len(task_df)):
            row = task_df.iloc[index]
            soc_code, task_id, task_text = row['O*NET-SOC Code'], row['Task ID'], row['Task']
            # Lookup Importance
            importance = task_im_lookup.get((soc_code, task_id), 0.0)

            if pd.notna(importance) and importance >= 3.0:
                title = code_to_title.get(soc_code)
                if title:
                    job_desc = code_to_desc.get(soc_code, '')
                    # Create the query text as done in onet_occupation_features.py
                    query_texts.append(f"Job: {title}. {job_desc}. Task: {task_text}")
                    # Store required info for downstream mapping: (title, task_text, task_type)
                    valid_tasks_info.append((title, task_text, row['Task Type']))

        print(f"  Embedding {len(query_texts)} {entity_name} queries in batches...")
        query_embeddings = embed_in_batches(query_texts, "RETRIEVAL_QUERY", batch_size=100)

        # Filter out rows where embedding failed
        final_embeddings = [emb for emb in query_embeddings if emb is not None]
        final_tasks_info = [valid_tasks_info[i] for i, emb in enumerate(query_embeddings) if emb is not None]

        if not final_embeddings:
            print(f"  WARNING: No valid embeddings generated for {entity_name}.")
            return False

        print(f"  Caching {len(final_embeddings)} valid {entity_name} queries and embeddings...")
        with open(cache_file, 'wb') as f:
            # Save task-specific metadata structure
            pickle.dump({'embeddings': final_embeddings, 'valid_tasks_info': final_tasks_info}, f)

        return True

    t_success = _process_tasks()

    return all([k_success, s_success, a_success, f_success, t_success])


# --- 6. Main Execution ---

if __name__ == "__main__":
    # 1. Configure API
    MODEL_NAME = configure_api_and_model()
    model_is_configured = MODEL_NAME is not None

    # 2. Generate/Load Taxonomy Embeddings
    embedding_results, taxonomy_success = generate_and_cache_taxonomy_embeddings(model_is_configured)

    if not taxonomy_success and model_is_configured:
        print("\nFATAL: Failed to generate/load critical Taxonomy Embeddings. Halting.")
    elif not model_is_configured:
        print("\nFATAL: API not configured. Cannot generate any embeddings. Halting.")
    else:
        # 3. Load O*NET Data
        required_csv_files = [
            'knowledge', 'skills', 'abilities', 'work_activities',
            'occupation_data', 'content_model_reference', 'task_ratings', 'task_statements'
        ]
        onet_dataframes = load_onet_data(required_csv_files)

        # 4. Generate/Cache O*NET Query Embeddings (KSA/Function/Task)
        query_success = generate_and_cache_onet_query_embeddings(model_is_configured, onet_dataframes)

        if query_success:
            print("\n--- Summary ---")
            print("SUCCESS: All Taxonomy and O*NET Query Embeddings generated and cached.")
            print(f"Cache location: {EMBEDDINGS_DIR}")
        else:
            print("\nFATAL: One or more O*NET Query Embedding caches failed to generate. Check logs.")