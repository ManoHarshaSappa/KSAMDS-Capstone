"""
Relationship Validator for KSAMDS Project
Uses a Large Language Model (Google's Gemini) to validate a sample of
inferred relationships from the embedding pipeline. This script is intended
to be run after embedding_relationship_builder.py has completed.
"""

import pandas as pd
from pathlib import Path
import google.generativeai as genai
import json
import time
import os
import logging

# --- Configuration ---
# Directory where the inferred relationship CSVs are located
INPUT_DIR = Path("data/processed")
# Directory to save the validated output CSVs
OUTPUT_DIR = Path("data/validated")
# The Generative AI model to use for validation
MODEL_NAME = 'models/gemini-2.5-flash'  # Cost-effective and fast
# Number of relationships to sample from each file for validation
SAMPLE_SIZE = 10
# --- End Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


# --- The Prompt ---
# A well-defined prompt is crucial for getting good results.
# We ask for a structured JSON output to make parsing easy.
PROMPT_TEMPLATE = """
You are an expert in job analysis, skills, and occupational data. Your task is to validate a semantic relationship inferred by an AI model.

Analyze the relationship between the Source Entity and the Target Entity below.

Source Type: {source_type}
Source Entity: "{source_entity}"

Target Type: {target_type}
Target Entity: "{target_entity}"

Based on your expert knowledge, is this a direct and meaningful relationship?

Respond only with a valid JSON object in the following format:
{{
  "verdict": "Related" or "Unrelated",
  "confidence": "High", "Medium", or "Low",
  "reasoning": "A brief, one-sentence explanation for your verdict."
}}
"""


def get_gemini_model():
    """Initializes and returns the Gemini GenerativeModel."""
    try:
        if 'GOOGLE_API_KEY' in os.environ:
            api_key = os.environ.get('GOOGLE_API_KEY')
            logger.info("Configuring Google AI with environment variable.")
        else:
            from google.colab import userdata
            api_key = userdata.get('GOOGLE_API_KEY')
            logger.info("Configuring Google AI with Colab userdata.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        logger.error("Please ensure your GOOGLE_API_KEY is set correctly.")
        return None


def validate_relationship(source_entity, target_entity, source_type, target_type, model):
    """Calls the Gemini API to validate a single relationship."""
    try:
        prompt = PROMPT_TEMPLATE.format(
            source_type=source_type.capitalize(),
            source_entity=source_entity,
            target_type=target_type.capitalize(),
            target_entity=target_entity
        )
        response = model.generate_content(prompt)

        # Clean the response and parse the JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        return result
    except Exception as e:
        logger.error(
            f"  - Error processing '{source_entity}' -> '{target_entity}': {e}")
        return {
            "verdict": "Error",
            "confidence": "N/A",
            "reasoning": str(e)
        }


def run_validation():
    """Loops through inferred CSVs, validates a sample, and saves the results."""
    logger.info("Starting relationship validation process...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {OUTPUT_DIR}")

    model = get_gemini_model()
    if not model:
        return

    files_to_process = [
        "knowledge_skill_inferred.csv",
        "skill_ability_inferred.csv",
        "knowledge_function_inferred.csv",
        "ability_task_inferred.csv",
        "function_task_inferred.csv"
    ]

    for filename in files_to_process:
        logger.info("-" * 60)
        logger.info(f"Processing file: {filename}...")

        input_path = INPUT_DIR / filename
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}. Skipping.")
            continue

        try:
            df = pd.read_csv(input_path)

            # Use the smaller of the sample size or the dataframe length
            actual_sample_size = min(SAMPLE_SIZE, len(df))
            if actual_sample_size == 0:
                logger.info("File is empty. Skipping.")
                continue

            df_sample = df.sample(n=actual_sample_size, random_state=42)
            logger.info(
                f"Loaded {len(df)} rows. Validating a random sample of {len(df_sample)} rows.")

            results = []
            source_type, target_type = filename.replace(
                "_inferred.csv", "").split('_')

            for index, row in df_sample.iterrows():
                source_entity = row['source_name']
                target_entity = row['target_name']

                logger.info(
                    f"  - Validating: '{source_entity}' ({source_type}) -> '{target_entity}' ({target_type})")

                # Call the validation function
                validation_result = validate_relationship(
                    source_entity, target_entity, source_type, target_type, model)

                # Combine original data with validation results
                full_result = {
                    'source_name': source_entity,
                    'target_name': target_entity,
                    'confidence_score': row['confidence_score'],
                    'llm_verdict': validation_result.get('verdict'),
                    'llm_confidence': validation_result.get('confidence'),
                    'llm_reasoning': validation_result.get('reasoning')
                }
                results.append(full_result)

                # Respect API rate limits
                time.sleep(2)

            # Save the results
            results_df = pd.DataFrame(results)
            output_filename = OUTPUT_DIR / \
                filename.replace("_inferred.csv", "_validated.csv")
            results_df.to_csv(output_filename, index=False)
            logger.info(
                f"✅ Validation complete. Results saved to: {output_filename}")

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while processing {filename}: {e}", exc_info=True)

    logger.info("-" * 60)
    logger.info("All files processed. Validation run finished.")


if __name__ == "__main__":
    run_validation()
