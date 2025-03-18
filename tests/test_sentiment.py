#!/usr/bin/env python
# coding: utf-8
"""
Test module for sentiment analysis functionality
"""

import os
import pandas as pd
import sys
import time
import warnings
from functools import wraps
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Only show error messages
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Add parent directory to path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src import config
from src import models
from src import language_utils
from src import analysis
from src import sentiment
from src.text_processor import preprocess_text, batch_process_texts

# Import test configuration - use relative import
from test_config import (
    TEST_SENTIMENT_DATA, 
    SENTIMENT_TEST_OUTPUT, 
    setup_test_logging
)

# Setup logger
logger = setup_test_logging()

def test_decorator(func):
    """
    Decorator to adapt production functions for testing
    - Adds logging for test runs
    - Times function execution
    - Handles exceptions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting test for {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Test {func.__name__} completed successfully in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Test {func.__name__} failed: {str(e)}")
            raise
            
    return wrapper

@test_decorator
def test_sentiment_analysis():
    """
    Test sentiment analysis on a small dataset
    Reuses the production code but with test data
    """
    # Create a test dataframe with our test sentences
    test_df = pd.DataFrame({
        "sentence_id": list(range(1, len(TEST_SENTIMENT_DATA) + 1)),
        "sentence": TEST_SENTIMENT_DATA,
        "class-label": ["positive", "negative", "mixed", "neutral", "negative"]  # Add ground truth labels
    })
    
    logger.info(f"Test dataframe has columns: {test_df.columns.tolist()}")
    
    # Preprocess test data - reusing production function
    logger.info("Preprocessing test data...")
    with tqdm(total=len(TEST_SENTIMENT_DATA), desc="Preprocessing", unit="text") as pbar:
        cleaned_texts = []
        for text in TEST_SENTIMENT_DATA:
            cleaned_texts.append(preprocess_text(text))
            pbar.update(1)
    
    logger.info(f"Preprocessed {len(cleaned_texts)} texts for sentiment test")
    
    # Process sentiment using the same function used in production
    logger.info("Running sentiment analysis...")
    sentiment_results = batch_process_texts(cleaned_texts, task_type="sentiment")
    logger.info(f"Completed sentiment analysis with {len(sentiment_results)} results")
    
    # Create results dataframe
    results_df = pd.DataFrame(sentiment_results)
    
    # Extract values from nested output structure
    extracted_results = []
    for result in sentiment_results:
        if isinstance(result, dict):
            if "output" in result and isinstance(result["output"], dict):
                # New structure: {"original_text": "...", "output": {"label": "...", "explanation": "..."}}
                output = result["output"]
                extracted_results.append({
                    "original_text": result.get("original_text", ""),
                    "label": output.get("label", "unknown"),
                    "explanation": output.get("explanation", "No explanation provided.")
                })
            elif "label" in result and "explanation" in result:
                # Old structure: {"label": "...", "explanation": "..."}
                extracted_results.append(result)
            else:
                # Fallback for unknown structure
                extracted_results.append({
                    "label": "unknown",
                    "explanation": "Failed to parse result structure."
                })
    
    # Create a new dataframe with the extracted values
    extracted_df = pd.DataFrame(extracted_results)
    
    # Merge with original dataframe to match standard format
    merged_df = pd.merge(
        test_df,
        extracted_df,
        left_index=True,
        right_index=True
    )
    
    # Ensure column order matches the standard format
    merged_df = merged_df[[
        "sentence_id", 
        "sentence", 
        "class-label",
        "label", 
        "explanation"
    ]]
    
    # Rename columns to match expected format
    merged_df = merged_df.rename(columns={"label": "predicted-label"})
    
    # Save the results
    try:
        # Clean text fields to handle newlines and quotes before saving
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':  # Only process string columns
                # Replace newlines with spaces
                merged_df[col] = merged_df[col].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
                # Handle quotes by ensuring proper escaping
                merged_df[col] = merged_df[col].apply(lambda x: x.replace('"', '\'') if isinstance(x, str) else x)
        
        # Save with quoting options to properly handle text fields
        merged_df.to_csv(SENTIMENT_TEST_OUTPUT, index=False, quoting=1)  # QUOTE_ALL mode
        logger.info(f"Saved sentiment test results to {SENTIMENT_TEST_OUTPUT}")
    except Exception as e:
        logger.error(f"Error saving sentiment test results: {e}")
        raise
        
    # Log the results for easier inspection
    logger.info("\n--- TEST RESULTS SUMMARY ---")
    sentiment_counts = merged_df['predicted-label'].value_counts()
    for label, count in sentiment_counts.items():
        logger.info(f"  {label}: {count} texts")
    logger.info("--- SAMPLE RESULTS ---")
    
    for i, row in merged_df.head(min(5, len(merged_df))).iterrows():
        logger.info(f"Result {i+1}: Text: '{row['sentence'][:30]}...' | Ground truth: {row['class-label']} | Predicted: {row['predicted-label']}")
        
    return merged_df

def initialize_models():
    """Initialize models needed for testing"""
    logger.info("Loading models for testing...")
    (
        language_utils.lang_detector, 
        language_utils.translation_tokenizer, language_utils.translation_model,
        analysis.analysis_tokenizer, analysis.analysis_model,
        _
    ) = models.load_models()
    logger.info("Models loaded successfully")

def main():
    """Main test function"""
    try:
        # Initialize models
        initialize_models()
        
        # Run sentiment test
        test_sentiment_analysis()
        
        logger.info("All sentiment tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 