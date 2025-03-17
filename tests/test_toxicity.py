#!/usr/bin/env python
# coding: utf-8
"""
Test module for toxicity analysis and detoxification functionality
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
from src import utils
from src import analysis
from src import toxicity
from src.processing import preprocess_text, batch_process_texts

# Import test configuration - use relative import
sys.path.append(current_dir)
from test_config import (
    TEST_TOXICITY_DATA, 
    TOXIC_TEST_OUTPUT, 
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
def test_toxicity_analysis():
    """
    Test toxicity analysis and detoxification on a small dataset
    Reuses the production code but with test data
    """
    # Create a test dataframe with our test texts - include all standard columns
    test_df = pd.DataFrame({
        "data_id": list(range(101, 101 + len(TEST_TOXICITY_DATA))),
        "sample_id": list(range(len(TEST_TOXICITY_DATA))),
        "text": TEST_TOXICITY_DATA,
        "source_label": ["toxic", "non-toxic", "toxic", "non-toxic", "toxic"],  # Add ground truth labels
        "platform": ["twitter", "facebook", "reddit", "youtube", "forum"]  # Sample platforms
    })
    
    logger.info(f"Test dataframe has columns: {test_df.columns.tolist()}")
    
    # Preprocess test data - reusing production function
    logger.info("Preprocessing test data...")
    with tqdm(total=len(TEST_TOXICITY_DATA), desc="Preprocessing", unit="text") as pbar:
        cleaned_texts = []
        for text in TEST_TOXICITY_DATA:
            cleaned_texts.append(preprocess_text(text))
            pbar.update(1)
    
    logger.info(f"Preprocessed {len(cleaned_texts)} texts for toxicity test")
    
    # First test simple toxicity detection
    logger.info("Testing toxicity detection...")
    toxicity_results = []
    with tqdm(total=len(cleaned_texts), desc="Toxicity Detection", unit="text") as pbar:
        for text in cleaned_texts:
            # Use the production toxicity analysis function
            result = toxicity.analyze_toxic(text)
            toxicity_results.append(result["output"])
            pbar.update(1)
        
    # Create results dataframe for toxicity analysis
    toxicity_df = pd.DataFrame({
        "original_text": TEST_TOXICITY_DATA,
        "toxicity_label": [r["label"] for r in toxicity_results],
        "explanation": [r["explanation"] for r in toxicity_results]
    })
    
    # Then test detoxification
    logger.info("Testing detoxification...")
    detox_results = batch_process_texts(cleaned_texts, task_type="detoxic")
    logger.info(f"Completed toxicity analysis with {len(detox_results)} results")
    
    # Create results dataframe for detoxification
    detox_df = pd.DataFrame(detox_results)
    
    # Add original text for clarity in test results
    detox_df['original_text'] = TEST_TOXICITY_DATA
    
    # Merge both test results
    final_df = pd.merge(
        toxicity_df[['original_text', 'toxicity_label', 'explanation']], 
        detox_df[['original_text', 'rewritten_text']],
        on='original_text'
    )
    
    # 添加标准格式的源标签和平台信息
    final_df['data_id'] = test_df['data_id'].values
    final_df['sample_id'] = test_df['sample_id'].values
    final_df['source_label'] = test_df['source_label'].values
    final_df['platform'] = test_df['platform'].values
    
    # 重命名列以符合标准格式
    final_df = final_df.rename(columns={
        'explanation': 'toxicity_explanation',
        'original_text': 'text',
        'toxicity_label': 'predicted_label'
    })
    
    # 按标准顺序重新排列列
    final_df = final_df[[
        'data_id', 'sample_id', 'text', 'source_label', 
        'platform', 'predicted_label', 'rewritten_text'
    ]]
    
    # Save the results
    try:
        # Clean text fields to handle newlines and quotes before saving
        for col in final_df.columns:
            if final_df[col].dtype == 'object':  # Only process string columns
                # Replace newlines with spaces
                final_df[col] = final_df[col].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
                # Handle quotes by ensuring proper escaping
                final_df[col] = final_df[col].apply(lambda x: x.replace('"', '\'') if isinstance(x, str) else x)
        
        # Save with quoting options to properly handle text fields
        final_df.to_csv(TOXIC_TEST_OUTPUT, index=False, quoting=1)  # QUOTE_ALL mode
        logger.info(f"Saved toxicity test results to {TOXIC_TEST_OUTPUT}")
    except Exception as e:
        logger.error(f"Error saving toxicity test results: {e}")
        raise
        
    # Log the results for easier inspection
    logger.info("\n--- TEST RESULTS SUMMARY ---")
    toxicity_counts = final_df['predicted_label'].value_counts()
    for label, count in toxicity_counts.items():
        logger.info(f"  {label}: {count} texts")
    logger.info("--- SAMPLE RESULTS ---")
    
    for i, row in final_df.head(min(5, len(final_df))).iterrows():
        logger.info(f"Result {i+1}: Text: '{row['text'][:30]}...' | Ground truth: {row['source_label']} | Predicted: {row['predicted_label']}")
        if row['predicted_label'] == 'toxic':
            logger.info(f"  Detoxified: '{row['rewritten_text'][:30]}...'")
        
    return final_df

def initialize_models():
    """Initialize models needed for testing"""
    logger.info("Loading models for testing...")
    (
        utils.lang_detector, 
        utils.translation_tokenizer, utils.translation_model,
        analysis.analysis_tokenizer, analysis.analysis_model,
        _
    ) = models.load_models()
    logger.info("Models loaded successfully")

def main():
    """Main test function"""
    try:
        # Initialize models
        initialize_models()
        
        # Run toxicity test
        test_toxicity_analysis()
        
        logger.info("All toxicity tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 