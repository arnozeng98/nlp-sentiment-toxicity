#!/usr/bin/env python
# coding: utf-8
"""
Text preprocessing and batch processing module
"""

import re
import time
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from src import config
from src import utils
from src.output_parser import check_result_cache, add_to_result_cache, validate_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum number of consecutive failures before skipping an item
MAX_CONSECUTIVE_FAILURES = 3

def preprocess_text(text: str) -> str:
    """
    Preprocess input text to clean and standardize format
    
    Args:
        text (str): Raw input text to preprocess
        
    Returns:
        str: Clean, standardized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove leading numbers, spaces, and punctuation
    text = re.sub(r'^[\s\d\W]+', '', text)
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Return the cleaned and formatted text
    return text.strip()


def batch_process_texts(texts: List[str], task_type: str, max_retries: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced batch processing with improved error handling and caching
    
    Args:
        texts (list): List of texts to be processed
        task_type (str): Task type, must be one of 'toxic', 'sentiment', or 'detoxic'
        max_retries (int, optional): Maximum number of retries in case of failure. Defaults to 20.
        
    Returns:
        list: List of processed results
    """
    if max_retries is None:
        max_retries = 20  # Reduced from 100 to 20
        
    results = []
    error_count = 0
    success_count = 0
    cache_hits = 0
    start_time = time.time()
    input_count = len(texts)
    output_count = 0

    # Validate task type
    if task_type not in config.VALID_TASKS:
        raise ValueError(f"Task type must be one of {config.VALID_TASKS}")

    # Dynamically import the needed processing function
    if task_type == "toxic":
        from src.toxicity import analyze_toxic as selected_function
    elif task_type == "sentiment":
        from src.sentiment import analyze_sentiment as selected_function
    elif task_type == "detoxic":
        from src.toxicity import detoxify_text as selected_function
    else:
        raise ValueError(f"Unknown task type: {task_type}")
        
    logger.info(f"Starting batch processing of {input_count} texts for {task_type}")

    # Create progress bar with additional stats - ensure it works in all environments
    # Set dynamic_ncols=True to adapt to terminal width
    progress_bar = tqdm(
        total=input_count,
        desc=f"Processing {task_type}",
        unit="text",
        dynamic_ncols=True,
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process each text with detailed progress tracking
    for i, text in enumerate(texts):
        # Update progress description with current stats - force refresh
        if i > 0:  # Only update after processing at least one item
            progress_desc = f"{task_type.capitalize()} [✓:{success_count} ⚠:{error_count} 💾:{cache_hits}]"
            progress_bar.set_description(progress_desc)
            progress_bar.refresh()
        
        if not text or text.strip() == "":
            # Handle empty text
            default_result = _get_default_result(text, task_type)
            results.append(default_result)
            # Silent logging for empty text - only update metrics
            error_count += 1
            output_count += 1
            progress_bar.update(1)
            continue
            
        # Check cache first
        cached_result = check_result_cache(text, task_type)
        if cached_result:
            results.append(cached_result)
            cache_hits += 1
            success_count += 1
            output_count += 1
            progress_bar.update(1)
            continue
            
        # Add a random delay to prevent excessive requests, with exponential backoff for errors
        backoff_factor = min(error_count, 5)  # Cap at 5 to avoid excessive delays
        delay = random.uniform(*config.RANDOM_DELAY_RANGE) * (1.5 ** backoff_factor)
        time.sleep(delay)

        # Perform language detection and translation if needed
        translated_text = text
        try:
            if utils.detect_language(text) != "en":
                translated_text = utils.translate_to_english(text)
        except Exception as e:
            # Silent logging for translation error - only log to file
            logger.debug(f"Translation error for item {i+1}: {str(e)}")
            # Continue with original text if translation fails
            
        # Track consecutive failures for this item
        consecutive_failures = 0
        result = None
        
        # Process with improved retry mechanism
        for retry_count in range(max_retries):
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # Silent logging for exceeded consecutive failures
                logger.debug(f"Skipping item {i+1} after {consecutive_failures} consecutive failures")
                break
                
            try:
                # Process the text
                result = selected_function(translated_text)
                
                # Validate the result format
                if "output" in result and validate_output(result["output"], task_type):
                    # Add to cache for future
                    add_to_result_cache(text, task_type, result["output"])
                    results.append(result["output"])
                    success_count += 1
                    output_count += 1
                    # No console output here, just update metrics
                    break
                else:
                    consecutive_failures += 1
                    # Silent logging for invalid result format
                    logger.debug(f"Invalid result format for item {i+1} on attempt {retry_count+1}")
                    # Short delay before retry
                    time.sleep(0.5)
                    
            except Exception as e:
                consecutive_failures += 1
                # Silent logging for processing error
                logger.debug(f"Error processing item {i+1} on attempt {retry_count+1}: {str(e)}")
                # Longer delay after errors
                time.sleep(1.0)

        # If max retries are reached or consecutive failures limit hit
        if result is None or "output" not in result or not validate_output(result["output"], task_type):
            # Use default result as fallback
            default_result = _get_default_result(text, task_type)
            results.append(default_result)
            error_count += 1
            output_count += 1
            # Silent logging for failed processing
            logger.debug(f"Failed to process item {i+1} after {max_retries} attempts. Using default result.")
        
        # Update progress bar after processing this item (whether successful or not)
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    # Log processing summary
    duration = time.time() - start_time
    avg_time = duration / input_count if input_count > 0 else 0
    
    # Generate concise statistics summary
    logger.info("=" * 50)
    logger.info("Processing Summary")
    logger.info("=" * 50)
    logger.info(f"Input count:    {input_count}")
    logger.info(f"Output count:   {output_count}")
    logger.info(f"Success count:  {success_count}")
    logger.info(f"Failure count:  {error_count}")
    logger.info(f"Cache hits:     {cache_hits}")
    logger.info(f"Processing time: {duration:.2f} seconds ({avg_time:.2f} seconds per item)")
    logger.info("=" * 50)

    return results


def _get_default_result(text: str, task_type: str) -> Dict[str, Any]:
    """
    Generate a default result for a given task type
    
    Args:
        text (str): Original text
        task_type (str): Type of task
        
    Returns:
        Dict[str, Any]: Default result
    """
    if task_type == "sentiment":
        return {"label": "mixed", "explanation": "Analysis failed to produce valid results."}
    elif task_type == "toxic":
        return {"label": "non-toxic", "explanation": "Analysis failed to produce valid results."}
    elif task_type == "detoxic":
        return {
            "toxicity_label": "non-toxic",
            "original_text": text,
            "rewritten_text": text,
            "explanation": "Analysis failed to produce valid results."
        }
    else:
        return {"label": "unknown", "explanation": "Unknown task type."} 