#!/usr/bin/env python
# coding: utf-8
"""
Text processing module for preprocessing and batch processing of text data
"""

import re
import time
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from src import config
from src import language_utils
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


def process_with_ollama(text: str, task_type: str, ollama_model) -> dict:
    """
    Process text using Ollama model for more sophisticated analysis
    
    Args:
        text (str): Text to process
        task_type (str): Type of task ('sentiment', 'toxic', 'detoxic')
        ollama_model: The langchain Ollama model
        
    Returns:
        dict: Processing results
    """
    from src import analysis
    from src.output_parser import parse_and_fix_output
    
    # Get the appropriate template for the task
    if task_type == "sentiment":
        template = analysis.SENTIMENT_BASE_TEMPLATE
    elif task_type == "toxic":
        template = analysis.TOXICITY_BASE_TEMPLATE
    else:  # detoxic
        template = analysis.DETOXIFY_BASE_TEMPLATE
    
    # Format template with text
    prompt = template.format(sentence=text)
    
    # Invoke Ollama model
    try:
        response = ollama_model.invoke(prompt)
        
        # Parse response
        content = response.content if hasattr(response, 'content') else str(response)
        result = parse_and_fix_output(content, task_type, text)
        
        if result:
            logger.debug(f"✅ Ollama processed {task_type} successfully")
            return {"original_text": text, "output": result}
        else:
            logger.debug(f"⚠️ Ollama failed to produce valid {task_type} output")
            return None
            
    except Exception as e:
        logger.error(f"Error using Ollama for {task_type}: {e}")
        return None


def batch_process_texts(texts: List[str], task_type: str, max_retries: int = None) -> List[Dict[str, Any]]:
    """
    Process a batch of texts for sentiment analysis or toxicity detection/detoxification
    with enhanced reliability through automatic retries and error handling.
    
    Args:
        texts (List[str]): List of text samples to process
        task_type (str): Type of task ('sentiment', 'toxic', 'detoxic')
        max_retries (int): Maximum number of retries per text sample
        
    Returns:
        List[Dict[str, Any]]: List of results for each text
    """
    if not texts:
        logger.warning("No texts provided for processing")
        return []
    
    # Validate task type
    if task_type not in config.VALID_TASKS:
        raise ValueError(f"Invalid task type: {task_type}. Must be one of {config.VALID_TASKS}")
    
    # Set up Ollama model for fallback if enabled
    ollama_model = None
    if config.ENABLE_OLLAMA_FALLBACK:
        try:
            from langchain_ollama import ChatOllama
            ollama_model = ChatOllama(
                model=config.OLLAMA_MODEL_NAME,
                base_url=config.OLLAMA_HOST,
                temperature=0.7
            )
            logger.info(f"Initialized Ollama fallback model: {config.OLLAMA_MODEL_NAME}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama model for fallback: {e}")
    
    # Process each text with progress tracking
    results = []
    success_count = 0
    
    # Track progress with tqdm
    with tqdm(total=len(texts), desc=f"Processing {task_type}", unit="text") as pbar:
        for i, text in enumerate(texts):
            # Skip empty texts
            if not text or not isinstance(text, str):
                logger.warning(f"Skipping invalid text at index {i}")
                results.append({"original_text": "", "output": {"error": "Invalid input"}})
                pbar.update(1)
                continue
                
            # Process the current text with retries
            try:
                # Use the enhanced process_text function
                result = _process_text(text, task_type, max_retries)
                
                # If result is valid, add it to results
                if result:
                    results.append(result)
                    success_count += 1
                    pbar.set_description(f"Processing {task_type} (Success: {success_count}/{i+1})")
                    pbar.update(1)
                    continue
                    
                # If primary processing failed and Ollama fallback is enabled
                if ollama_model and config.ENABLE_OLLAMA_FALLBACK:
                    logger.info(f"Trying Ollama fallback for text {i+1}")
                    ollama_result = process_with_ollama(text, task_type, ollama_model)
                    
                    if ollama_result:
                        logger.info(f"Ollama fallback succeeded for text {i+1}")
                        results.append(ollama_result)
                        success_count += 1
                        pbar.set_description(f"Processing {task_type} (Success: {success_count}/{i+1})")
                        pbar.update(1)
                        continue
                
                # If we get here, both main process and Ollama failed
                logger.warning(f"All processing methods failed for text {i+1}")
                results.append(_get_default_result(text, task_type))
                
            except Exception as e:
                logger.error(f"Unexpected error processing text {i+1}: {str(e)}")
                results.append(_get_default_result(text, task_type))
                
            pbar.update(1)
    
    logger.info(f"Completed {task_type} processing: {success_count}/{len(texts)} succeeded")
    return results


def _get_default_result(text: str, task_type: str) -> Dict[str, Any]:
    """
    Get a default result for a given task type when processing fails
    
    Args:
        text (str): Original input text
        task_type (str): Type of task ('sentiment', 'toxic', 'detoxic')
        
    Returns:
        Dict[str, Any]: Default result with appropriate structure for the task
    """
    if task_type == "sentiment":
        return {
            "label": "mixed",
            "explanation": "Failed to analyze sentiment."
        }
    elif task_type == "toxic":
        return {
            "label": "non-toxic", 
            "explanation": "Failed to analyze toxicity."
        }
    elif task_type == "detoxic":
        # For detoxification, return original text as default
        # This ensures we don't have placeholder text in results
        return {
            "toxicity_label": "toxic",  # Mark as toxic to be safe
            "explanation": "Failed to properly detoxify text.",
            "original_text": text,
            "rewritten_text": text  # Use original text instead of placeholder
        }
    else:
        return {
            "error": f"Unknown task type: {task_type}",
            "original_text": text
        }


def _process_text(text: str, task_type: str, max_retries: int = None) -> dict:
    """
    Process a single text for the specified task with enhanced reliability
    
    Args:
        text (str): Text to process
        task_type (str): Type of task ('sentiment', 'toxic', 'detoxic')
        max_retries (int): Maximum number of retries (default from config)
        
    Returns:
        dict: Processing results
    """
    # Import here to avoid circular imports
    from src import sentiment
    from src import toxicity
    from src.processing import rule_based_detoxify
    
    # Set default retries if not specified
    if max_retries is None:
        max_retries = config.MAX_PROCESS_RETRIES
    
    # Track retries
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Call appropriate function based on task type
            if task_type == "sentiment":
                result = sentiment.analyze_sentiment(text)
            elif task_type == "toxic":
                result = toxicity.analyze_toxic(text)
            elif task_type == "detoxic":
                result = toxicity.detoxify_text(text)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            # If we got a result, return it
            if result:
                return result
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Error processing text (attempt {retry_count+1}/{max_retries}): {last_error}")
            
        # Increment retry counter
        retry_count += 1
        
        # Add a small random delay before retrying to avoid rate limits
        delay = random.uniform(*config.RANDOM_DELAY_RANGE)
        time.sleep(delay)
    
    # If we get here, all retries failed
    logger.error(f"Failed to process text after {max_retries} attempts. Last error: {last_error}")
    
    # For detoxification, try rule-based fallback as a last resort
    if task_type == "detoxic":
        try:
            logger.warning("Using rule-based fallback for detoxification after all retries failed")
            detoxified = rule_based_detoxify(text)
            
            if detoxified != text:
                return {
                    "original_text": text,
                    "output": {
                        "toxicity_label": "toxic",
                        "explanation": "Detoxified using rule-based fallback after all retries failed",
                        "rewritten_text": detoxified
                    }
                }
        except Exception as fallback_error:
            logger.error(f"Rule-based fallback also failed: {fallback_error}")
    
    # Return default results for all task types
    return _get_default_result(text, task_type) 