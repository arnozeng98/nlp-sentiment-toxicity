#!/usr/bin/env python
# coding: utf-8
"""
Toxicity analysis and detoxification module
"""

import re
import pandas as pd
import logging
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
import torch

from src import config
from src import analysis
from src.text_processor import batch_process_texts, preprocess_text
from src.data_processor import process_toxicity_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_toxic(sentence: str) -> dict:
    """
    Analyze whether a text is toxic or non-toxic
    
    Args:
        sentence (str): Text to analyze for toxicity
        
    Returns:
        dict: Analysis results with toxicity label (toxic/non-toxic) and explanation
    """
    # Handle input if it's a list (sometimes happens with certain tool invocations)
    if isinstance(sentence, list):
        sentence = sentence[0]
        
    # Use the direct toxicity analysis function
    return analyze_toxic_direct(sentence)

def analyze_toxic_direct(sentence: str, temperature: float = 0.7, max_retries: int = 10) -> dict:
    """
    Direct toxicity analysis function to ensure proper toxicity analysis and extraction
    
    Args:
        sentence (str): Text to analyze for toxicity
        temperature (float): Temperature for model generation
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: Analysis results with toxicity label and explanation
    """
    # Handle input if it's a list (sometimes happens with certain tool invocations)
    if isinstance(sentence, list):
        sentence = sentence[0]
        
    # Check cache first
    from src.output_parser import check_result_cache, add_to_result_cache, extract_toxicity_info
    cached_result = check_result_cache(sentence, "toxic")
    if cached_result:
        logger.info(f"✅ Cache hit for toxicity analysis")
        return {
            "original_text": sentence,
            "output": cached_result,
        }
    
    # Prepare toxicity prompt
    toxic_prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=analysis.TOXICITY_BASE_TEMPLATE
    )
    prompt = toxic_prompt_template.format(sentence=sentence)
    
    for retry_count in range(max_retries):
        try:
            # Adjust temperature slightly for each retry
            current_temp = max(0.2, temperature - (retry_count * 0.1))
            
            # Generate response
            input_tokens = analysis.analysis_tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                output = analysis.analysis_model.generate(
                    **input_tokens, 
                    max_new_tokens=150,
                    do_sample=True, 
                    temperature=current_temp
                )
            
            output_text = analysis.analysis_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract toxicity information directly
            label, explanation = extract_toxicity_info(output_text)
            
            if label and explanation:
                # Format the result
                result = {"label": label, "explanation": explanation}
                
                # Add to cache
                add_to_result_cache(sentence, "toxic", result)
                
                return {
                    "original_text": sentence,
                    "output": result
                }
            else:
                logger.warning(f"⚠️ Toxicity extraction failed on attempt {retry_count+1}")
                
                # Log output for debugging
                if retry_count >= 2:
                    logger.debug(f"Raw output: {output_text[:200]}...")
        
        except Exception as e:
            logger.error(f"⚠️ Error during toxicity analysis (attempt {retry_count+1}): {str(e)}")
    
    # If all retries fail
    logger.error(f"❌ All {max_retries} attempts failed for toxicity analysis")
    logger.error(f"Sentence: '{sentence[:50]}...'")
    
    default_result = {"label": "non-toxic", "explanation": "Analysis failed to produce valid results."}
    
    return {
        "original_text": sentence,
        "output": default_result
    }

def detoxify_text(sentence: str) -> dict:
    """
    Detoxify a text that may contain toxic content
    
    Args:
        sentence (str): Text to detoxify
        
    Returns:
        dict: Detoxification results with toxicity label and rewritten text
    """
    # Create direct implementation for detoxification
    return detoxify_text_direct(sentence)
    
def detoxify_text_direct(sentence: str, temperature: float = 0.85, max_retries: int = 10) -> dict:
    """
    Direct implementation for text detoxification
    
    Args:
        sentence (str): Text to detoxify
        temperature (float): Temperature for generation
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: Detoxification results with toxicity label and rewritten text
    """
    # Handle list input if it occurs
    if isinstance(sentence, list):
        sentence = sentence[0]
    
    # Import required functions
    from src.output_parser import check_result_cache, add_to_result_cache, extract_toxicity_info, extract_detoxified_text, clean_text_for_csv
    
    # Check cache first
    cached_result = check_result_cache(sentence, "detoxic")
    if cached_result:
        logger.info(f"✅ Cache hit for detoxification")
        return {
            "original_text": sentence,
            "output": cached_result,
        }
    
    # Prepare prompt
    detox_prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=analysis.DETOXIFY_BASE_TEMPLATE
    )
    prompt = detox_prompt_template.format(sentence=sentence)
    
    for retry_count in range(max_retries):
        try:
            # Use higher temperature for detoxification to encourage creative rewrites
            current_temp = max(0.7, temperature - (retry_count * 0.05)) 
            
            # Generate response
            input_tokens = analysis.analysis_tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                output = analysis.analysis_model.generate(
                    **input_tokens, 
                    max_new_tokens=200,  # Longer output for detoxification
                    do_sample=True, 
                    temperature=current_temp
                )
            
            output_text = analysis.analysis_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # First check toxicity
            toxic_label, explanation = extract_toxicity_info(output_text)
            
            # If no label found, default to non-toxic to avoid unnecessary rewriting
            if not toxic_label:
                toxic_label = "non-toxic"
                explanation = "No explicit toxicity detected."
            
            # Extract detoxified text
            if toxic_label == "toxic":
                detoxified = extract_detoxified_text(output_text, sentence)
                
                if detoxified:
                    # Format and cache the successful result
                    result = {
                        "toxicity_label": toxic_label,
                        "original_text": clean_text_for_csv(sentence),
                        "rewritten_text": clean_text_for_csv(detoxified),
                        "explanation": clean_text_for_csv(explanation)
                    }
                    
                    # Add to cache
                    add_to_result_cache(sentence, "detoxic", result)
                    
                    return {
                        "original_text": sentence,
                        "output": result
                    }
                else:
                    # If extraction failed, log and continue to next retry
                    logger.warning(f"⚠️ Failed to extract detoxified text on attempt {retry_count+1}")
                    
                    # Log the raw output for debugging
                    if retry_count >= 2:
                        logger.debug(f"Raw output: {output_text[:200]}...")
            else:
                # If content is already non-toxic, return it as is
                result = {
                    "toxicity_label": toxic_label,
                    "original_text": clean_text_for_csv(sentence),
                    "rewritten_text": clean_text_for_csv(sentence),
                    "explanation": clean_text_for_csv(explanation)
                }
                
                # Add to cache
                add_to_result_cache(sentence, "detoxic", result)
                
                return {
                    "original_text": sentence,
                    "output": result
                }
                
        except Exception as e:
            logger.error(f"⚠️ Error during detoxification (attempt {retry_count+1}): {str(e)}")
    
    # If all retries fail, return the original text with a warning
    logger.error(f"❌ All {max_retries} attempts failed for detoxification")
    logger.error(f"Sentence: '{sentence[:50]}...'")
    
    default_result = {
        "toxicity_label": "unknown",
        "original_text": clean_text_for_csv(sentence),
        "rewritten_text": clean_text_for_csv(sentence),
        "explanation": "Failed to detoxify text after multiple attempts."
    }
    
    return {
        "original_text": sentence,
        "output": default_result
    }

# Note: The process_toxicity_dataset function is now imported from data_processor
# and only kept here for backwards compatibility 