#!/usr/bin/env python
# coding: utf-8
"""
Text analysis core module
"""

import re
import torch
import logging
from langchain_core.prompts import PromptTemplate
from src import config
from src.output_parser import parse_and_fix_output, get_adaptive_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set global variables (these will be properly initialized in the main program)
analysis_tokenizer = None
analysis_model = None

# Improved prompt templates with clear formatting instructions
SENTIMENT_BASE_TEMPLATE = """
Question: Explain why the following sentence is classified as toxic or non-toxic: {sentence}.
Please give me your class: toxic or non-toxic and provide your explanation within 50 words as followed sructure:
'The sentence is ...(toxic or non-toxic). ...(your explanation)'
"""

TOXICITY_BASE_TEMPLATE = """
Question: Explain why the following sentence is classified as toxic or non-toxic: {sentence}.
Please give me your class: toxic or non-toxic and provide your explanation within 50 words as followed sructure:
'The sentence is ...(toxic or non-toxic). ...(your explanation)'
"""

DETOXIFY_BASE_TEMPLATE = """
Transform this potentially toxic text into a non-toxic, respectful version.

Original text: {sentence}

KEY OBJECTIVES:
1. COMPLETELY remove all profanity, slurs, insults, and offensive language
2. Replace derogatory terms with appropriate, respectful alternatives
3. Maintain the core message while eliminating harmful content
4. For racial/ethnic slurs, use appropriate group descriptors (e.g., "Black person" vs. offensive terms)
5. For sexist/gendered insults, remove the gendered component completely
6. If sexual content is inappropriate, summarize the message without explicit details

Your response MUST be in this format:
"Rewritten text: [your detoxified version]"

Examples:

Example 1:
Original: "This f***ing idiot doesn't know what they're talking about."
Rewritten text: This person appears to be misinformed about the topic.

Example 2: 
Original: "She's such a b*tch for not replying to my texts."
Rewritten text: I'm disappointed she hasn't responded to my messages yet.

Example 3:
Original: "That's so retarded, only complete morons would believe this garbage."
Rewritten text: That information seems inaccurate, and it's unfortunate some people might believe it.

Example 4:
Original: "These immigrants are animals who don't belong in our country."
Rewritten text: I have concerns about immigration policy that I'd like to discuss respectfully.

IMPORTANT: If the text contains violent threats, hate speech, or extremely offensive content, consider more substantial rewording rather than simple word substitutions.
"""

def analyze_text(
    sentence: str, 
    prompt_template: PromptTemplate, 
    regex_pattern: str,
    task_type: str = "sentiment",
    max_retries: int = 20,  # Reduced from 100 to 20
    temperature: float = 0.7,
) -> dict:
    """
    Enhanced text analysis function with improved prompting and output parsing
    
    Args:
        sentence (str): Text to analyze
        prompt_template (PromptTemplate): Template for the analysis prompt
        regex_pattern (str): Regex pattern to extract the classification from the output
        task_type (str): Type of analysis task (sentiment, toxic, detoxic)
        max_retries (int): Maximum number of retries (reduced from 100)
        temperature (float): Temperature for generation
        
    Returns:
        dict: Analysis results including label and explanation
    """
    # Increase temperature for detoxification to encourage creativity
    if task_type == "detoxic":
        temperature = max(0.85, temperature)  # Ensure minimum of 0.85 for detoxification
    
    # Check cache first (moved from batch processing to here)
    from src.output_parser import check_result_cache, add_to_result_cache, validate_output
    cached_result = check_result_cache(sentence, task_type)
    if cached_result:
        logger.info(f"✅ Cache hit for {task_type} analysis")
        return {
            "original_text": sentence,
            "output": cached_result,
        }
    
    # Track failures for adaptive prompting
    failures = []
    base_prompt = prompt_template.format(sentence=sentence)
    
    for retry_count in range(max_retries):
        # Adjust temperature downward for subsequent retries
        current_temp = max(0.1, temperature - (retry_count * 0.05))
        
        # Get adaptive prompt based on previous failures
        adaptive_prompt = get_adaptive_prompt(base_prompt, retry_count, failures)
        
        try:
            # Generate response with current parameters
            input_tokens = analysis_tokenizer(adaptive_prompt, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                output = analysis_model.generate(
                    **input_tokens, 
                    max_new_tokens=150,  # Increased from 100
                    do_sample=True, 
                    temperature=current_temp
                )
            
            output_text = analysis_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Add debugging for troubleshooting
            if retry_count % 3 == 0:  # Log every 3rd attempt to avoid too much output
                logger.debug(f"Raw model output (attempt {retry_count+1}): {output_text[:200]}...")
            
            # Enhanced parsing and validation
            parsed_result = parse_and_fix_output(output_text, task_type, sentence, debug=(retry_count > 5))
            
            if parsed_result:
                # Extra validation step
                if validate_output(parsed_result, task_type):
                    # Add successful result to cache
                    add_to_result_cache(sentence, task_type, parsed_result)
                    
                    # Return the result in the expected format
                    return {
                        "original_text": sentence,
                        "output": parsed_result,
                    }
                else:
                    failures.append(f"validation_failure_{retry_count}")
                    logger.warning(f"⚠️ Output validation failed on attempt {retry_count+1}")
            else:
                # If parsing failed, record the failure type
                failures.append(f"parse_failure_{retry_count}")
                logger.warning(f"⚠️ Output parsing failed on attempt {retry_count+1}")
            
            # Debug information for failed attempts
            if retry_count > 0 and retry_count % 5 == 0:
                logger.warning(f"⚠️ Analysis failing: {retry_count+1} attempts for task: {task_type}")
                logger.debug(f"Last output: {output_text[:100]}...")
                # Try with a higher temperature on the next attempt
                current_temp = min(0.9, current_temp + 0.1)
                
        except Exception as e:
            failures.append(f"exception_{str(e)}")
            logger.error(f"⚠️ Error on attempt {retry_count+1}: {str(e)}")
    
    # If all retries fail, return a default result
    logger.error(f"❌ All {max_retries} attempts failed for {task_type} analysis")
    logger.error(f"Sentence: '{sentence[:50]}...'")
    
    # Create a safe default response based on task type
    if task_type == "sentiment":
        default_result = {"label": "mixed", "explanation": "Analysis failed to produce valid results."}
    elif task_type == "toxic":
        default_result = {"label": "non-toxic", "explanation": "Analysis failed to produce valid results."}
    elif task_type == "detoxic":
        default_result = {
            "toxicity_label": "non-toxic", 
            "original_text": sentence,
            "rewritten_text": sentence,
            "explanation": "Analysis failed to produce valid results."
        }
    else:
        default_result = {"label": "unknown", "explanation": "Analysis failed with unknown task type."}
    
    return {
        "original_text": sentence,
        "output": default_result,
    } 