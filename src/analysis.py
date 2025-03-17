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
Question: Analyze the sentiment of the following text and classify it as positive, negative, or mixed.

TEXT: "{sentence}"

INSTRUCTIONS:
1. Consider the overall emotion, tone, and content.
2. Classify as:
   - "positive" if it expresses positive emotions, approval, or satisfaction
   - "negative" if it expresses negative emotions, complaints, or dissatisfaction
   - "mixed" if it contains both positive and negative elements
3. IMPORTANT: The output MUST follow exactly this format:
   "The sentence is [CLASSIFICATION]. [YOUR BRIEF EXPLANATION]"
4. Do NOT include any ending markers like "END OF ANALYSIS" or similar phrases.
5. Do NOT include any additional formatting or metadata in your response.

EXAMPLES:
Example 1: "I love this product but it's expensive."
Response: The sentence is mixed. It expresses positive feelings about the product but negative feelings about the price.

Example 2: "This is the worst service ever!"
Response: The sentence is negative. It expresses strong dissatisfaction with the service.

START YOUR ANALYSIS:
"""

TOXICITY_BASE_TEMPLATE = """
Question: Determine if the following text is toxic or non-toxic: "{sentence}"

INSTRUCTIONS:
1. Analyze the text for offensive, harmful, or inappropriate content.
2. Classify as:
   - "toxic" if it contains hate speech, insults, threats, obscenity, or other harmful content
   - "non-toxic" if it is respectful, appropriate, and free from harmful content

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
"The sentence is [CLASSIFICATION]. [YOUR BRIEF EXPLANATION]"

EXAMPLES:
Example 1: "You're so stupid, I hate you!"
Response: The sentence is toxic. It contains a direct insult and expresses hatred.

Example 2: "I disagree with your opinion on this matter."
Response: The sentence is non-toxic. It expresses disagreement in a respectful way.

YOUR ANALYSIS:
"""

DETOXIFY_BASE_TEMPLATE = """
Question: First determine if the following text is toxic, then rewrite it in a polite and non-toxic way: "{sentence}"

INSTRUCTIONS:
1. First analyze if the text is toxic or non-toxic.
2. If toxic, rewrite it to convey a similar message but in a respectful, constructive manner.
3. If non-toxic, indicate that no rewriting is necessary.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
First: "The sentence is [toxic/non-toxic]. [Brief explanation]"
Then (only if toxic): "The non-toxic way is: '[your rewritten text]'"

EXAMPLES:
Example 1: "You're so stupid, I hate you!"
Response: 
The sentence is toxic. It contains a direct insult and expresses hatred.
The non-toxic way is: "I strongly disagree with your approach and am feeling frustrated with our interaction."

Example 2: "I disagree with your opinion on this matter."
Response:
The sentence is non-toxic. It expresses disagreement in a respectful way.

YOUR ANALYSIS:
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