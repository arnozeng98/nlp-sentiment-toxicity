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
from src.output_parser import parse_and_fix_output, get_adaptive_prompt, extract_sentiment_info, check_result_cache, add_to_result_cache, validate_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set global variables (these will be properly initialized in the main program)
analysis_tokenizer = None
analysis_model = None

# Improved prompt templates for sentiment and toxicity analysis
# Enhance output formatting and consistency with more structured prompts

# Sentiment analysis prompt template
SENTIMENT_BASE_TEMPLATE = """You are a sentiment analysis expert. Your task is to analyze the sentiment of the given sentence and classify it as positive, negative, or mixed.

Please determine the sentiment based on the overall tone and expression:
- Positive: The sentence expresses joy, praise, satisfaction or optimism
- Negative: The sentence expresses sadness, anger, dissatisfaction or pessimism
- Mixed: The sentence contains both positive and negative sentiments or is ambiguous

Note that this is purely a sentiment analysis task. Do NOT evaluate whether the sentence is toxic or offensive. Focus ONLY on the sentiment expressed.

Sentence: {text}

You MUST answer using the following format:
"The sentence is [positive/negative/mixed]. [Detailed explanation of why the sentence expresses this sentiment]"

Example output:
"The sentence is positive. It expresses enthusiasm and appreciation for the product, using words like 'love' and 'amazing' that indicate strong positive sentiment."

Or:
"The sentence is negative. It expresses disappointment and frustration with the service, using phrases like 'terrible experience' and 'would not recommend' that convey negative sentiment."

Analyze ONLY sentiment, not toxicity or offensiveness. Your answer:"""

TOXICITY_BASE_TEMPLATE = """You are a toxicity detection expert. Your task is to determine if the following text contains toxic or non-toxic content.

Text to analyze: {sentence}

Classify the text as either toxic or non-toxic.

GUIDELINES:
- Toxic content includes hate speech, insults, obscenities, threats, or language that attacks individuals/groups
- Non-toxic content may be negative or critical but does not contain hate speech, insults, or harmful language
- Focus ONLY on toxicity - not sentiment, quality, or accuracy of the content

Your response MUST follow this format exactly:
"The sentence is [toxic/non-toxic]. [Your brief explanation why the content is toxic or non-toxic]"

Sample responses:
"The sentence is toxic. It contains offensive language and racial slurs directed at a specific group."
"The sentence is non-toxic. While the text expresses criticism, it does not contain offensive language, slurs, or personal attacks."

Analyze the toxicity level ONLY - do not evaluate sentiment."""

DETOXIFY_BASE_TEMPLATE = """
Transform this potentially toxic text into a non-toxic, respectful version.

Original text: {sentence}

IMPORTANT:
1. Make substantial changes to remove ALL toxic language
2. Maintain the core message and meaning where possible
3. Replace any offensive terms, slurs, or profanity COMPLETELY
4. Keep approximately the same length and structure

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

WARNING: Do not repeat toxic content with minor changes. Transform it completely.
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

def analyze_sentiment_direct(sentence: str, temperature: float = 0.7, max_retries: int = 5) -> dict:
    """
    Direct sentiment analysis function that bypasses the general analyze_text
    to ensure proper sentiment output formatting and extraction
    
    Args:
        sentence (str): Text to analyze for sentiment
        temperature (float): Temperature for generation
        max_retries (int): Maximum number of retries
        
    Returns:
        dict: Analysis results with sentiment label and explanation
    """
    # Check cache first
    cached_result = check_result_cache(sentence, "sentiment")
    if cached_result:
        logger.info(f"✅ Cache hit for sentiment analysis")
        return {
            "original_text": sentence,
            "output": cached_result,
        }
    
    # Prepare prompt with clear sentiment focus
    prompt = SENTIMENT_BASE_TEMPLATE.format(text=sentence)
    
    for retry_count in range(max_retries):
        try:
            # Adjust temperature slightly for each retry
            current_temp = max(0.2, temperature - (retry_count * 0.1))
            
            # Generate response with current parameters
            input_tokens = analysis_tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                output = analysis_model.generate(
                    **input_tokens, 
                    max_new_tokens=150,
                    do_sample=True, 
                    temperature=current_temp
                )
            
            output_text = analysis_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract sentiment information directly
            label, explanation = extract_sentiment_info(output_text)
            
            if label and explanation:
                # Format the result
                result = {"label": label, "explanation": explanation}
                
                # Add to cache for future use
                add_to_result_cache(sentence, "sentiment", result)
                
                return {
                    "original_text": sentence,
                    "output": result
                }
            else:
                logger.warning(f"⚠️ Sentiment extraction failed on attempt {retry_count+1}")
                
                # Log the raw output for debugging on later attempts
                if retry_count >= 1:
                    logger.debug(f"Raw output: {output_text[:200]}...")
        
        except Exception as e:
            logger.error(f"⚠️ Error during sentiment analysis (attempt {retry_count+1}): {str(e)}")
    
    # If all retries fail, return a default result
    logger.error(f"❌ All {max_retries} attempts failed for sentiment analysis")
    logger.error(f"Sentence: '{sentence[:50]}...'")
    
    default_result = {"label": "mixed", "explanation": "Failed to analyze sentiment."}
    
    return {
        "original_text": sentence,
        "output": default_result
    } 