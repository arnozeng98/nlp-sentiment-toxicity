#!/usr/bin/env python
# coding: utf-8
"""
Output parsing and fixing module for handling inconsistent model outputs
"""

import re
import json
import logging
from typing import Dict, Any, Tuple, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define common patterns and templates
SENTIMENT_PATTERN = r"The sentence is\s+(positive|negative|mixed)\b"
TOXICITY_PATTERN = r"The sentence is\s+(toxic|non-toxic)\b"
DETOX_PATTERN = r'The non-toxic way.*?"(.*?)"'

# Cache for successful prompt-response pairs
result_cache = {}

def clean_text_for_csv(text: str) -> str:
    """
    Clean text to ensure it works well in CSV output
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text suitable for CSV output
    """
    if not isinstance(text, str):
        return text
        
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace double quotes with single quotes to avoid CSV issues
    text = text.replace('"', '\'')
    
    return text.strip()

def extract_sentiment_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract sentiment label and explanation from model output
    
    Args:
        text (str): Raw model output text
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Extracted label and explanation
    """
    try:
        # First, clean up any ending markers that might be in the text
        ending_markers = ["END OF ANALYSIS", "END ANALYSIS", "END OF RESPONSE", "END RESPONSE"]
        for marker in ending_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Isolate the "YOUR ANALYSIS:" section to avoid capturing example text
        analysis_section = None
        if "YOUR ANALYSIS:" in text:
            analysis_section = text.split("YOUR ANALYSIS:")[-1].strip()
        elif "START YOUR ANALYSIS:" in text:
            analysis_section = text.split("START YOUR ANALYSIS:")[-1].strip()
        else:
            analysis_section = text  # Use full text if marker not found
            
        # Try standard pattern on the analysis section
        if analysis_section:
            match = re.search(SENTIMENT_PATTERN, analysis_section, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                # Extract explanation starting from after "The sentence is [label]" pattern
                explanation = analysis_section[match.end():].strip()
                
                # Clean up explanation (remove any template remnants)
                if explanation.startswith('.'):
                    explanation = explanation[1:].strip()
                
                # Remove any ending markers from explanation
                for marker in ending_markers:
                    if marker in explanation:
                        explanation = explanation.split(marker)[0].strip()
                        
                return label, explanation
            
        # If standard pattern fails, try alternative patterns on full text
        for pattern in [
            r"(positive|negative|mixed)\s+sentiment", 
            r"sentiment.*?is\s+(positive|negative|mixed)",
            r"class.*?(positive|negative|mixed)"
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                # Extract nearby explanation text
                sentence_end = match.end() + 100  # Look ahead ~100 chars
                explanation_text = text[match.end():min(len(text), sentence_end)].strip()
                
                # Remove any ending markers from explanation
                for marker in ending_markers:
                    if marker in explanation_text:
                        explanation_text = explanation_text.split(marker)[0].strip()
                        
                return label, explanation_text
                
        # Last resort: look for keywords
        lower_text = text.lower()
        for label in ["positive", "negative", "mixed"]:
            if label in lower_text:
                # Find a nearby explanation
                index = lower_text.find(label)
                potential_explanation = text[index + len(label):index + len(label) + 150].strip()
                
                # Remove any ending markers
                for marker in ending_markers:
                    lower_marker = marker.lower()
                    if lower_marker in potential_explanation.lower():
                        potential_explanation = potential_explanation.split(lower_marker, 1)[0].strip()
                        
                return label, potential_explanation
                
        return None, None
        
    except Exception as e:
        logger.error(f"Error extracting sentiment info: {e}")
        return None, None


def extract_toxicity_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract toxicity label and explanation from model output
    
    Args:
        text (str): Raw model output text
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Extracted label and explanation
    """
    try:
        # Try standard pattern
        match = re.search(TOXICITY_PATTERN, text, re.IGNORECASE)
        if match:
            label = match.group(1).lower()
            explanation = text[match.end():].strip()
            return label, explanation
            
        # Try alternative patterns if standard fails
        for pattern in [
            r"(toxic|non-toxic)\s+content", 
            r"content.*?is\s+(toxic|non-toxic)",
            r"class.*?(toxic|non-toxic)"
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                # Extract explanation - look for sentences after the label
                parts = text.split('.')
                for i, part in enumerate(parts):
                    if label in part.lower():
                        explanation = '.'.join(parts[i+1:]).strip()
                        if not explanation:
                            explanation = "No explanation provided."
                        return label, explanation
        
        # If we got here, all patterns failed
        return None, None
        
    except Exception as e:
        logger.error(f"Error parsing toxicity: {e}")
        return None, None


def extract_detoxified_text(text: str) -> Optional[str]:
    """
    Extract detoxified text from model output
    
    Args:
        text (str): Raw model output text
        
    Returns:
        Optional[str]: Extracted detoxified text
    """
    try:
        # Try standard pattern with quotes
        match = re.search(DETOX_PATTERN, text, re.IGNORECASE)
        if match:
            return match.group(1)
            
        # Try alternative patterns
        for pattern in [
            r"non-toxic way is[:\s]+(.*)", 
            r"detoxified version[:\s]+(.*)",
            r"polite way to say it[:\s]+(.*)"
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip('" ')
        
        # If specific patterns fail, look for sentences after indicators
        indicators = ["non-toxic", "polite", "appropriate", "better way"]
        for indicator in indicators:
            if indicator in text.lower():
                parts = text.lower().split(indicator, 1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    # Find the first sentence
                    for delimiter in ['.', '!', '?', '\n']:
                        if delimiter in candidate:
                            return candidate.split(delimiter)[0].strip('" :')
                    return candidate.strip('" :')
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting detoxified text: {e}")
        return None


def parse_and_fix_output(
    output_text: str, 
    task_type: str,
    original_text: str,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Parse and fix model output for different tasks
    
    Args:
        output_text (str): Raw model output text
        task_type (str): Type of task ('sentiment', 'toxic', 'detoxic')
        original_text (str): Original input text
        debug (bool): Whether to print debug information
        
    Returns:
        Optional[Dict[str, Any]]: Parsed and fixed output as a dictionary
    """
    if debug:
        logger.info(f"Parsing output for {task_type}: {output_text[:100]}...")
    
    try:
        # Extract text after markers if they exist
        if "START YOUR ANALYSIS:" in output_text:
            output_text = output_text.split("START YOUR ANALYSIS:")[-1].strip()
        elif "YOUR ANALYSIS:" in output_text:
            output_text = output_text.split("YOUR ANALYSIS:")[-1].strip()
        
        # Remove any ending markers the model might generate
        ending_markers = ["END OF ANALYSIS", "END ANALYSIS", "END OF RESPONSE", "END RESPONSE"]
        for marker in ending_markers:
            if marker in output_text:
                output_text = output_text.split(marker)[0].strip()
        
        # Check if it's already in JSON format
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, dict) and "label" in parsed:
                return {"label": parsed["label"], "explanation": parsed.get("explanation", "")}
            if isinstance(parsed, dict) and "output" in parsed and isinstance(parsed["output"], dict):
                return parsed["output"]
        except json.JSONDecodeError:
            pass
        
        # Process based on task type
        if task_type == "sentiment":
            label, explanation = extract_sentiment_info(output_text)
            if debug and not label:
                logger.debug(f"Failed to extract sentiment: {output_text[:150]}")
                
            if label:
                # Additional sanity check for explanations
                if explanation and len(explanation) > 500:
                    explanation = explanation[:500] + "..." # Truncate very long explanations
                elif not explanation or len(explanation) < 5:
                    explanation = "No detailed explanation provided."
                    
                # Clean up explanation - remove any template text or ending markers
                explanation = explanation.replace("Example 1:", "").replace("Example 2:", "")
                explanation = explanation.replace("Response:", "").strip()
                
                # Remove any ending markers from the explanation
                for marker in ending_markers:
                    if marker in explanation:
                        explanation = explanation.split(marker)[0].strip()
                
                # Remove quotation marks around the explanation if present
                explanation = explanation.strip('"\'')
                
                # Clean text for CSV compatibility
                label = clean_text_for_csv(label)
                explanation = clean_text_for_csv(explanation)
                
                return {
                    "label": label, 
                    "explanation": explanation
                }
                
        elif task_type == "toxic":
            label, explanation = extract_toxicity_info(output_text)
            if label:
                # Clean text for CSV compatibility
                label = clean_text_for_csv(label)
                explanation = clean_text_for_csv(explanation or "No explanation provided.")
                
                return {
                    "label": label, 
                    "explanation": explanation
                }
                
        elif task_type == "detoxic":
            toxic_label, explanation = extract_toxicity_info(output_text)
            
            # For detoxification, first do toxicity analysis
            if not toxic_label:
                # If no label found, default to non-toxic to avoid unnecessary rewriting
                toxic_label = "non-toxic"
                explanation = "No explanation provided."
            
            # If toxic, try to extract detoxified version
            if toxic_label == "toxic":
                detoxified = extract_detoxified_text(output_text)
                if not detoxified:
                    # If no detoxified text found but we know it's toxic, use placeholder
                    detoxified = "[Failed to extract detoxified text]"
            else:
                # If non-toxic, keep original
                detoxified = original_text
            
            # Clean text for CSV compatibility
            toxic_label = clean_text_for_csv(toxic_label)
            explanation = clean_text_for_csv(explanation)
            detoxified = clean_text_for_csv(detoxified)
            original_text = clean_text_for_csv(original_text)
                
            return {
                "toxicity_label": toxic_label,
                "original_text": original_text,
                "rewritten_text": detoxified,
                "explanation": explanation
            }
            
        return None
        
    except Exception as e:
        logger.error(f"Error in parse_and_fix_output: {e}")
        return None


def validate_output(result: dict, task_type: str) -> bool:
    """
    Validate if the output has all required fields for the task type
    
    Args:
        result (dict): Output result to validate
        task_type (str): Type of task
        
    Returns:
        bool: Whether the output is valid
    """
    if not result or not isinstance(result, dict):
        return False
        
    if task_type == "sentiment":
        # Check for required fields
        if "label" not in result:
            logger.warning("Missing 'label' field in sentiment output")
            return False
            
        # Validate label value
        if result["label"] not in ["positive", "negative", "mixed"]:
            logger.warning(f"Invalid sentiment label: {result.get('label')}")
            return False
            
        # Validate explanation
        if "explanation" not in result or not result["explanation"]:
            logger.warning("Missing or empty 'explanation' field in sentiment output")
            return False
            
        # Check that explanation doesn't contain template text
        suspicious_phrases = [
            "example 1", "example 2", "your analysis", 
            "response:", "format your response", "instructions:",
            "end of analysis", "end analysis", "end of response", "end response"
        ]
        for phrase in suspicious_phrases:
            if phrase in result["explanation"].lower():
                logger.warning(f"Explanation contains template text: '{phrase}'")
                return False
                
        return True
        
    elif task_type == "toxic":
        # Basic field validation
        if "label" not in result or result["label"] not in ["toxic", "non-toxic"]:
            logger.warning(f"Invalid toxicity label: {result.get('label')}")
            return False
            
        if "explanation" not in result or not result["explanation"]:
            logger.warning("Missing or empty 'explanation' field in toxicity output")
            return False
            
        # Check for template text in explanation
        suspicious_phrases = [
            "example 1", "example 2", "your analysis", 
            "response:", "format your response", "instructions:",
            "end of analysis", "end analysis", "end of response", "end response"
        ]
        for phrase in suspicious_phrases:
            if phrase in result["explanation"].lower():
                logger.warning(f"Toxicity explanation contains template text: '{phrase}'")
                return False
                
        return True
        
    elif task_type == "detoxic":
        return ("toxicity_label" in result and 
                "original_text" in result and 
                "rewritten_text" in result and
                result["toxicity_label"] in ["toxic", "non-toxic"])
    
    return False


def get_adaptive_prompt(base_prompt: str, retry_count: int, failures: List[str]) -> str:
    """
    Generate an adaptive prompt based on previous failures
    
    Args:
        base_prompt (str): Original prompt template
        retry_count (int): Current retry count
        failures (List[str]): List of failure reasons
        
    Returns:
        str: Adapted prompt
    """
    if retry_count == 0 or not failures:
        return base_prompt
        
    # Add specifics about format requirements based on failures
    format_reminder = "\nIMPORTANT: Your response MUST follow this exact format: 'The sentence is [label]. [explanation]'"
    
    # Add more specific guidance based on common failures
    if "format" in " ".join(failures).lower() or "pattern" in " ".join(failures).lower():
        format_reminder += "\nDo not add any additional text or explanations outside this format."
    
    if "label" in " ".join(failures).lower():
        format_reminder += "\nEnsure you use ONLY one of the allowed labels."
        
    # For higher retry counts, add more structure and examples
    if retry_count >= 5:
        format_reminder += "\n\nExample of correct format:\nQuestion: Explain why 'I love this!' is classified.\nThe sentence is positive. It expresses enthusiasm and appreciation."
        
    # For even higher counts, reduce creativity
    if retry_count >= 10:
        format_reminder += "\n\nDo not be creative with your answer format. Stick EXACTLY to the required format."
    
    return base_prompt + format_reminder


def check_result_cache(text: str, task_type: str) -> Optional[Dict[str, Any]]:
    """
    Check if we already have a cached result for similar text
    
    Args:
        text (str): Text to check
        task_type (str): Type of task
        
    Returns:
        Optional[Dict[str, Any]]: Cached result if available
    """
    # Create a simple cache key
    simplified_text = text.lower().strip()
    cache_key = f"{task_type}:{simplified_text}"
    
    return result_cache.get(cache_key)


def add_to_result_cache(text: str, task_type: str, result: Dict[str, Any]) -> None:
    """
    Add successful result to cache
    
    Args:
        text (str): Original text
        task_type (str): Type of task
        result (Dict[str, Any]): Result to cache
    """
    # Create a simple cache key
    simplified_text = text.lower().strip()
    cache_key = f"{task_type}:{simplified_text}"
    
    # Store in cache, limiting cache size to 1000 entries
    if len(result_cache) >= 1000:
        # Remove a random item to keep the cache size in check
        try:
            key_to_remove = next(iter(result_cache))
            result_cache.pop(key_to_remove)
        except:
            pass
    
    result_cache[cache_key] = result 