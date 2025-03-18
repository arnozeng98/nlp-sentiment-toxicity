#!/usr/bin/env python
# coding: utf-8
"""
Output parsing and fixing module for handling inconsistent model outputs
"""

import re
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from difflib import SequenceMatcher

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
        
        # Clean the text - remove template examples
        if "Your answer:" in text:
            # Extract only the part after "Your answer:"
            text = text.split("Your answer:")[-1].strip()
        
        # Check if full template is still in output
        if "Or:" in text and "Example output:" in text:
            # Extract only the actual model output after all examples
            parts = text.split("Your answer:")
            if len(parts) > 1:
                text = parts[-1].strip()
            else:
                # Try other markers
                markers = ["Analyze ONLY sentiment", "not toxicity or offensiveness"]
                for marker in markers:
                    if marker in text:
                        text = text.split(marker)[-1].strip()
        
        # Check if the text contains toxicity classification by mistake
        if re.search(r"(toxic|non-toxic)\b", text, re.IGNORECASE) and not re.search(r"(positive|negative|mixed)\b", text, re.IGNORECASE):
            logger.warning("Output contains toxicity analysis instead of sentiment")
            return None, None
                
        # Try the most reliable pattern first - exact format we requested
        pattern = r"The sentence is\s+(positive|negative|mixed)\.?\s+(.+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            label = match.group(1).lower()
            explanation = match.group(2).strip()
            return label, explanation
            
        # Try alternative sentiment-specific patterns
        sentiment_patterns = [
            # Common variations
            r"sentiment[:\s]+is\s+(positive|negative|mixed)",
            r"sentiment[:\s]+(positive|negative|mixed)",
            r"(positive|negative|mixed)\s+sentiment",
            r"classified\s+as\s+(positive|negative|mixed)",
            r"text\s+is\s+(positive|negative|mixed)",
            r"the\s+text\s+expresses\s+a\s+(positive|negative|mixed)",
            r"expresses\s+(positive|negative|mixed)\s+sentiment"
        ]
        
        for pattern in sentiment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                
                # Try to extract explanation
                # Look for sentences after the label
                parts = text.split(match.group(0), 1)
                if len(parts) > 1:
                    explanation = parts[1].strip()
                    if not explanation:
                        # Look before the match if after is empty
                        sentences = text.split('.')
                        for i, sentence in enumerate(sentences):
                            if match.group(0) in sentence:
                                # Take the next sentence if available
                                if i+1 < len(sentences):
                                    explanation = sentences[i+1].strip()
                                break
                else:
                    # If we couldn't split on the match, extract a portion after it
                    pos = match.end()
                    if pos < len(text):
                        explanation = text[pos:pos+200].strip()
                    else:
                        explanation = "No explanation provided."
                
                return label, explanation
        
        # If no specific patterns match, look for sentiment keywords
        for sentiment in ["positive", "negative", "mixed"]:
            if re.search(r'\b' + sentiment + r'\b', text, re.IGNORECASE):
                # Found a sentiment, now try to extract an explanation
                matches = list(re.finditer(r'\b' + sentiment + r'\b', text, re.IGNORECASE))
                if matches:
                    last_match = matches[-1]  # Use the last occurrence
                    if last_match.end() < len(text):
                        # Extract up to 200 chars after the sentiment word
                        explanation = text[last_match.end():last_match.end()+200].strip()
                        if explanation.startswith('.'):
                            explanation = explanation[1:].strip()
                        return sentiment, explanation
        
        # Check if we have sentiment-related keywords but no clear classification
        sentiment_keywords = {
            "positive": ["happy", "great", "excellent", "good", "wonderful", "fantastic", "pleased", "satisfied"],
            "negative": ["unhappy", "terrible", "awful", "bad", "horrible", "disappointed", "unsatisfied", "poor"],
            "mixed": ["mixed", "both", "conflicted", "ambivalent", "balanced", "neutral"]
        }
        
        # Count occurrences of each sentiment's keywords
        keyword_counts = {sentiment: 0 for sentiment in sentiment_keywords}
        
        for sentiment, keywords in sentiment_keywords.items():
            for keyword in keywords:
                keyword_counts[sentiment] += len(re.findall(r'\b' + keyword + r'\b', text, re.IGNORECASE))
        
        # If any sentiment has more than 2 keywords, consider it the sentiment
        max_sentiment = max(keyword_counts.items(), key=lambda x: x[1])
        if max_sentiment[1] >= 2:
            return max_sentiment[0], "Based on keyword analysis of the output."
                
        logger.warning(f"Could not extract sentiment info from: {text[:100]}...")
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
        # First, clean up any ending markers that might be in the text
        ending_markers = ["END OF ANALYSIS", "END ANALYSIS", "END OF RESPONSE", "END RESPONSE"]
        for marker in ending_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Clean the text - remove template examples
        if "Sample responses:" in text and "Your response MUST" in text:
            # Find the actual response after the template instructions
            content_after_samples = text.split("Sample responses:")[-1]
            # Look for the actual response after the examples
            response_markers = [
                "Analyze the toxicity level ONLY",
                "The sentence is",
                "The text is",
                "This content is",
                "This text is"
            ]
            
            for marker in response_markers:
                if marker in content_after_samples:
                    parts = content_after_samples.split(marker)
                    if len(parts) > 1 and marker != "Analyze the toxicity level ONLY":
                        # If we found an actual response marker, extract from there
                        text = marker + parts[-1]
                        break
                    elif marker == "Analyze the toxicity level ONLY" and len(parts) > 1:
                        # Special case: this is the end of instructions, take everything after it
                        text = parts[-1].strip()
                        break
        
        # Check if the text contains sentiment classification by mistake
        if re.search(r"\b(positive|negative|mixed)\b", text, re.IGNORECASE) and not re.search(r"\b(toxic|non-toxic)\b", text, re.IGNORECASE):
            logger.warning("Output contains sentiment analysis instead of toxicity")
            return None, None
        
        # Try the most reliable pattern first - exact format we requested
        pattern = r"The sentence is\s+(toxic|non-toxic)\.?\s+(.+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            label = match.group(1).lower()
            explanation = match.group(2).strip()
            return label, explanation
        
        # Try alternative patterns for toxicity
        toxicity_patterns = [
            # Common variations
            r"The text is\s+(toxic|non-toxic)",
            r"This content is\s+(toxic|non-toxic)",
            r"This text is\s+(toxic|non-toxic)",
            r"(toxic|non-toxic)\s+content",
            r"content is\s+(toxic|non-toxic)",
            r"classified as\s+(toxic|non-toxic)",
            r"contains\s+(toxic|non-toxic)\s+content",
            r"(toxic|non-toxic)\s+language",
            r"language is\s+(toxic|non-toxic)"
        ]
        
        for pattern in toxicity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                
                # Try to extract explanation
                # Look for text after the label
                parts = text.split(match.group(0), 1)
                if len(parts) > 1:
                    explanation = parts[1].strip()
                    if not explanation:
                        # Look before the match if after is empty
                        sentences = text.split('.')
                        for i, sentence in enumerate(sentences):
                            if match.group(0) in sentence:
                                # Take the next sentence if available
                                if i+1 < len(sentences):
                                    explanation = sentences[i+1].strip()
                                break
                else:
                    # If we couldn't split on the match, extract a portion after it
                    pos = match.end()
                    if pos < len(text):
                        explanation = text[pos:pos+200].strip()
                    else:
                        explanation = "No explanation provided."
                
                # Clean up explanation
                explanation = explanation.strip('"\'').strip()
                if explanation.startswith('.') or explanation.startswith(','):
                    explanation = explanation[1:].strip()
                
                return label, explanation
        
        # If no specific patterns match, look for toxicity keywords
        for label in ["toxic", "non-toxic"]:
            if re.search(r'\b' + label + r'\b', text, re.IGNORECASE):
                # Found a toxicity indication, now try to extract an explanation
                matches = list(re.finditer(r'\b' + label + r'\b', text, re.IGNORECASE))
                if matches:
                    last_match = matches[-1]  # Use the last occurrence
                    if last_match.end() < len(text):
                        # Extract up to 200 chars after the toxicity word
                        explanation = text[last_match.end():last_match.end()+200].strip()
                        if explanation.startswith('.') or explanation.startswith(','):
                            explanation = explanation[1:].strip()
                        return label, explanation
        
        # Check for toxicity-related keywords
        toxicity_keywords = {
            "toxic": ["offensive", "insult", "slur", "hate", "racist", "sexist", "abusive", "threat", "obscene", "profanity"],
            "non-toxic": ["respectful", "appropriate", "neutral", "civil", "polite", "acceptable", "clean", "inoffensive"]
        }
        
        # Count occurrences of each toxicity category's keywords
        keyword_counts = {category: 0 for category in toxicity_keywords}
        
        for category, keywords in toxicity_keywords.items():
            for keyword in keywords:
                keyword_counts[category] += len(re.findall(r'\b' + keyword + r'\b', text, re.IGNORECASE))
        
        # If any category has more than 2 keywords, consider it the toxicity label
        max_category = max(keyword_counts.items(), key=lambda x: x[1])
        if max_category[1] >= 2:
            return max_category[0], "Based on keyword analysis of the output."
        
        logger.warning(f"Could not extract toxicity info from: {text[:100]}...")
        return None, None
        
    except Exception as e:
        logger.error(f"Error extracting toxicity info: {e}")
        return None, None


def extract_detoxified_text(text: str, original_text: str = None) -> Optional[str]:
    """
    Extract the detoxified text from the model output
    
    Args:
        text (str): Raw model output text
        original_text (str): Original text for comparison
        
    Returns:
        Optional[str]: Extracted detoxified text, or None if no match
    """
    try:
        # Debug log to see what we're trying to parse
        logger.debug(f"Trying to extract detoxified text from: {text[:200]}...")
        
        # First, clean up any ending markers that might be in the text
        ending_markers = ["END OF ANALYSIS", "END ANALYSIS", "END OF RESPONSE", "END RESPONSE"]
        for marker in ending_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Detect and handle placeholder formats
        placeholder_patterns = [
            r'\[your completely detoxified version\]', 
            r'your completely detoxified version',
            r'\[detoxified version\]',
            r'\[rewritten text\]'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, text.lower(), re.IGNORECASE):
                logger.warning("Found placeholder text instead of actual detoxification")
                return None
            
        # Try the most specific pattern first - exact format requested in template
        rewritten_match = re.search(r'(?:^|[\n\r])(?:"|\')?Rewritten text:(?:\s*)(.*?)(?:"|\')?(?:$|[\n\r])', text, re.IGNORECASE | re.DOTALL)
        if rewritten_match:
            extracted = rewritten_match.group(1).strip()
            # Remove brackets if they were included
            extracted = re.sub(r'^\[|\]$', '', extracted).strip()
            
            # Skip placeholder or too short extractions
            if len(extracted) < 5 or any(p.lower() in extracted.lower() for p in placeholder_patterns):
                logger.warning("Found placeholder or too short text in rewritten_text match")
                return None
                
            # Check similarity to original if provided
            if original_text and similarity_ratio(original_text, extracted) > 0.8:
                logger.warning("Extracted text too similar to original")
                return None
                
            return clean_text_for_csv(extracted)
            
        # Secondary pattern for the exact format in the template
        rewritten_match = re.search(r'Rewritten text:\s*["\']?(.*?)["\']?(?:$|[\n\r])', text, re.IGNORECASE | re.DOTALL)
        if rewritten_match:
            extracted = rewritten_match.group(1).strip()
            # Remove brackets if they were included
            extracted = re.sub(r'^\[|\]$', '', extracted).strip()
            
            # Skip placeholder or too short extractions
            if len(extracted) < 5 or any(p.lower() in extracted.lower() for p in placeholder_patterns):
                logger.warning("Found placeholder text in secondary match")
                return None
                
            # Check similarity to original if provided
            if original_text and similarity_ratio(original_text, extracted) > 0.8:
                logger.warning("Extracted text too similar to original")
                return None
                
            return clean_text_for_csv(extracted)
            
        # Look for the detoxified section by detecting examples and finding text after them
        if "Example" in text and ("Original:" in text or "Original text:" in text):
            # Find the last example
            examples = re.finditer(r'Example\s+\d+:', text)
            last_example_pos = 0
            for example in examples:
                last_example_pos = max(last_example_pos, example.start())
                
            if last_example_pos > 0:
                # Find where the examples end
                last_example_text = text[last_example_pos:]
                example_end_match = re.search(r'Rewritten text:.*?(?:\n\n|\n\s*\n|$)', last_example_text, re.DOTALL)
                if example_end_match:
                    # Look for actual content after the examples
                    post_examples_text = text[last_example_pos + example_end_match.end():]
                    # Try to find a "Rewritten text:" section in the actual response
                    actual_rewrite = re.search(r'Rewritten text:\s*(.*?)(?:$|[\n\r])', post_examples_text, re.IGNORECASE | re.DOTALL)
                    if actual_rewrite:
                        extracted = actual_rewrite.group(1).strip()
                        # Remove brackets if included
                        extracted = re.sub(r'^\[|\]$', '', extracted).strip()
                        
                        # Check for placeholders or too short extractions
                        if len(extracted) < 5 or any(p.lower() in extracted.lower() for p in placeholder_patterns):
                            return None
                            
                        # Check similarity to original if provided
                        if original_text and similarity_ratio(original_text, extracted) > 0.8:
                            logger.warning("Extracted text too similar to original")
                            return None
                            
                        return clean_text_for_csv(extracted)
        
        # Try finding non-toxic alternatives or rewrites in the text
        patterns = [
            # More specific patterns first
            r'Non-toxic version:\s*(.*?)(?:$|(?=\n\n))',
            r'Detoxified version:\s*(.*?)(?:$|(?=\n\n))',
            r'Here is a more appropriate version:\s*(.*?)(?:$|(?=\n\n))',
            r'A more respectful alternative:\s*(.*?)(?:$|(?=\n\n))',
            r'(?:^|[\n\r])(?:"|\')?(.*?)(?:"|\')(?:$|[\n\r])', # Look for quoted text on its own line
            r'The non-toxic way(?:\s+is)?(?:\s+to say)?[:\s]+(.*?)(?:$|(?=\n\n))',
            r'A better way to express this:\s*(.*?)(?:$|(?=\n\n))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip('" \'\n\t')
                
                # Skip placeholder or too short extractions
                if len(extracted) < 10 or any(p.lower() in extracted.lower() for p in placeholder_patterns):
                    continue
                    
                # Check similarity to original if provided
                if original_text and similarity_ratio(original_text, extracted) > 0.8:
                    logger.warning("Extracted text too similar to original")
                    continue
                    
                return clean_text_for_csv(extracted)
        
        # As a last resort, look for the longest non-instruction paragraph
        paragraphs = re.split(r'\n\s*\n', text)
        valid_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            # Skip if it's short, contains instruction keywords, or is a placeholder
            if (len(para) > 20 and 
                not re.search(r'\b(instruction|example|original|warning|important)\b', para.lower()) and
                not any(p.lower() in para.lower() for p in placeholder_patterns)):
                
                # Skip if too similar to original
                if original_text and similarity_ratio(original_text, para) > 0.8:
                    continue
                    
                valid_paragraphs.append(para)
                
        if valid_paragraphs:
            # Return the longest valid paragraph
            return clean_text_for_csv(max(valid_paragraphs, key=len))
                
        # If all else fails, return None to force retrying or using rule-based alternative
        logger.warning("Could not extract valid detoxified text")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting detoxified text: {e}")
        return None


def similarity_ratio(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        float: Similarity ratio between 0.0 and 1.0
    """
    # Handle edge cases
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
        
    # Use SequenceMatcher for string similarity
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


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
                detoxified = extract_detoxified_text(output_text, original_text)
                if not detoxified:
                    # Check if we're dealing with a placeholder response
                    if "your completely detoxified version" in output_text.lower():
                        logger.warning("Detected placeholder text in model output")
                        # Force returning None to trigger retries or fallback
                        return None
                    # If no detoxified text found but we know it's toxic, use original
                    detoxified = original_text
                    logger.warning("Using original text as fallback for failed detoxification")
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
        return ("label" in result and 
                "explanation" in result and 
                result["label"] in ["positive", "negative", "mixed"])
                
    elif task_type == "toxic":
        if not ("label" in result and 
                "explanation" in result and 
                result["label"] in ["toxic", "non-toxic"]):
            return False
            
        # Extra checks for the explanation to ensure it's not repeating templates
        if "explanation" in result:
            for phrase in ["your explanation", "within 50 words", "followed sructure"]:
                if phrase in result["explanation"].lower():
                    logger.warning(f"Toxicity explanation contains template text: '{phrase}'")
                    return False
                
        return True
        
    elif task_type == "detoxic":
        if not ("toxicity_label" in result and 
                "original_text" in result and 
                "rewritten_text" in result and
                result["toxicity_label"] in ["toxic", "non-toxic"]):
            return False
            
        # Check if the rewritten text is too similar to original text
        if "toxicity_label" in result and result["toxicity_label"] == "toxic":
            if "original_text" in result and "rewritten_text" in result:
                # Calculate similarity ratio between original and rewritten text
                original = result["original_text"].lower()
                rewritten = result["rewritten_text"].lower()
                
                # Skip short texts - they may be legitimately similar
                if len(original) > 20:
                    similarity = similarity_ratio(original, rewritten)
                    
                    # If more than 75% similar, likely not properly detoxified
                    if similarity > 0.75:
                        logger.warning(f"Detoxified text too similar to original: {similarity:.2f} similarity ratio")
                        return False
                        
                    # Log high similarity for monitoring
                    if similarity > 0.6:
                        logger.info(f"High similarity in detoxification: {similarity:.2f}")
                        
                # Check for simple character removal without actual rewriting
                if rewritten in original or original in rewritten:
                    logger.warning("Detoxified text is just a substring of original or vice versa")
                    return False
        
        return True
    
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