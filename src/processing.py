#!/usr/bin/env python
# coding: utf-8
"""
Text processing utility functions for handling various NLP tasks
"""

import re
import logging
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src import config
from src import analysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of common toxic words and their non-toxic replacements
TOXIC_REPLACEMENTS = {
    # Profanity
    "fuck": "darn",
    "fucking": "very",
    "fucked": "messed up",
    "f\\*+ck": "darn",  # Handle censored forms
    "f[\\*-_]ck": "darn",
    "f[\\*-_]+ing": "very",
    "shit": "stuff",
    "sh\\*+t": "stuff",
    "s[\\*-_]t": "stuff",
    "bullshit": "nonsense",
    "bull[\\*-_]+t": "nonsense",
    "crap": "garbage",
    "damn": "darn",
    "goddamn": "extremely",
    "ass": "behind",
    "asshole": "jerk",
    "a[\\*-_]+hole": "jerk",
    
    # Slurs and insults
    "idiot": "uninformed person",
    "stupid": "misguided",
    "moron": "confused person",
    "retarded": "inappropriate",
    "retard": "person with difficulties",
    "dumb": "mistaken",
    "bitch": "difficult person",
    "b[\\*-_]tch": "difficult person", 
    "bastard": "difficult person",
    "cunt": "mean person",
    "cunty": "unpleasant",
    "c[\\*-_]nt": "mean person",
    
    # Hate speech and problematic terms
    "nigger": "[inappropriate term]",
    "nigga": "[inappropriate term]",
    "n[\\*-_]+r": "[inappropriate term]",
    "faggot": "[inappropriate term]",
    "fag": "[inappropriate term]",
    "f[\\*-_]+t": "[inappropriate term]",
    "chink": "[inappropriate term]",
    "spic": "[inappropriate term]",
    "kike": "[inappropriate term]",
    "k[\\*-_]+e": "[inappropriate term]",
    "ape": "person",  # Often used in racist contexts
    "monkey": "person",  # Often used in racist contexts
    
    # Body parts in vulgar contexts
    "dick": "jerk",
    "cock": "inappropriate term",
    "pussy": "coward",
    "butthole": "behind",
    "butt": "behind",
    
    # Additional variants
    "jackass": "fool",
    "douche": "jerk",
    "douchebag": "unpleasant person",
    "tits": "chest",
    "titties": "chest",
    "whore": "promiscuous person",
    "slut": "promiscuous person",
    "hoe": "promiscuous person",
    "piss": "urinate",
    "pissed": "upset"
}

# Additional substitution patterns for complex phrases
PHRASE_REPLACEMENTS = [
    # Racist expressions
    (r"go\s+back\s+to\s+your\s+country", "belong wherever you choose"),
    (r"illegal\s+alien", "undocumented immigrant"),
    (r"towel\s*head", "person from the Middle East"),
    (r"sand\s*nigger", "person from the Middle East"),
    
    # Sexist phrases
    (r"belongs\s+in\s+the\s+kitchen", "has many potential roles"),
    (r"make\s+me\s+a\s+sandwich", "help me with something"),
    
    # Ableist phrases
    (r"special\s+needs", "diverse needs"),
    (r"mental\s+retard", "person with cognitive differences"),
    
    # Xenophobic phrases
    (r"china\s+virus", "coronavirus"),
    (r"wuhan\s+virus", "coronavirus"),
]

# Phrases that suggest harmful content
HARMFUL_PHRASES = [
    "kill yourself",
    "kill them all",
    "hate all",
    "die in a fire",
    "should be shot",
    "deserve to die",
    "wish you were dead",
    "hang yourself",
    "blow up",
    "bomb the",
    "wipe them out",
    "exterminate",
    "eliminate",
    "purge all",
    "gas the"
]

def clean_text(text):
    """Clean text by removing extra whitespace"""
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    return text.strip()


def rule_based_detoxify(text):
    """
    Use rule-based approach to detoxify text when model approaches fail
    
    Args:
        text (str): Text to detoxify
        
    Returns:
        str: Detoxified text using simple replacement rules
    """
    if not text:
        return text
        
    # Convert to lowercase for matching but preserve original case structure
    lower_text = text.lower()
    detoxified = text
    
    # Apply word replacements with improved regex patterns for word boundaries
    for toxic, replacement in TOXIC_REPLACEMENTS.items():
        # Use word boundary regex to avoid partial word matches
        pattern = r'\b' + re.escape(toxic) + r'\b'
        
        # For patterns that already contain regex characters, don't escape them
        if any(c in toxic for c in "\\*+[]"):
            pattern = r'\b' + toxic + r'\b'
            
        # Count matches for logging
        matches = len(re.findall(pattern, lower_text, re.IGNORECASE))
        
        if matches > 0:
            # Replace with proper case preservation where possible
            detoxified = re.sub(pattern, replacement, detoxified, flags=re.IGNORECASE)
            logger.debug(f"Replaced {matches} instances of '{toxic}' with '{replacement}'")
    
    # Apply phrase replacements
    for pattern, replacement in PHRASE_REPLACEMENTS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            detoxified = re.sub(pattern, replacement, detoxified, flags=re.IGNORECASE)
            logger.debug(f"Replaced phrase matching '{pattern}' with '{replacement}'")
    
    # Handle common slurs even if they don't exactly match our patterns
    for slur in ["racist", "sexist", "homophobic", "xenophobic"]:
        if slur in lower_text:
            # Add a note about problematic content
            if not "[Note:" in detoxified:
                detoxified += " [Note: This comment has been modified to remove potentially harmful content]"
            break
    
    # Remove warning marker if we didn't actually make any changes
    if detoxified == text and "[Note:" in detoxified:
        detoxified = detoxified.replace(" [Note: This comment has been modified to remove potentially harmful content]", "")
    
    return detoxified


def get_topic_summary(text):
    """
    Extract a simple topic summary from text to use in generic replacements
    
    Args:
        text (str): Text to summarize
        
    Returns:
        str: Short topic summary
    """
    lower_text = text.lower()
    
    if "vote" in lower_text or "election" in lower_text or "trump" in lower_text:
        return "concerns about political matters"
    elif "wikipedia" in lower_text or "edit" in lower_text:
        return "frustration with online content policies"
    elif "game" in lower_text or "trailer" in lower_text:
        return "opinions about media content"
    elif "obama" in lower_text or "president" in lower_text:
        return "views on political figures"
    elif "school" in lower_text or "education" in lower_text:
        return "thoughts about education"
    else:
        return "strong opinions about various topics"


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
        # Use different temperature for detoxification
        temp = 0.9 if task_type == "detoxic" else 0.7
        
        response = ollama_model.invoke(
            prompt, 
            temperature=temp,
            top_p=0.95,
            top_k=60
        )
        
        # Parse response
        content = response.content if hasattr(response, 'content') else str(response)
        result = parse_and_fix_output(content, task_type, text)
        
        if result:
            logger.debug(f"✅ Ollama processed {task_type} successfully")
            return {"original_text": text, "output": result}
        else:
            logger.debug(f"⚠️ Ollama failed to produce valid {task_type} output")
            
            # For detoxification, try rule-based fallback if model fails
            if task_type == "detoxic":
                detoxified = rule_based_detoxify(text)
                if detoxified != text:
                    logger.info("Using rule-based detoxification as fallback")
                    return {
                        "original_text": text,
                        "output": {
                            "toxicity_label": "toxic",
                            "explanation": "Detoxified using rule-based fallback",
                            "rewritten_text": detoxified
                        }
                    }
            
            return None
            
    except Exception as e:
        logger.error(f"Error using Ollama for {task_type}: {e}")
        
        # For detoxification, try rule-based fallback if model errors
        if task_type == "detoxic":
            try:
                detoxified = rule_based_detoxify(text)
                if detoxified != text:
                    logger.info("Using rule-based detoxification after Ollama error")
                    return {
                        "original_text": text,
                        "output": {
                            "toxicity_label": "toxic",
                            "explanation": "Detoxified using rule-based fallback after error",
                            "rewritten_text": detoxified
                        }
                    }
            except Exception as fallback_error:
                logger.error(f"Error in rule-based fallback: {fallback_error}")
                
        return None 