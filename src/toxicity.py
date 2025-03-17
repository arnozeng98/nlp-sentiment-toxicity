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

from src import config
from src import analysis
from src.processing import batch_process_texts, preprocess_text
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
        
    # Use the improved template from analysis module
    toxic_prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=analysis.TOXICITY_BASE_TEMPLATE
    )
    
    return analysis.analyze_text(
        sentence, 
        toxic_prompt_template, 
        r"The sentence is\s+(toxic|non-toxic)\b",
        task_type="toxic"
    )


def detoxify_text(sentence: str) -> dict:
    """
    Analyze toxicity and rewrite toxic text to be non-toxic
    
    Args:
        sentence (str): Text to detoxify
        
    Returns:
        dict: Analysis results with toxicity label, explanation, and detoxified text
    """
    # Handle input if it's a list
    if isinstance(sentence, list):
        sentence = sentence[0]
        
    # Use the detoxification template
    detoxify_prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=analysis.DETOXIFY_BASE_TEMPLATE
    )
    
    return analysis.analyze_text(
        sentence, 
        detoxify_prompt_template, 
        r"The sentence is\s+(toxic|non-toxic)\b",
        task_type="detoxic",
        temperature=0.6  # Lower temperature for more consistent rewrites
    )

# Note: The process_toxicity_dataset function is now imported from data_processor
# and only kept here for backwards compatibility 