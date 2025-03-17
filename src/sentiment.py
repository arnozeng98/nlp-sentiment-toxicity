#!/usr/bin/env python
# coding: utf-8
"""
Sentiment analysis module
"""

import pandas as pd
import logging
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate

from src import config
from src import analysis
from src.processing import batch_process_texts, preprocess_text
from src.data_processor import process_sentiment_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(sentence: str) -> dict:
    """
    Analyze the sentiment of a given text
    
    Args:
        sentence (str): Text to analyze for sentiment
        
    Returns:
        dict: Analysis results with sentiment label (positive/negative/mixed) and explanation
    """
    # Use the improved template from analysis module
    sentiment_prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template=analysis.SENTIMENT_BASE_TEMPLATE
    )
    
    return analysis.analyze_text(
        sentence, 
        sentiment_prompt_template, 
        r"The sentence is\s+(positive|negative|mixed)\b",
        task_type="sentiment"
    )

# Note: The process_sentiment_dataset function is now imported from data_processor
# and only kept here for backwards compatibility 