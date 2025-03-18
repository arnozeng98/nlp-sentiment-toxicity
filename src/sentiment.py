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
from src.text_processor import batch_process_texts, preprocess_text
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
    # Use the direct sentiment analysis function for better extraction
    return analysis.analyze_sentiment_direct(sentence)

# Note: The process_sentiment_dataset function is now imported from data_processor
# and only kept here for backwards compatibility 