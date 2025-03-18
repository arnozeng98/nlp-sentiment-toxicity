#!/usr/bin/env python
# coding: utf-8
"""
NLP Tool Suite for Multi-language Processing, Sentiment Analysis, and Toxicity Detection/Remediation.

This module provides a set of NLP tools for:
1. Language detection and translation
2. Sentiment analysis 
3. Toxicity detection
4. Detoxification of toxic content

It uses a combination of local models and LangChain tools to process text in various languages.
"""

# Disable TensorFlow oneDNN warning messages
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show error messages

import sys
import re
import time
import logging
import warnings
import traceback
import pandas as pd
from tqdm import tqdm
import torch
import fasttext
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    MT5ForConditionalGeneration
)
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Ignore TensorFlow warnings
warnings.filterwarnings('ignore', category=Warning)

# Import from our modules
from src.data_processor import process_sentiment_dataset, process_toxicity_dataset
from src import config
from src import models, language_utils, analysis  # Import necessary modules

# Ensure directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

# Create necessary directories first
ensure_directories()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("output", "processing.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the NLP processing pipeline.
    """
    start_time = time.time()
    logger.info("Starting NLP processing pipeline")
    
    try:
        # Directories already created at program start, no need to call again
        # ensure_directories()
        
        # Load models (critical step that was missing before)
        logger.info("Loading NLP models...")
        (
            language_utils.lang_detector, 
            language_utils.translation_tokenizer, language_utils.translation_model,
            analysis.analysis_tokenizer, analysis.analysis_model,
            _
        ) = models.load_models()
        logger.info("Models loaded successfully")
        
        # Check for model files
        model_files = {
            "lang_detector": os.path.join("models", "lid.176.bin"),
            "analysis_model": os.path.join("models", "analysis_model")
        }
        
        # Process datasets with progress tracking
        with tqdm(total=2, desc="Processing datasets", unit="dataset") as dataset_pbar:
            # Process sentiment dataset
            dataset_pbar.set_description("Processing sentiment dataset")
            logger.info(f"Processing sentiment dataset: {config.SENTIMENT_INPUT}")
            if not os.path.exists(config.SENTIMENT_INPUT):
                logger.error(f"Sentiment input file not found: {config.SENTIMENT_INPUT}")
                raise FileNotFoundError(f"Sentiment input file not found: {config.SENTIMENT_INPUT}")
                
            sentiment_df = process_sentiment_dataset(config.SENTIMENT_INPUT, config.SENTIMENT_OUTPUT)
            logger.info(f"Sentiment analysis complete. Results saved to {config.SENTIMENT_OUTPUT}")
            dataset_pbar.update(1)
    
    # Process toxicity dataset
            dataset_pbar.set_description("Processing toxicity dataset")
            logger.info(f"Processing toxicity dataset: {config.TOXICITY_INPUT}")
            if not os.path.exists(config.TOXICITY_INPUT):
                logger.error(f"Toxicity input file not found: {config.TOXICITY_INPUT}")
                raise FileNotFoundError(f"Toxicity input file not found: {config.TOXICITY_INPUT}")
                
            toxicity_df = process_toxicity_dataset(config.TOXICITY_INPUT, config.TOXICITY_OUTPUT)
            logger.info(f"Toxicity analysis complete. Results saved to {config.TOXICITY_OUTPUT}")
            dataset_pbar.update(1)
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("=" * 50)
        logger.info("Processing Summary")
        logger.info("=" * 50)
        logger.info(f"Sentiment analysis: {len(sentiment_df)} records processed")
        logger.info(f"Toxicity analysis: {len(toxicity_df)} records processed")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
