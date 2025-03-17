#!/usr/bin/env python
# coding: utf-8
"""
Model loading module
"""

import os
import warnings
import torch
import fasttext
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForCausalLM
from langchain_ollama import ChatOllama

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Only show error messages
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from src import config

def load_models():
    """
    Load all required NLP models
    
    Returns:
        tuple: Tuple containing all loaded models and tokenizers
    """
    # Language detection model
    lang_detector = fasttext.load_model(config.LANGUAGE_MODEL_PATH)
    
    # Translation model
    translation_tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATION_MODEL_PATH)
    translation_model = MT5ForConditionalGeneration.from_pretrained(
        config.TRANSLATION_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    translation_model.eval()
    
    # Analysis model (used for both sentiment and toxicity)
    analysis_tokenizer = AutoTokenizer.from_pretrained(config.GRANITE_MODEL_PATH)
    analysis_model = AutoModelForCausalLM.from_pretrained(config.GRANITE_MODEL_PATH, device_map="auto")
    analysis_model.eval()
    
    # LLM model for agents
    ollama_model = ChatOllama(
        model=config.OLLAMA_MODEL_NAME,
        temperature=0.75,
    )
    
    return (
        lang_detector, 
        translation_tokenizer, translation_model,
        analysis_tokenizer, analysis_model,
        ollama_model
    ) 