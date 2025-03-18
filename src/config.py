#!/usr/bin/env python
# coding: utf-8
"""
Configuration file for storing global variables and constants
"""

import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Model configurations
DEVICE = "cuda"
LANGUAGE_MODEL_PATH = os.path.join(MODELS_DIR, "lid.176.bin")
TRANSLATION_MODEL_PATH = "UBC-NLP/toucan-base"
GRANITE_MODEL_PATH = "ibm-granite/granite-3.0-2b-instruct"
OLLAMA_MODEL_NAME = "llama3.2:1b"

# Enable fallback to Ollama when transformers model fails
ENABLE_OLLAMA_FALLBACK = True
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama API endpoint

# Task types
VALID_TASKS = ["toxic", "sentiment", "detoxic"]

# Execution parameters
MAX_RETRIES = 100
MAX_PROCESS_RETRIES = 10  # Increased from 5 to 10 for more retries before fallback
MAX_ITERATIONS = 5
RANDOM_DELAY_RANGE = (0, 2)

# File paths
SENTIMENT_INPUT = os.path.join(DATA_DIR, "multilingual-sentiment-test-solutions.csv")
SENTIMENT_OUTPUT = os.path.join(OUTPUT_DIR, "answer-multilingual-sentiment-test-solutions.csv")
TOXICITY_INPUT = os.path.join(DATA_DIR, "toxic-test-solutions.csv")
TOXICITY_OUTPUT = os.path.join(OUTPUT_DIR, "answer-toxic-test-solutions.csv") 