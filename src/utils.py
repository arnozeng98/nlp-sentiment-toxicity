#!/usr/bin/env python
# coding: utf-8
"""
Language detection and translation utility module
"""

import torch
from src import config

# Set global variables (these will be properly initialized in the main program)
lang_detector = None
translation_tokenizer = None
translation_model = None

def detect_language(text: str) -> str:
    """
    Detect the language of input text using FastText
    
    Args:
        text (str): Input text for language detection
        
    Returns:
        str: Language code (e.g., 'en' for English)
    """
    detected_lang = lang_detector.predict(text)[0][0].replace("__label__", "")
    return detected_lang


def translate_to_english(text: str) -> str:
    """
    Translate non-English text to English using the Toucan-Base model
    
    Args:
        text (str): Text to translate
        
    Returns:
        str: Translated English text
    """
    input_text = f"eng: {text}"
    input_ids = translation_tokenizer(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    ).to(config.DEVICE)
    
    with torch.no_grad():
        generated_ids = translation_model.generate(
            **input_ids,
            num_beams=5,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    
    translated_text = translation_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, skip_prompt=True
    )[0]
    
    return translated_text 