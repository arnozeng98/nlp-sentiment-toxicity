#!/usr/bin/env python
# coding: utf-8
"""
Dataset processing module for handling input/output data formats and transformations
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from src.text_processor import preprocess_text, batch_process_texts

# Setup logging - default to INFO, can be overridden
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only configure if not already configured
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Define column mappings for different datasets
COLUMN_MAPPINGS = {
    # Sentiment dataset possible column names
    "sentiment": {
        "text_column": ["sentence", "text", "content"],
        "id_column": ["sentence id", "sentence_id", "id"],
        "label_column": ["class-label", "class_label", "label", "sentiment"]
    },
    # Toxicity dataset possible column names
    "toxic": {
        "text_column": ["text", "content", "message"],
        "id_column": ["data_id", "id"],
        "sample_id_column": ["sample_id"],
        "label_column": ["source_label", "label", "toxicity"],
        "platform_column": ["platform", "source"]
    }
}

# Define output column formats
OUTPUT_FORMATS = {
    "sentiment": ["sentence id", "sentence", "class-label", "predicted-label", "explanation"],
    "toxic": ["data_id", "sample_id", "text", "source_label", "platform", "predicted_label", "rewritten_text"]
}

def detect_columns(df: pd.DataFrame, task_type: str) -> Dict[str, str]:
    """
    Detect the appropriate columns in a DataFrame based on task type
    
    Args:
        df (pd.DataFrame): Input dataframe
        task_type (str): Type of task ('sentiment' or 'toxic')
        
    Returns:
        Dict[str, str]: Mapping of column types to column names
    """
    columns = {}
    mapping = COLUMN_MAPPINGS.get(task_type, {})
    
    logger.debug(f"Detecting columns for {task_type} task")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    
    # Debug: Print all column names with their normalized versions for comparison
    actual_columns = {col: col.lower().replace('-', '_').replace(' ', '_') for col in df.columns}
    logger.debug(f"Normalized columns: {actual_columns}")
    
    # Find the text column
    for col in mapping.get("text_column", []):
        normalized_col = col.lower().replace('-', '_').replace(' ', '_')
        for actual_col, normalized_actual in actual_columns.items():
            if normalized_col == normalized_actual:
                columns["text_column"] = actual_col
                logger.debug(f"Matched text column: {actual_col}")
                break
        if "text_column" in columns:
            break
    
    # Find the ID column
    for col in mapping.get("id_column", []):
        normalized_col = col.lower().replace('-', '_').replace(' ', '_')
        for actual_col, normalized_actual in actual_columns.items():
            if normalized_col == normalized_actual:
                columns["id_column"] = actual_col
                logger.debug(f"Matched ID column: {actual_col}")
                break
        if "id_column" in columns:
            break
    
    # Find the label column - add extra handling for common format issues
    for col in mapping.get("label_column", []):
        # Try exact match first
        if col in df.columns:
            columns["label_column"] = col
            logger.debug(f"Matched label column (exact): {col}")
            break
            
        # Try case-insensitive match
        normalized_col = col.lower().replace('-', '_').replace(' ', '_')
        for actual_col, normalized_actual in actual_columns.items():
            if normalized_col == normalized_actual:
                columns["label_column"] = actual_col
                logger.debug(f"Matched label column (normalized): {actual_col}")
                break
                
        # Special case: check for class-label with various formats
        if col in ["class-label", "class_label"] and "class-label" in df.columns:
            columns["label_column"] = "class-label"
            logger.debug("Matched special case: class-label")
            break
            
        if "label_column" in columns:
            break
    
    # For toxicity, find additional columns
    if task_type == "toxic":
        # Find sample_id column
        for col in mapping.get("sample_id_column", []):
            normalized_col = col.lower().replace('-', '_').replace(' ', '_')
            for actual_col, normalized_actual in actual_columns.items():
                if normalized_col == normalized_actual:
                    columns["sample_id_column"] = actual_col
                    logger.debug(f"Matched sample_id column: {actual_col}")
                    break
            if "sample_id_column" in columns:
                break
        
        # Find platform column
        for col in mapping.get("platform_column", []):
            normalized_col = col.lower().replace('-', '_').replace(' ', '_')
            for actual_col, normalized_actual in actual_columns.items():
                if normalized_col == normalized_actual:
                    columns["platform_column"] = actual_col
                    logger.debug(f"Matched platform column: {actual_col}")
                    break
            if "platform_column" in columns:
                break
    
    logger.info(f"Detected columns for {task_type}: {columns}")
    return columns

def process_dataset(
    file_path: str, 
    output_path: str, 
    task_type: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generic function to process a dataset for sentiment or toxicity analysis
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str): Path to save the output CSV
        task_type (str): Type of task ('sentiment' or 'toxic')
        debug (bool): Enable debug logging
        
    Returns:
        pd.DataFrame: The processed dataframe with analysis results
    """
    # Set debug mode if requested
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {task_type} dataset with {len(df)} rows")
        logger.debug(f"Columns in dataset: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading {task_type} dataset: {e}")
        raise
    
    # Detect columns based on dataset type
    columns = detect_columns(df, task_type)
    
    if not columns.get("text_column"):
        error_msg = f"Could not find text column in {task_type} dataset"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    text_column = columns["text_column"]
    
    # Preprocess the texts with progress bar
    texts = df[text_column]
    cleaned_texts = []
    
    logger.info(f"Preprocessing {task_type} texts...")
    with tqdm(total=len(texts), desc="Preprocessing", unit="text") as pbar:
        for text in texts:
            cleaned_texts.append(preprocess_text(text))
            pbar.update(1)
    
    logger.info(f"Preprocessed {len(cleaned_texts)} texts for {task_type} analysis")
    
    # Run the appropriate analysis
    logger.info(f"Running {task_type} analysis...")
    
    if task_type == "sentiment":
        analysis_type = "sentiment"
    else:  # toxic
        analysis_type = "detoxic"
    
    results = batch_process_texts(cleaned_texts, task_type=analysis_type)
    logger.info(f"Completed {task_type} analysis with {len(results)} results")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Format output according to required format
    output_df = format_output(df, results_df, task_type, columns)
    
    # Save results - ensure index=False to avoid row indices in output
    try:
        # Clean text fields to handle newlines and quotes before saving
        for col in output_df.columns:
            if output_df[col].dtype == 'object':  # Only process string columns
                # Replace newlines with spaces
                output_df[col] = output_df[col].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
                # Handle quotes by ensuring proper escaping (pandas to_csv will handle this, 
                # but we make sure the text itself doesn't have malformed quotes)
                output_df[col] = output_df[col].apply(lambda x: x.replace('"', '\'') if isinstance(x, str) else x)
        
        # Save with quoting options to properly handle text fields
        output_df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL mode
        logger.info(f"Saved {task_type} analysis results to {output_path}")
    except Exception as e:
        logger.error(f"Error saving {task_type} results: {e}")
        raise
    
    # Reset logger level back to INFO if we changed it
    if debug:
        logger.setLevel(logging.INFO)
    
    return output_df

def format_output(
    original_df: pd.DataFrame,
    results_df: pd.DataFrame, 
    task_type: str,
    columns: Dict[str, str]
) -> pd.DataFrame:
    """
    Format the output dataframe according to the required format
    
    Args:
        original_df (pd.DataFrame): Original input dataframe
        results_df (pd.DataFrame): Results dataframe
        task_type (str): Type of task ('sentiment' or 'toxic')
        columns (Dict[str, str]): Mapping of column types to column names
        
    Returns:
        pd.DataFrame: Formatted output dataframe
    """
    logger.debug(f"Original DataFrame columns: {original_df.columns.tolist()}")
    logger.debug(f"Results DataFrame columns: {results_df.columns.tolist()}")
    logger.debug(f"Detected columns mapping: {columns}")
    
    if task_type == "sentiment":
        # Create a new dataframe with the required columns
        output_df = pd.DataFrame()
        
        # Add original columns
        if "id_column" in columns:
            output_df["sentence id"] = original_df[columns["id_column"]]
        else:
            # If no ID column, check for common ID column names
            for id_col in ["sentence id", "sentence_id", "id"]:
                if id_col in original_df.columns:
                    output_df["sentence id"] = original_df[id_col]
                    logger.info(f"Using {id_col} as ID column")
                    break
            else:
                # If still not found, create one
                output_df["sentence id"] = list(range(1, len(original_df) + 1))
                logger.info("Created new ID column")
        
        # Add text column
        if "text_column" in columns:
            output_df["sentence"] = original_df[columns["text_column"]]
        elif "sentence" in original_df.columns:
            output_df["sentence"] = original_df["sentence"]
            logger.info("Using 'sentence' column directly")
        else:
            # Look for any text-like column
            for text_col in ["text", "content", "message"]:
                if text_col in original_df.columns:
                    output_df["sentence"] = original_df[text_col]
                    logger.info(f"Using {text_col} as text column")
                    break
        
        # Add class-label column with special handling
        if "class-label" in original_df.columns:
            # Direct use of class-label if it exists
            output_df["class-label"] = original_df["class-label"]
            logger.info("Using 'class-label' column directly")
        elif "label_column" in columns:
            output_df["class-label"] = original_df[columns["label_column"]]
            logger.info(f"Using {columns['label_column']} as class-label")
        else:
            # If no label column, leave it empty
            output_df["class-label"] = [""] * len(original_df)
            logger.info("No class-label found, using empty values")
        
        # Add results columns with correct column names
        output_df["predicted-label"] = results_df["label"]
        output_df["explanation"] = results_df["explanation"]
        
        # Ensure column order matches the expected format
        output_df = output_df[["sentence id", "sentence", "class-label", "predicted-label", "explanation"]]
        
    elif task_type == "toxic":
        # Create a new dataframe with the required columns
        output_df = pd.DataFrame()
        
        # Add original columns
        if "id_column" in columns:
            output_df["data_id"] = original_df[columns["id_column"]]
        elif "data_id" in original_df.columns:
            output_df["data_id"] = original_df["data_id"]
            logger.info("Using 'data_id' column directly")
        else:
            # If no ID column, create one
            output_df["data_id"] = list(range(1, len(original_df) + 1))
            logger.info("Created new data_id column")
        
        # Add sample_id column
        if "sample_id_column" in columns:
            output_df["sample_id"] = original_df[columns["sample_id_column"]]
        elif "sample_id" in original_df.columns:
            output_df["sample_id"] = original_df["sample_id"]
            logger.info("Using 'sample_id' column directly")
        else:
            # If no sample_id column, use index
            output_df["sample_id"] = list(range(len(original_df)))
            logger.info("Created new sample_id column")
        
        # Add text column
        if "text_column" in columns:
            output_df["text"] = original_df[columns["text_column"]]
        elif "text" in original_df.columns:
            output_df["text"] = original_df["text"]
            logger.info("Using 'text' column directly")
        else:
            # Look for any text-like column
            for text_col in ["sentence", "content", "message"]:
                if text_col in original_df.columns:
                    output_df["text"] = original_df[text_col]
                    logger.info(f"Using {text_col} as text column")
                    break
        
        # Add source_label column
        if "source_label" in original_df.columns:
            output_df["source_label"] = original_df["source_label"]
            logger.info("Using 'source_label' column directly")
        elif "label_column" in columns:
            output_df["source_label"] = original_df[columns["label_column"]]
            logger.info(f"Using {columns['label_column']} as source_label")
        else:
            # If no label column, leave it empty
            output_df["source_label"] = [""] * len(original_df)
            logger.info("No source_label found, using empty values")
        
        # Add platform column
        if "platform_column" in columns:
            output_df["platform"] = original_df[columns["platform_column"]]
        elif "platform" in original_df.columns:
            output_df["platform"] = original_df["platform"]
            logger.info("Using 'platform' column directly")
        else:
            # If no platform column, leave it empty
            output_df["platform"] = [""] * len(original_df)
            logger.info("No platform column found, using empty values")
        
        # Add results columns - ensure correct column names
        output_df["predicted_label"] = results_df["toxicity_label"]
        output_df["rewritten_text"] = results_df["rewritten_text"]
        
        # Ensure column order matches the expected format
        output_df = output_df[["data_id", "sample_id", "text", "source_label", "platform", "predicted_label", "rewritten_text"]]
    
    # Debug the final output
    logger.debug(f"Final output columns: {output_df.columns.tolist()}")
    return output_df

# Update wrapper functions to include debug parameter
def process_sentiment_dataset(file_path: str, output_path: str, debug: bool = False) -> pd.DataFrame:
    """Wrapper for process_dataset with sentiment task type"""
    return process_dataset(file_path, output_path, "sentiment", debug=debug)

def process_toxicity_dataset(file_path: str, output_path: str, debug: bool = False) -> pd.DataFrame:
    """Wrapper for process_dataset with toxic task type"""
    return process_dataset(file_path, output_path, "toxic", debug=debug) 