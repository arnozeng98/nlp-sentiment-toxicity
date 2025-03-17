#!/usr/bin/env python
# coding: utf-8
"""
Configuration for tests
"""

import os
import logging
import datetime

# Test data for sentiment analysis
TEST_SENTIMENT_DATA = [
    "I really enjoyed this movie, it was fantastic!",
    "The service at this restaurant was terrible and the food was cold.",
    "The product works fine but the packaging was damaged.",
    "This is absolutely the worst experience I've ever had.",
    "The new update has improved performance significantly."
]

# Test data for toxicity analysis
TEST_TOXICITY_DATA = [
    "You're an idiot and nobody likes you!",
    "I respectfully disagree with your opinion on this matter.",
    "Shut up and go back to where you came from!",
    "Thank you for sharing your perspective, I appreciate it.",
    "This post is full of lies and misinformation, you're stupid."
]

# Output file paths with timestamps
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Use absolute path for test output directory to avoid path issues
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_output_dir = os.path.join(root_dir, "test_output")

# Create output directory if it doesn't exist
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Define output file paths with absolute paths
SENTIMENT_TEST_OUTPUT = os.path.join(test_output_dir, f"sentiment_test_results_{timestamp}.csv")
TOXIC_TEST_OUTPUT = os.path.join(test_output_dir, f"toxicity_test_results_{timestamp}.csv")

def setup_test_logging():
    """Setup logging for tests"""
    # Configure logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatters to handlers
    c_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    
    return logger 