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

# Test data for toxicity analysis - using first 5 samples from the original dataset
TEST_TOXICITY_DATA = [
    "I know he's an idiot I think he might of been held back a year in school then again Texans aren't really smart or average for that matter.",
    "==my butthole==  i love buttholes. they taste mmm goood. i will never stop vandalizing wikipedia. u no why?  cuz u dont tell me wat to do. k bye i hate you",
    "This is absolute bullshit! We fucking knew this shit was going down. Now we damn well need to fucking dig into those fucking voting machines 'cause I'm goddamn sure Trump ain't fucking lying about that fucking voter fraud!",
    "\"Moochelle Nobama takes swipe at Trump it sez. Trump better look out, apes have a long reach.\"",
    "2. 	> Lol, I guessed IPN (the Institue of National Remembrance) was going to be mentioned  From what I understood from their message on IPN's page, they actually went after that cunt that modified the trailer (and has two more cunty videos on his channel), not after the actual game itself.  The journalists however - especially the rightwing ones - seen IPN's message and went after everything. No surprise there, lol.  >  Germans are the last people who would make that mistake with the phrasing.  You would think so. Sadly, you would be wrong.  - http://wyborcza.pl/1,76842,6928930,_Polish_Camps__in_Polish_Court.html - https://fakty.interia.pl/news-byly-wiezien-auschwitz-skarzy-zdf-za-polskie-obozy,nId,999725"
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
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(test_output_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a file handler with timestamp
    log_file_name = os.path.join(logs_dir, f"test_run_{timestamp}.log")
    f_handler = logging.FileHandler(log_file_name)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatters to handlers
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    # Log the log file location
    logger.info(f"Logging to file: {log_file_name}")
    
    return logger 