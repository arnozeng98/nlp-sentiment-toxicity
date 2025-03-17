#!/usr/bin/env python
# coding: utf-8
"""
Main test runner for all NLP tests
"""

import os
import sys
import time
import subprocess
import warnings
from datetime import datetime
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Only show error messages
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Add parent directory to path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Import test configuration
from test_config import setup_test_logging

# Setup logger
logger = setup_test_logging()

def run_test(test_script):
    """Run a test script and return status"""
    test_name = os.path.basename(test_script).replace('.py', '')
    logger.info(f"Running {test_name}...")
    
    try:
        start_time = time.time()
        # Run the test script as a subprocess
        # Change directory to tests directory to make relative imports work
        result = subprocess.run(
            [sys.executable, os.path.basename(test_script)], 
            capture_output=True, 
            text=True,
            check=True,
            cwd=os.path.dirname(test_script)  # Run from the tests directory
        )
        end_time = time.time()
        
        logger.info(f"{test_name} completed in {end_time - start_time:.2f} seconds")
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{test_name} failed with exit code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr

def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info(f"Starting test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # Get all test scripts (except this one and __init__.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_scripts = [
        os.path.join(current_dir, f) for f in os.listdir(current_dir)
        if f.startswith('test_') and f.endswith('.py') and f != 'test_config.py' and f != os.path.basename(__file__)
    ]
    
    logger.info(f"Found {len(test_scripts)} test scripts: {[os.path.basename(s) for s in test_scripts]}")
    
    # Run all tests with progress bar
    results = []
    start_time = time.time()
    
    # Create progress bar for test suite
    with tqdm(total=len(test_scripts), desc="Running test suite", unit="test") as test_progress:
        for script in test_scripts:
            test_name = os.path.basename(script).replace('.py', '')
            test_progress.set_description(f"Testing {test_name}")
            
            success, output = run_test(script)
            results.append((os.path.basename(script), success))
            
            # Update progress bar
            if success:
                test_progress.set_postfix_str("✅ PASSED")
            else:
                test_progress.set_postfix_str("❌ FAILED")
            test_progress.update(1)
    
    total_time = time.time() - start_time
    
    # Summarize results
    logger.info("=" * 50)
    logger.info("Test Run Summary")
    logger.info("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for script, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{script}: {status}")
    
    logger.info("-" * 50)
    logger.info(f"Tests passed: {passed}/{len(results)}")
    logger.info(f"Tests failed: {failed}/{len(results)}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("=" * 50)
    
    if failed > 0:
        sys.exit(1)
    
if __name__ == "__main__":
    main() 