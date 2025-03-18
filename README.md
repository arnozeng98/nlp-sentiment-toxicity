# NLP Sentiment & Toxicity Analysis Toolkit

A comprehensive NLP toolkit for multi-language sentiment analysis, toxicity detection, and text detoxification using state-of-the-art language models.

## Project Introduction

This toolkit provides a set of modular NLP tools designed to:

1. **Detect languages** in text using FastText
2. **Translate non-English text** to English for unified analysis
3. **Analyze sentiment** (positive/negative/mixed) with detailed explanations
4. **Detect toxic content** with contextual understanding
5. **Detoxify text** by rewriting toxic content in a polite, constructive manner

The system is designed to handle batch processing of text datasets with robust error handling, caching, and adaptive prompting to optimize results from language models.

## Project Structure

```bash
nlp-sentiment-toxicity/
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration settings
│   ├── models.py          # Model loading utilities
│   ├── language_utils.py  # Language detection and translation
│   ├── text_processor.py  # Text processing and batch handling
│   ├── data_processor.py  # Data processing functions
│   ├── analysis.py        # Core analysis functions
│   ├── sentiment.py       # Sentiment analysis module
│   ├── toxicity.py        # Toxicity detection and detoxification
│   ├── processing.py      # Text preprocessing and rule-based processing
│   └── output_parser.py   # Output parsing and fixing
│
├── tests/                 # Test modules
│   ├── __init__.py        # Test package initialization
│   ├── test_config.py     # Test configuration
│   ├── test_sentiment.py  # Sentiment analysis tests
│   ├── test_toxicity.py   # Toxicity analysis tests
│   └── run_all_tests.py   # Test runner
│
├── models/                # Stores model files
│   └── lid.176.bin        # FastText language identification model
│
├── data/                  # Input datasets
│   ├── multilingual-sentiment-test-solutions.csv
│   └── toxic-test-solutions.csv
│
├── output/                # Results output directory
│   └── ...                # Generated result files
│
├── test_output/           # Test results and logs
│   ├── logs/              # Test log files
│   └── ...                # Test output files
│
└── README.md              # Project documentation
```

## Technologies Used

- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework for model inference
- **Transformers (Hugging Face)**: For accessing and using pre-trained models
- **FastText**: For language detection
- **LangChain**: For agent-based processing and tool integration
- **Pandas**: For data manipulation and CSV processing
- **Ollama**: For accessing lightweight LLMs
- **Logging**: For comprehensive tracking and debugging

## Models

The toolkit uses several specialized models:

1. **Language Detection**: FastText's `lid.176.bin` model (supports 176 languages)
2. **Translation**: `UBC-NLP/toucan-base` (Multilingual MT5-based model)
3. **Sentiment & Toxicity Analysis**: `ibm-granite/granite-3.0-2b-instruct` (Instruction-tuned LLM)
4. **Agent-based Processing**: `llama3.2:1b` via Ollama (Local lightweight LLM)

Each model is used for its specific strengths to create a robust pipeline for text analysis.

## Setup and Running

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Ollama installed locally (for agent-based processing)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nlp-sentiment-toxicity.git
   cd nlp-sentiment-toxicity
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install torch pandas tqdm transformers fasttext langchain-ollama
   ```

4. Download required models:

   ```bash
   # FastText language model needs to be downloaded manually to models/ directory
   mkdir -p models
   curl -o models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

5. Set up Ollama (if not already installed):

   ```bash
   # Follow instructions at https://ollama.ai/
   # Then pull the required model
   ollama pull llama3.2:1b
   ```

### Running the Main Program

To process the datasets:

```bash
python -m src.main
```

This will:

1. Load all required models
2. Process sentiment analysis on multilingual data
3. Process toxicity analysis and detoxification
4. Save results to the `output/` directory

### Running Tests

To run individual tests:

```bash
cd tests  # Go to the tests directory first
python test_sentiment.py  # For sentiment analysis tests
python test_toxicity.py   # For toxicity analysis tests
```

To run all tests:

```bash
cd tests  # Go to the tests directory first
python run_all_tests.py
```

Test results will be saved in the `test_output/` directory with timestamped filenames, and logs will be stored in `test_output/logs/`.

## Configuration

Key configuration settings can be adjusted in `src/config.py`, including:

- Model paths
- Input/output file paths
- Processing parameters (retries, delay between requests)
- Device selection (CUDA/CPU)

## Performance Notes

- Processing large datasets may take significant time
- GPU acceleration is strongly recommended for optimal performance
- The system includes caching to avoid reprocessing identical texts
- Adaptive prompting improves reliability with large language models
- Direct function implementations ensure independence between sentiment and toxicity analysis

## New Features and Improvements

- **Enhanced Sentiment Analysis**: Improved sentiment analysis templates and output parsing to avoid confusion with toxicity analysis
- **Optimized Toxicity Detection**: Added direct function implementations for better toxicity detection accuracy
- **Improved Detoxification**: Using specialized direct functions for more effective text detoxification
- **Better Logging**: Enhanced logging functionality including test run logs
- **Improved Error Handling**: Better error handling and fallback mechanisms

## Troubleshooting

### Import Errors in Tests

If you encounter import errors when running tests, make sure to:

1. **Run tests from the test directory**: Always navigate to the `tests` directory before running test scripts

   ```bash
   cd tests
   python test_sentiment.py
   ```

2. **Python Module Path**: If you're running tests as modules with `-m`, make sure you're in the project root directory:

   ```bash
   # From project root
   PYTHONPATH=. python -m tests.test_sentiment  # Linux/Mac
   set PYTHONPATH=. && python -m tests.test_sentiment  # Windows
   ```

3. **Missing Modules**: If you see errors about missing modules, ensure you've installed all dependencies:

   ```bash
   pip install -r requirements.txt  # If available
   # or manually install required packages
   pip install torch pandas tqdm transformers fasttext langchain-ollama
   ```

### TensorFlow Warnings

If you see warnings like:

```bash
I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results...
```

These are informational messages from TensorFlow about optimizations. The code already contains fixes to suppress these messages, but if they still appear, you can:

1. **Additional Environment Variables**: Set these environment variables before running your script:

   ```bash
   # Linux/Mac
   export TF_ENABLE_ONEDNN_OPTS=0
   export TF_CPP_MIN_LOG_LEVEL=2
   
   # Windows
   set TF_ENABLE_ONEDNN_OPTS=0
   set TF_CPP_MIN_LOG_LEVEL=2
   ```

2. **Run with Python Flag**: Use the `-W` flag to ignore warnings:

   ```bash
   python -W ignore test_sentiment.py
   ```

3. **Alternative TensorFlow Installation**: Consider installing the CPU-only version of TensorFlow if you're not using its GPU features:

   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```
