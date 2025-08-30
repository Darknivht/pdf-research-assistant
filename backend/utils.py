"""
Utility functions for configuration, caching, file I/O, and logging.
"""

import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management class."""
    
    def __init__(self):
        # LLM Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.2-3b-instruct:free')
        
        # Embedding Configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Storage Paths
        self.index_path = os.getenv('INDEX_PATH', './storage/index.faiss')
        self.docstore_path = os.getenv('DOCSTORE_PATH', './storage/docstore.pkl')
        self.metadata_path = os.getenv('METADATA_PATH', './storage/metadata.json')
        
        # RAG Configuration
        self.top_k = int(os.getenv('TOP_K', '4'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '512'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.2'))
        
        # Logging
        self.enable_logging = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            os.path.dirname(self.index_path),
            os.path.dirname(self.docstore_path),
            os.path.dirname(self.metadata_path),
            './logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    Path('./logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler('./logs/app.log'),
            logging.StreamHandler()
        ]
    )

def save_json(data: Any, filepath: str) -> bool:
    """Save data as JSON to file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Any]:
    """Load JSON data from file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return None

def save_pickle(data: Any, filepath: str) -> bool:
    """Save data as pickle to file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logging.error(f"Error saving pickle to {filepath}: {e}")
        return False

def load_pickle(filepath: str) -> Optional[Any]:
    """Load pickle data from file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        logging.error(f"Error loading pickle from {filepath}: {e}")
        return None

def log_qa_pair(question: str, answer: str, sources: List[Dict], 
               metadata: Dict = None, log_file: str = './logs/qa_log.jsonl'):
    """Log question-answer pairs for evaluation."""
    try:
        qa_entry = {
            'timestamp': str(pd.Timestamp.now()),
            'question': question,
            'answer': answer,
            'sources': sources,
            'metadata': metadata or {}
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        logging.error(f"Error logging QA pair: {e}")

def get_file_hash(filepath: str) -> str:
    """Get file hash for caching."""
    import hashlib
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Error getting file hash for {filepath}: {e}")
        return ""

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

# Import pandas for timestamp handling
try:
    import pandas as pd
except ImportError:
    logging.warning("Pandas not available for timestamp handling")
    pd = None