"""
insanityllm.config
Environment configuration utilities for Insanity LLM project.
"""
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def setup_environment(env_file: Optional[str] = None) -> dict[str, str]:
    project_root = Path(__file__).parent.parent
    if load_dotenv is not None:
        env_path = env_file or project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"No .env file found at: {env_path}")
    config = {
        'DATASETS_DIR': os.getenv('DATASETS_DIR', 'data'),
        'CACHE_DIR': os.getenv('CACHE_DIR', 'cache'),
        'HF_DATASETS_CACHE': os.getenv('HF_DATASETS_CACHE', 'cache/huggingface/datasets'),
        'HF_MODELS_CACHE': os.getenv('HF_MODELS_CACHE', 'cache/huggingface/models'),
        'MODELS_DIR': os.getenv('MODELS_DIR', 'models'),
        'LOGS_DIR': os.getenv('LOGS_DIR', 'logs'),
    }
    for key in config:
        if not os.path.isabs(config[key]):
            config[key] = str(project_root / config[key])
    os.environ['HF_DATASETS_CACHE'] = config['HF_DATASETS_CACHE']
    os.environ['HF_HUB_CACHE'] = config['HF_MODELS_CACHE']
    os.environ['TRANSFORMERS_CACHE'] = config['HF_MODELS_CACHE']
    for key, path in config.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    return config
