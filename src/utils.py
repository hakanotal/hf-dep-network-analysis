"""
Utility functions for HuggingFace model analysis.
"""

import requests
import numpy as np
from typing import Dict, Any
from time import sleep

from .constants import (
    DEFAULT_MAX_LENGTH, 
    UNKNOWN,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    MIN_NODE_SIZE,
    MAX_NODE_SIZE,
    NODE_SIZE_SCALE
)


def fetch_config_architecture(model_id: str) -> str:
    """Fetch architecture from model's config.json file."""
    try:
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(config_url, timeout=REQUEST_TIMEOUT)
        sleep(REQUEST_DELAY)
        
        if response.status_code == 200:
            config_data = response.json()
            architectures = config_data.get("architectures", [])
            if architectures:
                return architectures[0]
        
        return UNKNOWN
    except Exception:
        return UNKNOWN


def truncate_string_fields(data: Dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH) -> Dict[str, Any]:
    """Recursively truncate string fields in metadata dictionary."""
    cleaned_data = {}
    
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_length:
            cleaned_data[key] = value[:max_length]
        elif isinstance(value, list):
            cleaned_list = []
            for item in value:
                if isinstance(item, str) and len(item) > max_length:
                    cleaned_list.append(item[:max_length])
                else:
                    cleaned_list.append(item)
            cleaned_data[key] = cleaned_list
        elif isinstance(value, dict):
            cleaned_data[key] = truncate_string_fields(value, max_length)
        else:
            cleaned_data[key] = value
    
    return cleaned_data


def create_minimal_metadata(model_id: str) -> Dict[str, Any]:
    """Create minimal metadata for models that can't be fetched."""
    return {
        'id': model_id,
        'author': model_id.split('/')[0] if '/' in model_id else UNKNOWN,
        'downloads': 0,
        'likes': 0,
        'pipeline_tag': None,
        'library_name': None,
        'tags': [],
        'created_at': None,
        'last_modified': None,
        'private': None,
        'gated': None,
        'card_data': None,
        'siblings_count': 0,
        'config_available': False,
        'transformers_info_available': False,
        'safetensors_available': False,
        'model_type': 'base',
        'model_arch': UNKNOWN,
        'base_models': [],
        'base_models_count': 0,
        'type_details': {}
    }


def determine_edge_type(model_type: str) -> str:
    """Determine edge type based on model type."""
    edge_type_mapping = {
        'quantized': 'quantized',
        'finetuned': 'finetuned',
        'merged': 'merged',
        'adapter': 'adapter'
    }
    return edge_type_mapping.get(model_type, 'derived')


def calculate_node_size(downloads: int) -> int:
    """Calculate node size based on downloads with logarithmic scaling."""
    if downloads > 0:
        return min(MAX_NODE_SIZE, max(MIN_NODE_SIZE, int(np.log10(downloads + 1) * NODE_SIZE_SCALE)))
    return MIN_NODE_SIZE 