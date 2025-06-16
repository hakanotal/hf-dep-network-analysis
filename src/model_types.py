"""
Model type detection and classification functions.
"""

import re
from typing import List, Dict, Any

from .constants import MODEL_TYPE_PATTERNS


def detect_model_type_and_base_models(tags: List[str]) -> Dict[str, Any]:
    """
    Detect model type and extract base model information from tags.
    
    Args:
        tags: List of model tags
        
    Returns:
        dict: {
            'type': str,  # 'base', 'quantized', 'finetuned', 'merged', 'adapter'
            'base_models': list,  # List of base model names
            'type_details': dict  # Additional details about the model type
        }
    """
    if not tags:
        return {
            'type': 'base',
            'base_models': [],
            'type_details': {}
        }
    
    # Convert to list if it's a string representation
    if isinstance(tags, str):
        try:
            tags = eval(tags)
        except:
            tags = [tags]
    
    # Track found types
    found_types = {
        'quantized': [],
        'finetune': [],
        'merge': [],
        'adapter': [],
        'generic_base': []
    }
    
    # Search through all tags
    for tag in tags:
        if isinstance(tag, str):
            for type_name, pattern in MODEL_TYPE_PATTERNS.items():
                matches = re.findall(pattern, tag)
                if matches:
                    found_types[type_name].extend(matches)
    
    # Determine model type based on found patterns
    if found_types['quantized']:
        model_type = 'quantized'
        base_models = found_types['quantized']
        type_details = {
            'quantization_source': found_types['quantized'][0] if found_types['quantized'] else None
        }
    elif found_types['adapter']:
        model_type = 'adapter'
        base_models = found_types['adapter']
        type_details = {
            'adapter_source': found_types['adapter'][0] if found_types['adapter'] else None
        }
    elif len(found_types['merge']) >= 2:
        model_type = 'merged'
        base_models = found_types['merge']
        type_details = {
            'merge_count': len(found_types['merge']),
            'merged_models': found_types['merge']
        }
    elif found_types['finetune'] or found_types['generic_base']:
        model_type = 'finetuned'
        base_models = found_types['finetune'] + found_types['generic_base']
        type_details = {
            'finetune_source': base_models[0] if base_models else None,
            'has_explicit_finetune_tag': bool(found_types['finetune'])
        }
    else:
        model_type = 'base'
        base_models = []
        type_details = {}
    
    return {
        'type': model_type,
        'base_models': base_models,
        'type_details': type_details
    }


def format_model_type_info(type_info: Dict[str, Any]) -> str:
    """Format model type information for display."""
    model_type = type_info['type']
    details = type_info['type_details']
    
    if model_type == 'base':
        return "🔵 BASE MODEL"
    elif model_type == 'quantized':
        source = details.get('quantization_source', 'Unknown')
        return f"⚡ QUANTIZED (from: {source})"
    elif model_type == 'finetuned':
        source = details.get('finetune_source', 'Unknown')
        explicit = details.get('has_explicit_finetune_tag', False)
        tag_type = "explicit" if explicit else "inferred"
        return f"🎯 FINETUNED (from: {source}) [{tag_type}]"
    elif model_type == 'merged':
        count = details.get('merge_count', 0)
        models = details.get('merged_models', [])
        models_str = ', '.join(models[:2])  # Show first 2
        if len(models) > 2:
            models_str += f" + {len(models) - 2} more"
        return f"🔗 MERGED ({count} models: {models_str})"
    elif model_type == 'adapter':
        source = details.get('adapter_source', 'Unknown')
        return f"🔧 ADAPTER (from: {source})"
    
    return f"❓ UNKNOWN ({model_type})" 