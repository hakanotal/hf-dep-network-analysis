"""
Model analysis functions for extracting and analyzing HuggingFace model metadata.
"""

from huggingface_hub import HfApi, model_info
from typing import Dict, Any, List
from pprint import pprint
from tqdm import tqdm
import pandas as pd

from .utils import truncate_string_fields
from .model_types import detect_model_type_and_base_models, format_model_type_info
from .constants import UNKNOWN


def extract_model_metadata(model: Any, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a single model object.
    
    Args:
        model: HuggingFace model object
        verbose: Whether to print detailed information
        
    Returns:
        dict: Comprehensive model metadata
    """
    if verbose:
        print(f"{'='*60}")
        print(f"MODEL: {model.id}")
        print(f"{'='*60}")
    
    # MODEL TYPE DETECTION
    model_tags = getattr(model, 'tags', [])
    type_info = detect_model_type_and_base_models(model_tags)
    type_display = format_model_type_info(type_info)
    
    if verbose:
        print(f"\n🤖 MODEL TYPE ANALYSIS:")
        print(f"📊 Type: {type_display}")
        if type_info['base_models']:
            print(f"🎯 Base Models: {', '.join(type_info['base_models'])}")
        
        # Core Information
        print(f"\n📝 CORE INFO:")
        print(f"📝 Model ID: {getattr(model, 'id', 'N/A')}")
        print(f"👤 Author: {getattr(model, 'author', 'N/A')}")
        print(f"📅 Created: {getattr(model, 'created_at', 'N/A')}")
        print(f"📅 Modified: {getattr(model, 'last_modified', 'N/A')}")
        
        # Popularity
        print(f"\n📊 POPULARITY:")
        print(f"⬇️ Downloads: {getattr(model, 'downloads', 'N/A')}")
        print(f"👍 Likes: {getattr(model, 'likes', 'N/A')}")
        
        # Technical Info
        print(f"\n🔧 TECHNICAL:")
        print(f"🎯 Pipeline: {getattr(model, 'pipeline_tag', 'N/A')}")
        print(f"📚 Library: {getattr(model, 'library_name', 'N/A')}")
        print(f"🔒 Private: {getattr(model, 'private', 'N/A')}")
        print(f"🚪 Gated: {getattr(model, 'gated', 'N/A')}")
    
    # Get additional metadata
    card_data = getattr(model, 'card_data', None)
    config = getattr(model, 'config', None)
    transformers_info = getattr(model, 'transformers_info', None)
    safetensors = getattr(model, 'safetensors', None)
    siblings = getattr(model, 'siblings', None)
    
    if verbose and card_data:
        print(f"\n📄 MODEL CARD:")
        print(f"🌐 Language: {getattr(card_data, 'language', 'N/A')}")
        print(f"📄 License: {getattr(card_data, 'license', 'N/A')}")
        print(f"📊 Datasets: {getattr(card_data, 'datasets', 'N/A')}")
    
    if verbose and siblings:
        print(f"\n📁 FILES: {len(siblings)} total")
        for i, sibling in enumerate(siblings[:5]):  # Show first 5
            print(f"  📄 {getattr(sibling, 'rfilename', 'N/A')}")
        if len(siblings) > 5:
            print(f"  ... and {len(siblings) - 5} more files")
    
    model_arch = UNKNOWN
    if model.config is not None and "model_type" in model.config:
        model_arch = model.config["model_type"]

    # Collect structured data
    model_metadata = {
        'id': getattr(model, 'id', None),
        'author': getattr(model, 'author', None),
        'downloads': getattr(model, 'downloads', 0),
        'likes': getattr(model, 'likes', 0),
        'pipeline_tag': getattr(model, 'pipeline_tag', None),
        'library_name': getattr(model, 'library_name', None),
        'tags': getattr(model, 'tags', []),
        'created_at': getattr(model, 'created_at', None),
        'last_modified': getattr(model, 'last_modified', None),
        'private': getattr(model, 'private', None),
        'gated': getattr(model, 'gated', None),
        'card_data': card_data.__dict__ if card_data and hasattr(card_data, '__dict__') else None,
        'siblings_count': len(siblings) if siblings else 0,
        'config_available': config is not None,
        'transformers_info_available': transformers_info is not None,
        'safetensors_available': safetensors is not None,
        # Model type fields
        'model_type': type_info['type'],
        'model_arch': model_arch,
        'base_models': type_info['base_models'],
        'base_models_count': len(type_info['base_models']),
        'type_details': type_info['type_details']
    }
    
    if verbose:
        print(f"\n{'='*60}\n")
    
    return truncate_string_fields(model_metadata)


def analyze_single_model(model_name: str, hf_api: HfApi, verbose: bool = True) -> Dict[str, Any]:
    """Analyze a single HuggingFace model."""
    if verbose:
        print(f"\n🎯 ANALYZING MODEL: {model_name}")
        print(f"Fetching detailed metadata...\n")
    
    try:
        model = model_info(model_name, files_metadata=True)
        if verbose:
            pprint(model)

        metadata = extract_model_metadata(model, verbose=verbose)
        if verbose:
            pprint(metadata)
            type_display = format_model_type_info({
                'type': metadata['model_type'],
                'base_models': metadata['base_models'],
                'type_details': metadata['type_details']
            })
            print(f"✅ Successfully analyzed: {model_name}")
            print(f"📊 Model Type: {type_display}")
            if metadata['base_models']:
                print(f"🎯 Dependencies: {', '.join(metadata['base_models'])}")
        
        return {
            'success': True,
            'model_name': model_name,
            'metadata': metadata
        }
        
    except Exception as e:
        error_msg = f"❌ Error analyzing model {model_name}: {e}"
        if verbose:
            print(error_msg)
        return {
            'success': False,
            'model_name': model_name,
            'error': str(e)
        }


def analyze_model_list(hf_api: HfApi, 
                      pipeline_tag: List[str] = ["text-generation"], 
                      sort: str = "downloads", 
                      limit: int = 10, 
                      verbose: bool = False) -> Dict[str, Any]:
    """Analyze a list of HuggingFace models."""
    print(f"\n📊 ANALYZING {limit} MODELS")
    print(f"Pipeline: {pipeline_tag}, Sort: {sort}")
    print("=" * 60)
    
    # Get models with full metadata
    models = hf_api.list_models(
        pipeline_tag=pipeline_tag,
        sort=sort,
        language="en",
        limit=limit,
        cardData=True,
        full=True,
        fetch_config=True
    )
    
    model_metadata = []
    type_counts = {'base': 0, 'finetuned': 0, 'quantized': 0, 'merged': 0, 'adapter': 0}
    
    # Convert to list to get proper length for tqdm
    models_list = list(models)
    
    # Use tqdm with proper description and total
    for i, model in enumerate(tqdm(models_list, desc="Analyzing models", unit="model")):
        if verbose:
            print(f"\n📋 Analyzing model {i+1}/{len(models_list)}: {model.id}")
        
        metadata = extract_model_metadata(model, verbose=verbose)
        model_metadata.append(metadata)
        
        # Count model types
        model_type = metadata['model_type']
        if model_type in type_counts:
            type_counts[model_type] += 1
        
        # Show brief summary unless verbose (but don't print during tqdm to avoid interference)
        if not verbose:
            # Update tqdm description with current model info
            type_display = format_model_type_info({
                'type': metadata['model_type'],
                'base_models': metadata['base_models'],
                'type_details': metadata['type_details']
            })
    
    # Create summary
    _print_analysis_summary(model_metadata, type_counts)
    
    return {
        'success': True,
        'total_models': len(model_metadata),
        'metadata': model_metadata,
        'type_distribution': type_counts,
        'dataframe': pd.DataFrame(model_metadata)
    }


def _print_analysis_summary(model_metadata: List[Dict], type_counts: Dict[str, int]) -> None:
    """Print analysis summary statistics."""
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    df = pd.DataFrame(model_metadata)
    
    print(f"\n🤖 MODEL TYPE DISTRIBUTION:")
    for type_name, count in type_counts.items():
        if count > 0:
            percentage = (count / len(model_metadata)) * 100
            print(f"  {type_name.upper()}: {count} models ({percentage:.1f}%)")
    
    if len(model_metadata) > 0:
        print(f"\n📈 TOP PERFORMERS:")
        top_downloads = df.nlargest(min(3, len(df)), 'downloads')
        for idx, row in top_downloads.iterrows():
            print(f"  📈 {row['id']}: {row['downloads']:,} downloads")
    else:
        print(f"\n⚠️ No models found with the specified criteria") 