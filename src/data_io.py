"""
Data import/export functions for saving analysis results and managing network data.
"""

import os
import csv
import json
import pandas as pd
import networkx as nx
from typing import Dict, Any, List
from huggingface_hub import model_info

from .utils import create_minimal_metadata
from .analysis import extract_model_metadata


def save_analysis_results(results: Dict[str, Any], output_prefix: str = "hf_analysis") -> None:
    """Save analysis results to files and update network data."""
    if not results['success']:
        print(f"❌ Cannot save results - analysis failed")
        return
    
    # Prepare metadata
    if 'metadata' in results and not isinstance(results['metadata'], list):
        metadata_list = [results['metadata']]
        filename_suffix = f"_{results['model_name'].replace('/', '_')}"
    else:
        metadata_list = results['metadata']
        filename_suffix = f"_{results['total_models']}_models"
    
    # Save to main model_data directory
    df = pd.DataFrame(metadata_list)
    os.makedirs("model_data", exist_ok=True)
    
    csv_file = f"model_data/{output_prefix}{filename_suffix}.csv"
    df.to_csv(csv_file, index=False)
    
    json_file = f"model_data/{output_prefix}{filename_suffix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False, default=str)
    
    # Update network_data directory
    _update_network_data(df)
    
    print(f"💾 Saved CSV: {csv_file}")
    print(f"💾 Saved JSON: {json_file}")
    
    # Fetch missing models automatically
    fetch_missing_models()


def _update_network_data(df: pd.DataFrame) -> None:
    """Update network_data directory with essential metadata."""
    network_data_dir = "network_data"
    os.makedirs(network_data_dir, exist_ok=True)
    
    # Select essential columns for network analysis
    essential_columns = ['id', 'author', 'downloads', 'likes', 'created_at', 'last_modified', 
                        'private', 'gated', 'model_type', 'model_arch']
    
    metadata_df = df[essential_columns].copy()
    metadata_df.rename(columns={'created_at': 'time'}, inplace=True)
    metadata_df.to_csv(f"{network_data_dir}/metadata.csv", index=False)


def export_network_csv(G: nx.DiGraph) -> None:
    """
    Export network data in CSV format compatible with external analysis tools.
    
    Args:
        G: NetworkX directed graph
    """
    # Ensure directory exists
    output_dir: str = "network_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare edges data
    edges = []
    for source, target, data in G.edges(data=True):
        edges.append({
            'source': source,
            'target': target,
            'edge': data.get('edge_type', 'unknown')
        })
    
    # Save edges to CSV
    edges_file = os.path.join(output_dir, 'edges.csv')
    print(f"\nSaving {len(edges)} edges to {edges_file}...")
    with open(edges_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['source', 'target', 'edge'])
        writer.writeheader()
        writer.writerows(edges)
    print(f"✅ Edges saved to {edges_file}")


def fetch_missing_models(network_data_dir: str = "network_data", verbose: bool = False) -> None:
    """Find models referenced in edges.csv but missing from metadata.csv and fetch their metadata."""
    edges_file = os.path.join(network_data_dir, 'edges.csv')
    metadata_file = os.path.join(network_data_dir, 'metadata.csv')
    
    # Check if files exist
    if not os.path.exists(edges_file) or not os.path.exists(metadata_file):
        if verbose:
            print(f"⚠️ Network data files not found in {network_data_dir}")
        return
    
    if verbose:
        print(f"🔍 Checking for missing models in network data...")
    
    # Read existing data
    edges_df = pd.read_csv(edges_file)
    metadata_df = pd.read_csv(metadata_file)
    
    # Get all unique model IDs from edges
    all_edge_models = set()
    all_edge_models.update(edges_df['source'].unique())
    all_edge_models.update(edges_df['target'].unique())
    
    # Find missing models
    existing_models = set(metadata_df['id'].unique())
    missing_models = all_edge_models - existing_models
    
    if verbose:
        print(f"📊 Found {len(missing_models)} missing models")
    
    if len(missing_models) == 0:
        if verbose:
            print("✅ No missing models found!")
        return
    
    # Fetch and update
    new_metadata = _fetch_models_list(list(missing_models), verbose)
    
    if new_metadata:
        # Update metadata file
        new_df = pd.DataFrame(new_metadata)
        updated_df = pd.concat([metadata_df, new_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=['id'], keep='first')
        updated_df.to_csv(metadata_file, index=False)
        
        if verbose:
            print(f"✅ Updated {metadata_file} with {len(new_metadata)} new models")


def _fetch_models_list(model_ids: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """Fetch metadata for a list of model IDs."""
    new_metadata = []
    successful_fetches = 0
    
    for i, model_id in enumerate(model_ids):
        if verbose:
            print(f"📋 Fetching {i+1}/{len(model_ids)}: {model_id}")
        
        try:
            model = model_info(model_id, files_metadata=False)
            metadata = extract_model_metadata(model, verbose=False)
            new_metadata.append(metadata)
            successful_fetches += 1
            
            if verbose:
                print(f"  ✅ Successfully fetched: {model_id}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to fetch {model_id}: {str(e)[:100]}")
            
            # Create minimal metadata for missing model
            minimal_metadata = create_minimal_metadata(model_id)
            new_metadata.append(minimal_metadata)
    
    if verbose:
        print(f"📊 Successfully fetched: {successful_fetches}/{len(model_ids)}")
    
    return new_metadata


# Re-export from network module to maintain backward compatibility  
from .network import analyze_network_metrics, print_network_summary 