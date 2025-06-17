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
from tqdm import tqdm

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
    
    # Prepare and save updated metadata with propagated architectures
    metadata_rows = []
    for node_id, node_data in G.nodes(data=True):
        # Extract relevant metadata from node
        metadata_row = {
            'id': node_id,
            'author': node_id.split('/')[0] if '/' in node_id else 'unknown',
            'downloads': 0,  # Default value - this would need to come from original data
            'likes': 0,  # Default value - this would need to come from original data
            'time': '',  # Default value - this would need to come from original data
            'last_modified': '',  # Default value - this would need to come from original data
            'private': False,  # Default value
            'gated': False,  # Default value
            'model_type': node_data.get('model_type', 'unknown'),
            'model_arch': node_data.get('model_arch', 'unknown')
        }
        metadata_rows.append(metadata_row)
    
    # Save updated metadata to CSV
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    print(f"Saving {len(metadata_rows)} nodes metadata to {metadata_file}...")
    
    # Read existing metadata.csv to preserve other fields if it exists
    existing_metadata = {}
    if os.path.exists(metadata_file):
        try:
            existing_df = pd.read_csv(metadata_file)
            for _, row in existing_df.iterrows():
                existing_metadata[row['id']] = row.to_dict()
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing metadata.csv: {e}")
    
    # Merge with existing metadata to preserve other fields
    final_metadata_rows = []
    for row in metadata_rows:
        if row['id'] in existing_metadata:
            # Merge with existing data, but update model_arch from graph
            merged_row = existing_metadata[row['id']].copy()
            merged_row['model_arch'] = row['model_arch']  # Update with propagated architecture
            final_metadata_rows.append(merged_row)
        else:
            final_metadata_rows.append(row)
    
    # Save to CSV
    if final_metadata_rows:
        metadata_df = pd.DataFrame(final_metadata_rows)
        metadata_df.to_csv(metadata_file, index=False)
        
        # Show architecture propagation results
        arch_counts = metadata_df['model_arch'].value_counts()
        print(f"✅ Updated metadata saved to {metadata_file}")
        print(f"📊 Architecture distribution after propagation:")
        for arch, count in arch_counts.items():
            print(f"  {arch}: {count} models")
    else:
        print(f"⚠️ No metadata to save")


def fetch_missing_models_from_graph(G: nx.DiGraph, metadata_list: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Find models in the graph that are missing from metadata_list and fetch their information.
    
    Args:
        G: NetworkX directed graph
        metadata_list: List of existing model metadata dictionaries
        verbose: Whether to print detailed information
        
    Returns:
        List of updated metadata including fetched missing models
    """
    
    # Get all node IDs from graph
    graph_nodes = set(G.nodes())
    
    # Get existing model IDs from metadata
    existing_models = set(metadata['id'] for metadata in metadata_list)
    
    # Find missing models
    missing_models = graph_nodes - existing_models
    
    if verbose:
        print(f"🔍 Graph contains {len(graph_nodes)} nodes")
        print(f"📊 Existing metadata covers {len(existing_models)} models")
        print(f"❓ Found {len(missing_models)} missing models to fetch")
    
    if len(missing_models) == 0:
        if verbose:
            print("✅ No missing models found!")
        return metadata_list
    
    if verbose:
        print(f"\n🔍 Missing models to fetch:")
        for model in list(missing_models)[:5]:  # Show first 5
            print(f"  - {model}")
        if len(missing_models) > 5:
            print(f"  ... and {len(missing_models) - 5} more")
    
    # Fetch missing models one by one
    new_metadata = []
    successful_fetches = 0
    failed_fetches = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(missing_models), total=len(missing_models), 
                       desc="🔍 Fetching missing models", unit="model")
    
    for i, model_id in progress_bar:
        progress_bar.set_postfix(model=model_id.split('/')[-1][:20])
        
        try:
            # Fetch model info
            model = model_info(model_id, files_metadata=False)  # Skip file metadata for speed
            
            # Extract metadata
            metadata = extract_model_metadata(model, verbose=False)
            new_metadata.append(metadata)
            successful_fetches += 1
            
            if verbose:
                tqdm.write(f"  ✅ Successfully fetched: {model_id}")
                
        except Exception as e:
            failed_fetches += 1
            if verbose:
                tqdm.write(f"  ❌ Failed to fetch {model_id}: {str(e)[:100]}")
            
            # Create minimal metadata for missing model
            minimal_metadata = create_minimal_metadata(model_id)
            new_metadata.append(minimal_metadata)
            
            if verbose:
                tqdm.write(f"  ⚠️ Created minimal metadata for {model_id}")
    
    progress_bar.close()
    
    # Combine existing and new metadata
    updated_metadata_list = metadata_list + new_metadata

    # Save updated metadata to CSV
    updated_metadata_df = pd.DataFrame(updated_metadata_list)
    essential_columns = ['id', 'author', 'downloads', 'likes', 'created_at', 'last_modified', 
                        'private', 'gated', 'model_type', 'model_arch', 'base_models', 'base_models_count', 'type_details']
    updated_metadata_df = updated_metadata_df[essential_columns]
    updated_metadata_df.rename(columns={'created_at': 'time'}, inplace=True)
    updated_metadata_df.to_csv('network_data/metadata.csv', index=False)
    
    if verbose:
        print(f"\n✅ MISSING MODELS FETCH COMPLETE")
        print(f"📊 Successfully fetched: {successful_fetches}")
        print(f"❌ Failed to fetch: {failed_fetches}")
        print(f"🔢 Total models in metadata: {len(updated_metadata_list)}")
        
        if failed_fetches > 0:
            print(f"💡 Failed models were added with minimal metadata")
    
    return updated_metadata_list