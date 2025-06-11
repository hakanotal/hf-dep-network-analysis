from huggingface_hub import HfApi, list_models, model_info
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pprint import pprint
import networkx as nx
import pandas as pd
import numpy as np
import csv
import json
import re
import os

# ============================================================================
# NETWORK GRAPH FUNCTIONS
# ============================================================================

def build_model_network(analysis_results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                       include_isolated: bool = False) -> nx.DiGraph:
    """
    Build a directed network graph of model relationships.
    
    Args:
        analysis_results: Results from analyze_single_model or analyze_model_list,
                         or a list of model metadata dictionaries
        include_isolated: Whether to include models with no relationships as isolated nodes
        
    Returns:
        nx.DiGraph: Directed graph with model relationships
    """
    G = nx.DiGraph()
    
    # Extract metadata list from different input formats
    if isinstance(analysis_results, dict):
        if 'metadata' in analysis_results:
            # Single model or list results
            metadata_list = analysis_results['metadata']
            if not isinstance(metadata_list, list):
                metadata_list = [metadata_list]
        else:
            # Direct metadata dict
            metadata_list = [analysis_results]
    elif isinstance(analysis_results, list):
        # Direct list of metadata
        metadata_list = analysis_results
    else:
        raise ValueError("Invalid analysis_results format")
    
    # Track all models mentioned (including base models not in our dataset)
    all_models = set()
    model_relationships = []
    
    # First pass: collect all models and relationships
    for metadata in metadata_list:
        model_id = metadata['id']
        model_type = metadata['model_type']
        base_models = metadata.get('base_models', [])
        type_details = metadata.get('type_details', {})
        
        all_models.add(model_id)
        
        # Add relationships based on model type
        if model_type != 'base' and base_models:
            for base_model in base_models:
                all_models.add(base_model)
                
                # Determine edge type based on model type
                if model_type == 'quantized':
                    edge_type = 'quantized'
                elif model_type == 'finetuned':
                    edge_type = 'finetuned'
                elif model_type == 'merged':
                    edge_type = 'merged'
                elif model_type == 'adapter':
                    edge_type = 'adapter'
                else:
                    edge_type = 'derived'
                
                model_relationships.append({
                    'source': base_model,
                    'target': model_id,
                    'edge_type': edge_type,
                    'target_type': model_type
                })
    
    # Create mappings for known models
    model_types = {}
    model_archs = {}
    for metadata in metadata_list:
        model_types[metadata['id']] = metadata['model_type']
        model_archs[metadata['id']] = metadata.get('model_arch', 'unknown')
    
    # Add nodes
    for model_id in all_models:
        # Determine node type and architecture
        if model_id in model_types:
            node_type = model_types[model_id]
            node_arch = model_archs[model_id]
        else:
            # This is a base model not in our dataset, assume it's base
            node_type = 'base'
            node_arch = 'unknown'
        
        # Only add isolated nodes if requested
        has_relationships = any(
            rel['source'] == model_id or rel['target'] == model_id 
            for rel in model_relationships
        )
        
        if has_relationships or include_isolated:
            G.add_node(model_id, 
                      model_type=node_type,
                      model_arch=node_arch,
                      model_name=model_id.split('/')[-1],  # Short name
                      full_name=model_id)
    
    # Add edges
    for rel in model_relationships:
        if G.has_node(rel['source']) and G.has_node(rel['target']):
            G.add_edge(rel['source'], rel['target'],
                      edge_type=rel['edge_type'],
                      relationship=rel['edge_type'])
    
    return G

def visualize_model_network(G: nx.DiGraph, 
                          output_file: str = "model_data/model_network.png",
                          figsize: tuple = (20, 16),
                          node_size_factor: int = 200,
                          show_labels: bool = False,
                          layout: str = "force_atlas") -> None:
    """
    Visualize the model network graph with architecture-based communities.
    
    Args:
        G: NetworkX directed graph
        output_file: Path to save the visualization
        figsize: Figure size (width, height)
        node_size_factor: Base size for nodes
        show_labels: Whether to show node labels (disabled for large graphs)
        layout: Layout algorithm ('force_atlas', 'spring_multi', 'fruchterman', 'spiral')
    """
    if len(G.nodes()) == 0:
        print("⚠️ No nodes in graph to visualize")
        return
    
    # Create figure with dark background for better visibility
    plt.figure(figsize=figsize, facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Get architecture communities
    arch_communities = {}
    for node in G.nodes():
        arch = G.nodes[node].get('model_arch', 'unknown')
        if arch not in arch_communities:
            arch_communities[arch] = []
        arch_communities[arch].append(node)
    
    print(f"🏗️ Found {len(arch_communities)} architecture communities: {list(arch_communities.keys())}")
    
    # Choose optimized layout for large graphs
    n_nodes = G.number_of_nodes()
    
    if layout == "force_atlas" or n_nodes > 100:
        # Force-directed layout optimized for large graphs
        pos = nx.spring_layout(G, k=3/np.sqrt(n_nodes), iterations=30, seed=42)
    elif layout == "spring_multi":
        # Multi-level spring layout
        pos = nx.spring_layout(G, k=2/np.sqrt(n_nodes), iterations=50, seed=42)
    elif layout == "fruchterman":
        # Fruchterman-Reingold layout
        pos = nx.spring_layout(G, k=1/np.sqrt(n_nodes), iterations=100, seed=42)
    elif layout == "spiral":
        # Spiral layout for very large graphs
        pos = nx.spiral_layout(G)
    elif layout == "kamada_kawai":
        # Kamada-Kawai layout
        pos = nx.kamada_kawai_layout(G)
    else:
        # Default spring layout with optimized parameters
        pos = nx.spring_layout(G, k=1.5/np.sqrt(n_nodes), iterations=50, seed=42)
    

    
    arch_list = list(arch_communities.keys())
    colors = cm.Set3(np.linspace(0, 1, len(arch_list)))
    arch_colors = dict(zip(arch_list, colors))
    
    # Define model type markers
    type_markers = {
        'base': 'o',        # Circle
        'finetuned': 's',   # Square
        'quantized': '^',   # Triangle
        'merged': 'D',      # Diamond
        'adapter': 'v',     # Triangle down
        'unknown': 'o'      # Circle
    }
    
    # Draw nodes by architecture community
    for arch, nodes in arch_communities.items():
        arch_color = arch_colors[arch]
        
        # Separate nodes by type within each architecture
        for model_type in type_markers.keys():
            type_nodes = [node for node in nodes if G.nodes[node].get('model_type', 'unknown') == model_type]
            if not type_nodes:
                continue
                
            # Get positions and sizes for this group
            node_pos = {node: pos[node] for node in type_nodes}
            
            # Size based on degree (smaller for large graphs)
            node_sizes = []
            for node in type_nodes:
                in_degree = G.in_degree(node)
                out_degree = G.out_degree(node)
                size = max(node_size_factor, node_size_factor + (in_degree * 50) + (out_degree * 25))
                node_sizes.append(size)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, node_pos,
                                  nodelist=type_nodes,
                                  node_color=[arch_color] * len(type_nodes),
                                  node_size=node_sizes,
                                  node_shape=type_markers[model_type],
                                  alpha=0.8,
                                  edgecolors='white',
                                  linewidths=0.5)
    
    # Draw edges with reduced visibility for large graphs
    edge_alpha = max(0.1, min(0.4, 100/n_nodes))  # Adaptive alpha based on graph size
    edge_width = max(0.3, min(1.0, 50/n_nodes))   # Adaptive width based on graph size
    
    # Draw edges by type
    edge_type_colors = {
        'finetuned': '#e74c3c',   # Red
        'quantized': '#f39c12',   # Orange
        'merged': '#9b59b6',      # Purple
        'adapter': '#2ecc71',     # Green
        'derived': '#34495e'      # Dark gray
    }
    
    for edge_type, color in edge_type_colors.items():
        type_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type', 'derived') == edge_type]
        if type_edges:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=type_edges,
                                  edge_color=color,
                                  width=edge_width,
                                  alpha=edge_alpha,
                                  arrows=True,
                                  arrowsize=max(5, min(15, 200/n_nodes)),
                                  arrowstyle='->')
    
    # Create comprehensive legend
    legend_elements = []
    
    # Architecture communities legend
    for arch, color in arch_colors.items():
        count = len(arch_communities[arch])
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        label=f'{arch} ({count} models)'))
    
    # Add separator
    legend_elements.append(plt.Line2D([0], [0], color='none', label=''))
    
    # Model type markers legend
    for model_type, marker in type_markers.items():
        count = sum(1 for node in G.nodes() if G.nodes[node].get('model_type', 'unknown') == model_type)
        if count > 0:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='gray', markersize=8, 
                                            label=f'{model_type.title()} ({count})'))
    
    # Create legend with smaller font for large graphs
    font_size = max(6, min(10, 100/len(legend_elements)))
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              fontsize=font_size, facecolor='white', edgecolor='white', framealpha=0.9)
    
    # Title and styling
    plt.title("HuggingFace Model Architecture Communities", 
             fontsize=max(12, min(20, 200/np.sqrt(n_nodes))), 
             fontweight='bold', color='white', pad=20)
    
    # Add statistics text
    stats_text = f"Networks: {G.number_of_nodes()} models, {G.number_of_edges()} relationships\n"
    stats_text += f"Architectures: {len(arch_communities)} communities"
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the plot with higher DPI for large graphs
    dpi = 300 if n_nodes < 500 else 200
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='black')
    print(f"📊 Network visualization saved: {output_file}")
    print(f"🎨 Visualization: {n_nodes} nodes, {len(arch_communities)} architecture communities")
    
    # Don't show plot for very large graphs (can be slow)
    if n_nodes < 200:
        plt.show()
    else:
        print("⚠️ Graph too large for display, saved to file only")
        plt.close()

def export_network_csv(G: nx.DiGraph, output_dir: str = "network_data") -> None:
    """
    Export network data in CSV format compatible with external analysis tools.
    
    Args:
        G: NetworkX directed graph
        output_dir: Directory to save CSV files
    """
    import csv
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare edges data
    edges = []
    for source, target, data in G.edges(data=True):
        edges.append({
            'source': source,
            'target': target,
            'edge': data.get('edge_type', 'unknown')
        })
    
    # Prepare metadata
    metadata = []
    for node in G.nodes():
        # Extract owner from model ID (format: owner/model_name)
        owner = node.split('/')[0] if '/' in node else 'unknown'
        
        metadata.append({
            'id': node,
            'type': G.nodes[node].get('model_type', 'unknown'),
            'owner': owner,
            'architecture': G.nodes[node].get('model_arch', 'unknown')
        })
    
    # Save edges to CSV
    edges_file = os.path.join(output_dir, 'edges.csv')
    print(f"\nSaving {len(edges)} edges to {edges_file}...")
    with open(edges_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['source', 'target', 'edge'])
        writer.writeheader()
        writer.writerows(edges)
    print(f"✅ Edges saved to {edges_file}")
    
    # Save metadata to CSV
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    print(f"\nSaving {len(metadata)} node metadata to {metadata_file}...")
    with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'type', 'owner', 'architecture'])
        writer.writeheader()
        writer.writerows(metadata)
    print(f"✅ Metadata saved to {metadata_file}")

def analyze_network_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze network metrics and statistics.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dict with network analysis metrics
    """
    if len(G.nodes()) == 0:
        return {"error": "Empty graph"}
    
    metrics = {
        'basic_stats': {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G)
        },
        'node_metrics': {},
        'centrality': {},
        'model_types': {},
        'relationship_types': {}
    }
    
    # Node degree analysis
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    metrics['node_metrics'] = {
        'most_influential': max(out_degrees.items(), key=lambda x: x[1]) if out_degrees else None,
        'most_derived': max(in_degrees.items(), key=lambda x: x[1]) if in_degrees else None,
        'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
        'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0
    }
    
    # Centrality measures
    if G.number_of_nodes() > 1:
        try:
            metrics['centrality'] = {
                'pagerank': nx.pagerank(G),
                'in_degree_centrality': nx.in_degree_centrality(G),
                'out_degree_centrality': nx.out_degree_centrality(G)
            }
        except:
            metrics['centrality'] = {'error': 'Could not compute centrality measures'}
    
    # Model type distribution
    type_counts = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('model_type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    metrics['model_types'] = type_counts
    
    # Relationship type distribution
    rel_counts = {}
    for edge in G.edges():
        edge_type = G.edges[edge].get('edge_type', 'unknown')
        rel_counts[edge_type] = rel_counts.get(edge_type, 0) + 1
    metrics['relationship_types'] = rel_counts
    
    return metrics

def print_network_summary(G: nx.DiGraph, metrics: Dict[str, Any] = None) -> None:
    """
    Print a summary of the network graph.
    
    Args:
        G: NetworkX directed graph
        metrics: Pre-computed metrics (optional)
    """
    if metrics is None:
        metrics = analyze_network_metrics(G)
    
    if 'error' in metrics:
        print(f"❌ {metrics['error']}")
        return
    
    print(f"\n🔗 MODEL NETWORK ANALYSIS")
    print("=" * 50)
    
    # Basic stats
    basic = metrics['basic_stats']
    print(f"📊 Nodes: {basic['total_nodes']}")
    print(f"🔗 Edges: {basic['total_edges']}")
    print(f"📈 Density: {basic['density']:.3f}")
    print(f"🔄 Connected: {basic['is_connected']}")
    
    # Model types
    if metrics['model_types']:
        print(f"\n🤖 Model Types:")
        for model_type, count in metrics['model_types'].items():
            percentage = (count / basic['total_nodes']) * 100
            print(f"  {model_type.title()}: {count} ({percentage:.1f}%)")
    
    # Relationship types
    if metrics['relationship_types']:
        print(f"\n🔗 Relationship Types:")
        for rel_type, count in metrics['relationship_types'].items():
            percentage = (count / basic['total_edges']) * 100 if basic['total_edges'] > 0 else 0
            print(f"  {rel_type.title()}: {count} ({percentage:.1f}%)")
    
    # Key nodes
    node_metrics = metrics['node_metrics']
    if node_metrics['most_influential']:
        model, count = node_metrics['most_influential']
        print(f"\n🌟 Most Influential: {model} ({count} derivatives)")
    
    if node_metrics['most_derived']:
        model, count = node_metrics['most_derived']
        if count > 0:
            print(f"🎯 Most Derived: {model} (from {count} base models)")
    
    # Top PageRank nodes (if available)
    if 'pagerank' in metrics.get('centrality', {}):
        pagerank = metrics['centrality']['pagerank']
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n📈 Top PageRank Nodes:")
        for model, score in top_pagerank:
            print(f"  {model}: {score:.3f}")

# ============================================================================
# MODEL TYPE DETECTION FUNCTIONS
# ============================================================================

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
    
    # Regex patterns for different model types
    patterns = {
        'quantized': r'base_model:quantized:(.+)',
        'finetune': r'base_model:finetune:(.+)',
        'merge': r'base_model:merge:(.+)',
        'adapter': r'base_model:adapter:(.+)',
        'generic_base': r'base_model:([^:]+)$'  # base_model:model_name without type
    }
    
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
            for type_name, pattern in patterns.items():
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

# ============================================================================
# MODEL ANALYSIS FUNCTIONS
# ============================================================================

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
    

    model_arch = "unknown"
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
    
    return model_metadata

def analyze_single_model(model_name: str, hf_api: HfApi, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a single HuggingFace model.
    
    Args:
        model_name: Name of the model to analyze
        hf_api: HuggingFace API instance
        verbose: Whether to print detailed information
        
    Returns:
        dict: Model analysis results
    """
    if verbose:
        print(f"\n🎯 ANALYZING MODEL: {model_name}")
        print(f"Fetching detailed metadata...\n")
    
    try:
        model = model_info(model_name, files_metadata=True)
        metadata = extract_model_metadata(model, verbose=verbose)
        
        if verbose:
            print(f"✅ Successfully analyzed: {model_name}")
            type_display = format_model_type_info({
                'type': metadata['model_type'],
                'base_models': metadata['base_models'],
                'type_details': metadata['type_details']
            })
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

def analyze_model_list(hf_api: HfApi, pipeline_tag: List[str] = ["text-generation"], 
                      sort: str = "downloads", limit: int = 10, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a list of HuggingFace models.
    
    Args:
        hf_api: HuggingFace API instance
        pipeline_tag: Pipeline tags to filter by
        sort: Sort criteria
        limit: Number of models to analyze
        verbose: Whether to print detailed information for each model
        
    Returns:
        dict: Analysis results with summary
    """
    print(f"\n📊 ANALYZING {limit} MODELS")
    print(f"Pipeline: {pipeline_tag}, Sort: {sort}")
    print("=" * 60)
    
    # Get models with full metadata
    models = hf_api.list_models(
        pipeline_tag=pipeline_tag,
        sort=sort,
        limit=limit,
        cardData=True,
        full=True,
        fetch_config=True
    )
    
    print(f"✅ Found {len(list(models))} models")
    
    # Re-fetch models since list() consumed the iterator
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
    
    for i, model in enumerate(models):
        print(f"\n📋 Analyzing model {i+1}/{limit}: {model.id}")
        
        metadata = extract_model_metadata(model, verbose=verbose)
        model_metadata.append(metadata)
        
        # Count model types
        model_type = metadata['model_type']
        if model_type in type_counts:
            type_counts[model_type] += 1
        
        # Show brief summary unless verbose
        if not verbose:
            type_display = format_model_type_info({
                'type': metadata['model_type'],
                'base_models': metadata['base_models'],
                'type_details': metadata['type_details']
            })
            print(f"  {type_display}")
            print(f"  📊 {metadata['downloads']:,} downloads, {metadata['likes']} likes")
    
    # Create summary
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
    
    return {
        'success': True,
        'total_models': len(model_metadata),
        'metadata': model_metadata,
        'type_distribution': type_counts,
        'dataframe': df
    }

def save_analysis_results(results: Dict[str, Any], output_prefix: str = "hf_analysis") -> None:
    """
    Save analysis results to files.
    
    Args:
        results: Analysis results from analyze_single_model or analyze_model_list
        output_prefix: Prefix for output files
    """
    if not results['success']:
        print(f"❌ Cannot save results - analysis failed")
        return
    
    # For single model results
    if 'metadata' in results and not isinstance(results['metadata'], list):
        metadata_list = [results['metadata']]
        filename_suffix = f"_{results['model_name'].replace('/', '_')}"
    else:
        metadata_list = results['metadata']
        filename_suffix = f"_{results['total_models']}_models"
    
    # Save to CSV
    df = pd.DataFrame(metadata_list)
    csv_file = f"model_data/{output_prefix}{filename_suffix}.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Saved CSV: {csv_file}")
    
    # Save to JSON
    json_file = f"model_data/{output_prefix}{filename_suffix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 Saved JSON: {json_file}")
