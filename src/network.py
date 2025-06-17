"""
Network graph building and operations for model dependency analysis.
"""

from typing import Dict, Any, List, Union
import networkx as nx

from .data_io import fetch_missing_models_from_graph
from .architecture import propagate_metadata
from .utils import determine_edge_type
from .constants import UNKNOWN


def build_model_network(analysis_results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                       include_isolated: bool = False,
                       propagate_metadata_flag: bool = True,
                       fetch_missing: bool = True) -> nx.DiGraph:
    """
    Build a directed network graph of model relationships.
    
    Args:
        analysis_results: Results from analyze_single_model or analyze_model_list,
                         or a list of model metadata dictionaries
        include_isolated: Whether to include models with no relationships as isolated nodes
        propagate_metadata_flag: Whether to propagate metadata through the graph
        fetch_missing: Whether to fetch missing models referenced in relationships
        
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
                edge_type = determine_edge_type(model_type)
                
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
        model_archs[metadata['id']] = metadata.get('model_arch', UNKNOWN)
    
    # Add nodes
    for model_id in all_models:
        # Determine node type and architecture
        if model_id in model_types:
            node_type = model_types[model_id]
            node_arch = model_archs[model_id]
        else:
            # This is a base model not in our dataset, assume it's base
            node_type = 'base'
            node_arch = UNKNOWN
        
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
    
    # FETCH MISSING MODELS HERE
    if fetch_missing:
        print(f"\n🔍 FETCHING MISSING MODELS")
        updated_metadata_list = fetch_missing_models_from_graph(G, metadata_list)
        
        # Update the analysis_results to include the new metadata
        if isinstance(analysis_results, dict) and 'metadata' in analysis_results:
            analysis_results['metadata'] = updated_metadata_list
        elif isinstance(analysis_results, list):
            analysis_results[:] = updated_metadata_list  # Update in place

        # Update the graph G with the new metadata
        return build_model_network(updated_metadata_list, include_isolated=False, propagate_metadata_flag=True, fetch_missing=False)
    
    if propagate_metadata_flag:
        print(f"\n🔄 PROPAGATING METADATA: {len(metadata_list)}")
        propagate_metadata(G, verbose=False)
    
    return G


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
        node_type = G.nodes[node].get('model_type', UNKNOWN)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    metrics['model_types'] = type_counts
    
    # Relationship type distribution
    rel_counts = {}
    for edge in G.edges():
        edge_type = G.edges[edge].get('edge_type', UNKNOWN)
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