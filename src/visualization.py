"""
Network visualization using pyvis for interactive HTML visualizations with community detection.
"""

import os
import pandas as pd
from pyvis.network import Network
import math

from .utils import calculate_node_size
from .constants import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT, 
    NETWORK_BGCOLOR,
    FONT_COLOR,
    ARCH_COLORS,
    EDGE_COLORS
)


def create_community_mapping(metadata_df: pd.DataFrame, min_community_size: int = 25) -> dict:
    """
    Create a mapping of architectures to communities, grouping small ones into 'other'.
    
    Args:
        metadata_df: DataFrame with model metadata
        min_community_size: Minimum size for a community to remain separate
        
    Returns:
        Dictionary mapping original architecture to community name
    """
    arch_counts = metadata_df['model_arch'].value_counts()
    
    # Create community mapping
    community_mapping = {}
    for arch, count in arch_counts.items():
        if count >= min_community_size:
            community_mapping[arch] = arch
        else:
            community_mapping[arch] = 'other'
    
    return community_mapping


def get_community_colors(communities: list) -> dict:
    """
    Get colors for each community, ensuring 'other' gets a distinct color.
    
    Args:
        communities: List of unique community names
        
    Returns:
        Dictionary mapping community names to colors
    """
    # Sort communities to ensure consistent coloring, but put 'other' last
    sorted_communities = sorted([c for c in communities if c != 'other'])
    if 'other' in communities:
        sorted_communities.append('other')
    
    community_colors = {}
    for i, community in enumerate(sorted_communities):
        if community == 'other':
            # Use gray for 'other' community
            community_colors[community] = '#95a5a6'
        else:
            community_colors[community] = ARCH_COLORS[i % len(ARCH_COLORS)]
    
    return community_colors


def calculate_community_positions(metadata_df: pd.DataFrame, community_mapping: dict) -> dict:
    """
    Calculate positions for nodes based on their communities using a circular layout.
    
    Args:
        metadata_df: DataFrame with model metadata
        community_mapping: Mapping of architectures to communities
        
    Returns:
        Dictionary with node positions
    """
    # Add community column
    metadata_df['community'] = metadata_df['model_arch'].map(community_mapping)
    
    # Get unique communities
    communities = metadata_df['community'].unique()
    community_positions = {}
    
    # Calculate community centers in a circular layout
    num_communities = len(communities)
    community_centers = {}
    
    for i, community in enumerate(communities):
        angle = 2 * math.pi * i / num_communities
        radius = 500  # Distance from center
        center_x = radius * math.cos(angle)
        center_y = radius * math.sin(angle)
        community_centers[community] = (center_x, center_y)
    
    # Position nodes within each community
    node_positions = {}
    for community in communities:
        community_nodes = metadata_df[metadata_df['community'] == community]
        center_x, center_y = community_centers[community]
        
        # Arrange nodes in a circle within the community
        num_nodes = len(community_nodes)
        community_radius = min(150, 50 + num_nodes * 2)  # Adaptive radius based on node count
        
        for j, (_, node) in enumerate(community_nodes.iterrows()):
            if num_nodes == 1:
                # Single node at center
                node_x, node_y = center_x, center_y
            else:
                # Multiple nodes in circle
                angle = 2 * math.pi * j / num_nodes
                node_x = center_x + community_radius * math.cos(angle)
                node_y = center_y + community_radius * math.sin(angle)
            
            node_positions[node['id']] = {'x': node_x, 'y': node_y}
    
    return node_positions


def visualize_model_network(output_file: str = "model_network.html",
                          network_data_dir: str = "network_data",
                          width: str = DEFAULT_WIDTH,
                          height: str = DEFAULT_HEIGHT,
                          min_community_size: int = 25) -> None:
    """
    Create an interactive network visualization using pyvis with community detection.
    
    Args:
        output_file: Path to save the HTML visualization
        network_data_dir: Directory containing edges.csv and metadata.csv
        width: Width of the visualization
        height: Height of the visualization
        min_community_size: Minimum size for architecture communities
    """
    
    # Check if network data files exist
    edges_file = os.path.join(network_data_dir, 'edges.csv')
    metadata_file = os.path.join(network_data_dir, 'metadata.csv')
    output_file = os.path.join(network_data_dir, output_file)
    
    if not os.path.exists(edges_file):
        print(f"❌ Edges file not found: {edges_file}")
        return
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    print(f"📊 Reading network data from {network_data_dir}...")
    
    # Read data
    edges_df = pd.read_csv(edges_file)
    metadata_df = pd.read_csv(metadata_file)
    
    print(f"✅ Loaded {len(edges_df)} edges and {len(metadata_df)} nodes")
    
    # Create community mapping
    community_mapping = create_community_mapping(metadata_df, min_community_size)
    metadata_df['community'] = metadata_df['model_arch'].map(community_mapping)
    
    # Print community statistics
    community_stats = metadata_df['community'].value_counts()
    print(f"🏘️ Communities created:")
    for community, count in community_stats.items():
        if community == 'other':
            original_archs = metadata_df[metadata_df['community'] == 'other']['model_arch'].value_counts()
            print(f"  {community}: {count} nodes from {len(original_archs)} architectures")
        else:
            print(f"  {community}: {count} nodes")
    
    # Get community colors
    communities = metadata_df['community'].unique()
    community_colors = get_community_colors(communities)
    
    # Create pyvis network
    net = Network(width=width, height=height, directed=True, bgcolor=NETWORK_BGCOLOR, font_color=FONT_COLOR)
    
    # Configure physics for community-based layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200},
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springConstant": 0.08,
          "springLength": 100,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "arrows": {
          "to": {"enabled": true, "scaleFactor": 1.2}
        },
        "color": {"inherit": false},
        "smooth": {"enabled": true, "type": "continuous"}
      },
      "nodes": {
        "font": {"size": 12, "color": "white"},
        "borderWidth": 2,
        "shadow": {"enabled": true}
      }
    }
    """)
    
    print(f"🎨 Using {len(communities)} community colors")
    
    # Calculate initial positions for better community layout
    node_positions = calculate_community_positions(metadata_df, community_mapping)
    
    # Add nodes
    for _, row in metadata_df.iterrows():
        node_id = row['id']
        downloads = row.get('downloads', 0)
        arch = row.get('model_arch', 'unknown')
        community = row.get('community', 'other')
        model_type = row.get('model_type', 'unknown')
        author = row.get('author', 'unknown')
        
        # Scale node size based on downloads (min 10, max 50)
        size = calculate_node_size(downloads)
        
        # Get color for community
        color = community_colors.get(community, '#CCCCCC')
        
        # Create hover tooltip
        title = f"""
        <b>{node_id}</b><br>
        Author: {author}<br>
        Type: {model_type}<br>
        Architecture: {arch}<br>
        Community: {community}<br>
        Downloads: {downloads:,}
        """
        
        # Get position if available
        position = node_positions.get(node_id, {})
        
        # Add node with label as the model ID
        net.add_node(node_id, 
                    label=node_id,
                    size=size, 
                    color=color,
                    title=title,
                    borderWidth=2,
                    borderWidthSelected=4,
                    **position)  # Add x, y coordinates if available
    
    # Add edges (only if both source and target nodes exist)
    valid_node_ids = set(metadata_df['id'].tolist())
    valid_edges = []
    
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row.get('edge', 'unknown')
        
        # Only add edge if both nodes exist in metadata
        if source in valid_node_ids and target in valid_node_ids:
            # Get edge color
            color = EDGE_COLORS.get(edge_type, '#95a5a6')
            
            # Add edge with label
            net.add_edge(source, target, 
                        label=edge_type,
                        color=color,
                        width=2,
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.2}})
            valid_edges.append(row)
    
    print(f"✅ Added {len(valid_edges)} valid edges (skipped {len(edges_df) - len(valid_edges)} invalid ones)")
    
    # Create legend HTML
    legend_html = """
    <div id="legend" style="position: absolute; top: 10px; left: 10px; 
                           background: rgba(0,0,0,0.8); padding: 15px; 
                           border-radius: 8px; color: white; font-family: Arial;">
        <h3 style="margin-top: 0;">🏘️ Model Communities</h3>
        
        <h4>📊 Downloads</h4>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 10px;">●</span> Low downloads<br>
            <span style="font-size: 14px;">●</span> Medium downloads<br>
            <span style="font-size: 18px;">●</span> High downloads
        </div>
        
        <h4>🏘️ Communities (Architecture)</h4>
    """
    
    # Add community colors to legend
    for community, color in community_colors.items():
        if community == 'other':
            legend_html += f'<div><span style="color: {color}; font-size: 14px;">●</span> {community} (small architectures)</div>'
        else:
            legend_html += f'<div><span style="color: {color}; font-size: 14px;">●</span> {community}</div>'
    
    legend_html += """
        <h4>🔗 Relationship</h4>
    """
    
    # Add edge colors to legend
    for edge_type, color in EDGE_COLORS.items():
        if edge_type != 'unknown':  # Don't show unknown in legend
            legend_html += f'<div><span style="color: {color};">→</span> {edge_type.title()}</div>'
    
    legend_html += """
        </div>
    """
    
    # Save the network
    net.save_graph(output_file)
    
    # Add legend to the HTML file
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Insert legend before closing body tag
    content = content.replace('</body>', f'{legend_html}</body>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"🎨 Interactive community-based network visualization saved: {output_file}")
    print(f"📊 Network: {len(metadata_df)} nodes, {len(edges_df)} edges")
    print(f"🏘️ Communities: {len(communities)} groups")
    for community, count in community_stats.items():
        print(f"  - {community}: {count} nodes")
    print(f"🔗 Relationships: {len(edges_df['edge'].unique())} types")
    print(f"💡 Open {output_file} in your browser to view the interactive network") 