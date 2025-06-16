"""
Network visualization using pyvis for interactive HTML visualizations.
"""

import os
import pandas as pd
from pyvis.network import Network

from .utils import calculate_node_size
from .constants import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT, 
    NETWORK_BGCOLOR,
    FONT_COLOR,
    ARCH_COLORS,
    EDGE_COLORS
)


def visualize_model_network(output_file: str = "model_network.html",
                          network_data_dir: str = "network_data",
                          width: str = DEFAULT_WIDTH,
                          height: str = DEFAULT_HEIGHT) -> None:
    """
    Create an interactive network visualization using pyvis.
    
    Args:
        output_file: Path to save the HTML visualization
        network_data_dir: Directory containing edges.csv and metadata.csv
        width: Width of the visualization
        height: Height of the visualization
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
    
    # Create pyvis network
    net = Network(width=width, height=height, directed=True, bgcolor=NETWORK_BGCOLOR, font_color=FONT_COLOR)
    
    # Configure physics for better layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        }
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
    
    # Color palette for architectures
    architectures = metadata_df['model_arch'].unique()
    colors = ARCH_COLORS[:len(architectures)]
    
    # Create color mapping
    arch_colors = {}
    for i, arch in enumerate(architectures):
        arch_colors[arch] = colors[i % len(colors)]
    
    print(f"🎨 Using {len(architectures)} architecture colors: {list(architectures)}")
    
    # Add nodes
    for _, row in metadata_df.iterrows():
        node_id = row['id']
        downloads = row.get('downloads', 0)
        arch = row.get('model_arch', 'unknown')
        model_type = row.get('model_type', 'unknown')
        author = row.get('author', 'unknown')
        
        # Scale node size based on downloads (min 10, max 50)
        size = calculate_node_size(downloads)
        
        # Get color for architecture
        color = arch_colors.get(arch, '#CCCCCC')
        
        # Create hover tooltip
        title = f"""
        <b>{node_id}</b><br>
        Author: {author}<br>
        Type: {model_type}<br>
        Architecture: {arch}<br>
        Downloads: {downloads:,}
        """
        
        # Add node with label as the model ID
        net.add_node(node_id, 
                    label=node_id,
                    size=size, 
                    color=color,
                    title=title,
                    borderWidth=2,
                    borderWidthSelected=4)
    
    # Add edges
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row.get('edge', 'unknown')
        
        # Get edge color
        color = EDGE_COLORS.get(edge_type, '#95a5a6')
        
        # Add edge with label
        net.add_edge(source, target, 
                    label=edge_type,
                    color=color,
                    width=2,
                    arrows={'to': {'enabled': True, 'scaleFactor': 1.2}})
    
    # Create legend HTML
    legend_html = """
    <div id="legend" style="position: absolute; top: 10px; left: 10px; 
                           background: rgba(0,0,0,0.8); padding: 15px; 
                           border-radius: 8px; color: white; font-family: Arial;">
        <h3 style="margin-top: 0;">🔗 Model Network Legend</h3>
        
        <h4>📊 Downloads</h4>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 10px;">●</span> Low downloads<br>
            <span style="font-size: 14px;">●</span> Medium downloads<br>
            <span style="font-size: 18px;">●</span> High downloads
        </div>
        
        <h4>🎨 Architecture</h4>
    """
    
    # Add architecture colors to legend
    for arch, color in arch_colors.items():
        legend_html += f'<div><span style="color: {color}; font-size: 14px;">●</span> {arch}</div>'
    
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
    
    print(f"🎨 Interactive network visualization saved: {output_file}")
    print(f"📊 Network: {len(metadata_df)} nodes, {len(edges_df)} edges")
    print(f"🏗️ Architectures: {len(architectures)} types")
    print(f"🔗 Relationships: {len(edges_df['edge'].unique())} types")
    print(f"💡 Open {output_file} in your browser to view the interactive network") 