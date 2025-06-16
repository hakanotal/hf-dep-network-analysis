"""
Architecture propagation functionality for model dependency networks.
"""

import networkx as nx
from typing import List, Optional

from .constants import UNKNOWN


class ArchitecturePropagator:
    """Handles propagation of architecture metadata through the network graph."""
    
    def __init__(self, graph: nx.DiGraph, verbose: bool = False):
        self.graph = graph
        self.verbose = verbose
    
    def propagate(self) -> None:
        """Main method to propagate architecture through the graph."""
        if self.verbose:
            print(f"\n🔄 PROPAGATING ARCHITECTURE THROUGH GRAPH")
            print(f"📊 Total nodes: {self.graph.number_of_nodes()}")
        
        unknown_arch_nodes = self._find_unknown_arch_nodes()
        
        if self.verbose:
            print(f"❓ Nodes with unknown arch: {len(unknown_arch_nodes)}")
        
        self._process_unknown_nodes(unknown_arch_nodes)
        self._print_completion_stats()
    
    def _find_unknown_arch_nodes(self) -> List[str]:
        """Find all nodes with unknown architecture."""
        return [node for node in self.graph.nodes() 
                if self.graph.nodes[node].get('model_arch', UNKNOWN) == UNKNOWN]
    
    def _process_unknown_nodes(self, unknown_nodes: List[str]) -> None:
        """Process nodes with unknown architecture."""
        processed_nodes = set()
        
        for node in unknown_nodes:
            if node in processed_nodes:
                continue
            
            if self.verbose:
                print(f"\n🔍 Processing node: {node}")
            
            # Try ancestors first, then descendants
            found_arch = self._find_arch_in_ancestors(node) or self._find_arch_in_descendants(node)
            
            if found_arch:
                self._propagate_architecture(node, found_arch)
                processed_nodes.add(node)
            elif self.verbose:
                print(f"  ⚠️ No consistent architecture found for {node}")
    
    def _find_arch_in_ancestors(self, node: str, visited: set = None) -> Optional[str]:
        """Recursively search ancestors for known architecture."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return None
        
        visited.add(node)
        current_arch = self.graph.nodes[node].get('model_arch', UNKNOWN)
        
        if current_arch != UNKNOWN:
            return current_arch
        
        for parent in self.graph.predecessors(node):
            parent_arch = self._find_arch_in_ancestors(parent, visited.copy())
            if parent_arch and parent_arch != UNKNOWN:
                return parent_arch
        
        return None
    
    def _find_arch_in_descendants(self, node: str, visited: set = None) -> Optional[str]:
        """Recursively search descendants for consistent architecture."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return None
        
        visited.add(node)
        current_arch = self.graph.nodes[node].get('model_arch', UNKNOWN)
        
        if current_arch != UNKNOWN:
            return current_arch
        
        child_architectures = set()
        for child in self.graph.successors(node):
            child_arch = self._find_arch_in_descendants(child, visited.copy())
            if child_arch and child_arch != UNKNOWN:
                child_architectures.add(child_arch)
        
        # Only return if all children have the same architecture
        return child_architectures.pop() if len(child_architectures) == 1 else None
    
    def _propagate_architecture(self, start_node: str, architecture: str) -> None:
        """Propagate architecture to node and its connected components."""
        if self.verbose:
            print(f"  ✅ Found architecture: '{architecture}', propagating...")
        
        # Update the starting node
        self.graph.nodes[start_node]['model_arch'] = architecture
        
        # Propagate to descendants and ancestors
        self._propagate_to_descendants(start_node, architecture)
        self._propagate_to_ancestors(start_node, architecture)
    
    def _propagate_to_descendants(self, node: str, architecture: str, visited: set = None) -> None:
        """Propagate architecture to descendants."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return
        
        visited.add(node)
        
        for child in self.graph.successors(node):
            if child not in visited:
                child_arch = self.graph.nodes[child].get('model_arch', UNKNOWN)
                if child_arch == UNKNOWN:
                    self.graph.nodes[child]['model_arch'] = architecture
                    if self.verbose:
                        print(f"  🔄 Updated {child}: arch='{architecture}'")
                
                self._propagate_to_descendants(child, architecture, visited.copy())
    
    def _propagate_to_ancestors(self, node: str, architecture: str, visited: set = None) -> None:
        """Propagate architecture to ancestors."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return
        
        visited.add(node)
        
        for parent in self.graph.predecessors(node):
            if parent not in visited:
                parent_arch = self.graph.nodes[parent].get('model_arch', UNKNOWN)
                if parent_arch == UNKNOWN:
                    self.graph.nodes[parent]['model_arch'] = architecture
                    if self.verbose:
                        print(f"  🔄 Updated {parent}: arch='{architecture}' (from descendants)")
                
                self._propagate_to_ancestors(parent, architecture, visited.copy())
    
    def _print_completion_stats(self) -> None:
        """Print completion statistics."""
        if not self.verbose:
            return
        
        final_unknown_arch = sum(1 for node in self.graph.nodes() 
                               if self.graph.nodes[node].get('model_arch', UNKNOWN) == UNKNOWN)
        
        print(f"\n✅ BIDIRECTIONAL ARCHITECTURE PROPAGATION COMPLETE")
        print(f"📊 Remaining unknown archs: {final_unknown_arch}")
        
        if final_unknown_arch > 0:
            print(f"💡 These nodes have no connected ancestors or descendants with consistent architecture info")


def propagate_metadata(G: nx.DiGraph, verbose: bool = False) -> None:
    """Propagate model_arch metadata through the graph."""
    propagator = ArchitecturePropagator(G, verbose)
    propagator.propagate() 