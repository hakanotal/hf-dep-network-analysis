"""
Architecture propagation functionality for model dependency networks.
"""

import networkx as nx
from typing import List, Optional, Dict, Set
from collections import Counter

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
        
        # Run multiple iterations until no more changes
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            unknown_arch_nodes = self._find_unknown_arch_nodes()
            
            if not unknown_arch_nodes:
                if self.verbose:
                    print(f"✅ All nodes have known architectures!")
                break
            
            if self.verbose:
                print(f"\n🔄 Iteration {iteration}: {len(unknown_arch_nodes)} nodes with unknown arch")
            
            changes_made = self._process_unknown_nodes_smart(unknown_arch_nodes)
            
            if not changes_made:
                if self.verbose:
                    print(f"⏹️ No more changes possible, stopping iterations")
                break
        
        self._print_completion_stats()
    
    def _find_unknown_arch_nodes(self) -> List[str]:
        """Find all nodes with unknown architecture."""
        return [node for node in self.graph.nodes() 
                if self.graph.nodes[node].get('model_arch', UNKNOWN) == UNKNOWN]
    
    def _process_unknown_nodes_smart(self, unknown_nodes: List[str]) -> bool:
        """Process nodes with unknown architecture using smart prioritization."""
        changes_made = False
        
        # Score nodes by how likely they are to be successfully determined
        node_scores = self._score_nodes_by_determinability(unknown_nodes)
        
        # Sort by score (higher score = more likely to be determined)
        sorted_nodes = sorted(unknown_nodes, key=lambda x: node_scores.get(x, 0), reverse=True)
        
        for node in sorted_nodes:
            if self.graph.nodes[node].get('model_arch', UNKNOWN) != UNKNOWN:
                continue  # Already determined in this iteration
            
            if self.verbose:
                print(f"🔍 Processing node: {node} (score: {node_scores.get(node, 0)})")
            
            # Try multiple strategies to determine architecture
            found_arch = (
                self._find_arch_by_consensus(node) or
                self._find_arch_in_ancestors(node) or 
                self._find_arch_in_descendants(node) or
                self._find_arch_by_majority_vote(node)
            )
            
            if found_arch:
                self._set_node_architecture(node, found_arch)
                changes_made = True
                if self.verbose:
                    print(f"  ✅ Assigned architecture: '{found_arch}'")
            elif self.verbose:
                print(f"  ⚠️ No architecture found for {node}")
        
        return changes_made
    
    def _score_nodes_by_determinability(self, unknown_nodes: List[str]) -> Dict[str, int]:
        """Score nodes by how likely they are to have their architecture determined."""
        scores = {}
        
        for node in unknown_nodes:
            score = 0
            
            # Count neighbors with known architectures
            for neighbor in self.graph.predecessors(node):
                if self.graph.nodes[neighbor].get('model_arch', UNKNOWN) != UNKNOWN:
                    score += 2  # Ancestors are more important
            
            for neighbor in self.graph.successors(node):
                if self.graph.nodes[neighbor].get('model_arch', UNKNOWN) != UNKNOWN:
                    score += 1  # Descendants are less important
            
            scores[node] = score
        
        return scores
    
    def _find_arch_by_consensus(self, node: str) -> Optional[str]:
        """Find architecture by looking at direct neighbors and finding consensus."""
        architectures = []
        
        # Collect architectures from direct predecessors and successors
        for neighbor in list(self.graph.predecessors(node)) + list(self.graph.successors(node)):
            arch = self.graph.nodes[neighbor].get('model_arch', UNKNOWN)
            if arch != UNKNOWN:
                architectures.append(arch)
        
        if not architectures:
            return None
        
        # Count occurrences
        arch_counts = Counter(architectures)
        most_common_arch, count = arch_counts.most_common(1)[0]
        
        # Require at least 2 neighbors with the same architecture for consensus
        if count >= 2:
            return most_common_arch
        
        # If only one neighbor, still use it if it's a predecessor (more reliable)
        if len(architectures) == 1:
            for neighbor in self.graph.predecessors(node):
                arch = self.graph.nodes[neighbor].get('model_arch', UNKNOWN)
                if arch == most_common_arch:
                    return most_common_arch
        
        return None
    
    def _find_arch_by_majority_vote(self, node: str) -> Optional[str]:
        """Find architecture by majority vote among all connected nodes."""
        architectures = []
        
        # Collect architectures from all nodes within 2 hops
        for neighbor in nx.single_source_shortest_path_length(self.graph.to_undirected(), node, cutoff=2):
            if neighbor != node:
                arch = self.graph.nodes[neighbor].get('model_arch', UNKNOWN)
                if arch != UNKNOWN:
                    architectures.append(arch)
        
        if len(architectures) < 3:  # Need at least 3 for majority vote
            return None
        
        arch_counts = Counter(architectures)
        most_common_arch, count = arch_counts.most_common(1)[0]
        
        # Require majority (>50%)
        if count > len(architectures) / 2:
            return most_common_arch
        
        return None
    
    def _find_arch_in_ancestors(self, node: str, visited: Set[str] = None) -> Optional[str]:
        """Recursively search ancestors for known architecture."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return None
        
        visited.add(node)
        
        # Check direct predecessors first
        for parent in self.graph.predecessors(node):
            parent_arch = self.graph.nodes[parent].get('model_arch', UNKNOWN)
            if parent_arch != UNKNOWN:
                return parent_arch
        
        # Then check ancestors recursively
        for parent in self.graph.predecessors(node):
            parent_arch = self._find_arch_in_ancestors(parent, visited.copy())
            if parent_arch and parent_arch != UNKNOWN:
                return parent_arch
        
        return None
    
    def _find_arch_in_descendants(self, node: str, visited: Set[str] = None) -> Optional[str]:
        """Recursively search descendants for consistent architecture."""
        if visited is None:
            visited = set()
        
        if node in visited:
            return None
        
        visited.add(node)
        
        # Check direct successors first
        child_architectures = set()
        for child in self.graph.successors(node):
            child_arch = self.graph.nodes[child].get('model_arch', UNKNOWN)
            if child_arch != UNKNOWN:
                child_architectures.add(child_arch)
        
        # If all direct children have the same architecture, use it
        if len(child_architectures) == 1:
            return child_architectures.pop()
        
        # Otherwise, check descendants recursively
        for child in self.graph.successors(node):
            child_arch = self._find_arch_in_descendants(child, visited.copy())
            if child_arch and child_arch != UNKNOWN:
                child_architectures.add(child_arch)
        
        # Only return if all descendants have the same architecture
        return child_architectures.pop() if len(child_architectures) == 1 else None
    
    def _set_node_architecture(self, node: str, architecture: str) -> None:
        """Set architecture for a single node."""
        self.graph.nodes[node]['model_arch'] = architecture
        if self.verbose:
            print(f"  🔄 Set {node}: arch='{architecture}'")
    
    def _print_completion_stats(self) -> None:
        """Print completion statistics."""
        if not self.verbose:
            return
        
        final_unknown_arch = sum(1 for node in self.graph.nodes() 
                               if self.graph.nodes[node].get('model_arch', UNKNOWN) == UNKNOWN)
        
        total_nodes = self.graph.number_of_nodes()
        determined_nodes = total_nodes - final_unknown_arch
        
        print(f"\n✅ SMART ARCHITECTURE PROPAGATION COMPLETE")
        print(f"📊 Total nodes: {total_nodes}")
        print(f"✅ Determined architectures: {determined_nodes}")
        print(f"❓ Remaining unknown: {final_unknown_arch}")
        print(f"📈 Success rate: {(determined_nodes/total_nodes)*100:.1f}%")
        
        if final_unknown_arch > 0:
            print(f"💡 Remaining nodes likely have no connected components with architecture info")


def propagate_metadata(G: nx.DiGraph, verbose: bool = False) -> None:
    """Propagate model_arch metadata through the graph."""
    propagator = ArchitecturePropagator(G, verbose)
    propagator.propagate() 