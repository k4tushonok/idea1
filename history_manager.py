"""
Optimization history manager.

Stores all PromptNode objects indexed by generation, source, and
prompt text. Provides methods for history analysis, stagnation
detection, statistics, and JSON serialization.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import json
import os
from datetime import datetime
import numpy as np
from data_structures import PromptNode, OptimizationSource
from config import (
    MIN_IMPROVEMENT,
    GLOBAL_HISTORY_WINDOW,
    TOP_BEST_NODES,
    DEFAULT_STAGNATION_WINDOW,
)


class HistoryManager:
    """Storage and index for all optimization nodes.

    Maintains indexes by generation, source, and prompt text;
    provides fast lookup, successful-operation analysis, and serialization.
    """

    def __init__(self):
        self.nodes: Dict[str, PromptNode] = {}  # Primary storage: id -> PromptNode
        self.nodes_by_generation: Dict[int, List[str]] = defaultdict(
            list
        )  # Index by generation
        self.nodes_by_source: Dict[OptimizationSource, List[str]] = defaultdict(
            list
        )  # Index by source
        self.nodes_by_prompt_text: Dict[str, List[str]] = defaultdict(
            list
        )  # Index by exact prompt text
        self.root_ids: List[str] = []          # Root nodes (initial prompts)
        self.total_evaluations = 0             # Total number of evaluated nodes
        self.creation_time = datetime.now()    # History creation timestamp

    def add_node(self, node: PromptNode) -> str:
        """Add a node to history, updating all indexes and parent-child links."""
        # Add to primary storage
        self.nodes[node.id] = node

        # Update indexes
        self.nodes_by_generation[node.generation].append(node.id)
        self.nodes_by_source[node.source].append(node.id)
        self.nodes_by_prompt_text[node.prompt_text].append(node.id)

        # If the node has no parent, it is a root
        if node.parent_id is None:
            if node.id not in self.root_ids:
                self.root_ids.append(node.id)
        else:
            # Register link to parent
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].add_child(node.id)

        if node.is_evaluated:
            self.total_evaluations += 1

        return node.id

    def get_node(self, node_id: str) -> Optional[PromptNode]:
        """Look up a node by its ID."""
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, node: PromptNode):
        """Update an existing node. Used after evaluating a prompt."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        old_node = self.nodes[node_id]

        # Update node
        self.nodes[node_id] = node

        # If key fields changed, update indexes
        if old_node.generation != node.generation or old_node.source != node.source:
            if node_id in self.nodes_by_generation[old_node.generation]:
                self.nodes_by_generation[old_node.generation].remove(node_id)
            self.nodes_by_generation[node.generation].append(node_id)
            if node_id in self.nodes_by_source[old_node.source]:
                self.nodes_by_source[old_node.source].remove(node_id)
            self.nodes_by_source[node.source].append(node_id)
        if old_node.prompt_text != node.prompt_text:
            if node_id in self.nodes_by_prompt_text[old_node.prompt_text]:
                self.nodes_by_prompt_text[old_node.prompt_text].remove(node_id)
            self.nodes_by_prompt_text[node.prompt_text].append(node_id)
        # Adjust total_evaluations
        if not old_node.is_evaluated and node.is_evaluated:
            self.total_evaluations += 1

    def find_node_by_prompt_text(
        self, prompt_text: str, evaluated_only: bool = True
    ) -> Optional[PromptNode]:
        """Find a node by exact prompt text match."""
        for node_id in self.nodes_by_prompt_text.get(prompt_text, []):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if evaluated_only and not node.is_evaluated:
                continue
            return node
        return None

    def get_lineage(self, node_id: str) -> List[PromptNode]:
        """Return the full ancestry chain from the root down to the given node."""
        lineage = []
        current_id = node_id

        while current_id is not None:
            node = self.get_node(current_id)
            if node is None:
                break
            lineage.append(node)
            current_id = node.parent_id

        # Reverse so that root comes first
        return list(reversed(lineage))

    def get_nodes_by_generation(self, generation: int) -> List[PromptNode]:
        """Return all nodes from a specific generation."""
        node_ids = self.nodes_by_generation.get(generation, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_nodes_by_source(self, source: OptimizationSource) -> List[PromptNode]:
        """Return all nodes produced by a specific source."""
        node_ids = self.nodes_by_source.get(source, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_evaluated_nodes(self) -> List[PromptNode]:
        """Return all evaluated nodes."""
        return [node for node in self.nodes.values() if node.is_evaluated]

    def get_best_nodes(self, top_k: int) -> List[PromptNode]:
        """Return the top-k nodes sorted by selection score."""
        # Filter evaluated nodes
        candidates = self.get_evaluated_nodes()

        if not candidates:
            return []

        # Sort by score
        candidates.sort(key=lambda n: n.selection_score(), reverse=True)

        return candidates[:top_k]

    def analyze_successful_operations(self) -> Dict[str, int]:
        """Analyse which edit operations led to score improvements."""
        operation_counts = defaultdict(int)

        for node in self.get_evaluated_nodes():
            if node.parent_id is None:
                continue

            parent = self.get_node(node.parent_id)
            if parent is None or not parent.is_evaluated:
                continue

            # Check if there is an improvement
            improvement = node.selection_score() - parent.selection_score()

            if improvement >= MIN_IMPROVEMENT:
                # Count edits that led to improvement (by description)
                for op in node.operations:
                    key = op.description[:50] if op.description else "unknown"
                    operation_counts[key] += 1

        return dict(operation_counts)

    def get_stagnation_info(self) -> Dict[str, any]:
        """Detect stagnation by comparing best scores over the last few generations."""
        current_gen = (
            max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0
        )

        if current_gen < DEFAULT_STAGNATION_WINDOW:
            return {"is_stagnant": False, "best_score": 0.0}

        # Collect best scores from the last `window` generations
        recent_best_scores = []
        for gen in range(
            max(0, current_gen - DEFAULT_STAGNATION_WINDOW + 1), current_gen + 1
        ):
            gen_nodes = self.get_nodes_by_generation(gen)
            evaluated = [n for n in gen_nodes if n.is_evaluated]
            if evaluated:
                best_score = max(n.selection_score() for n in evaluated)
                recent_best_scores.append(best_score)

        if len(recent_best_scores) < 2:
            return {
                "is_stagnant": False,
                "best_score": recent_best_scores[0] if recent_best_scores else 0.0,
            }

        # Check whether there was meaningful progress
        max_score = max(recent_best_scores)
        min_score = min(recent_best_scores)
        improvement = max_score - min_score

        is_stagnant = improvement < MIN_IMPROVEMENT

        return {
            "is_stagnant": is_stagnant,
            "best_score": max_score,
            "improvement": improvement,
            "recent_scores": recent_best_scores,
        }

    def get_optimization_summary(self) -> Dict[str, any]:
        """Return an optimization summary: best nodes, successful operations, and source distribution."""
        # Get best nodes
        best_nodes = self.get_best_nodes(top_k=TOP_BEST_NODES)

        # Analyse successful operations
        successful_ops = self.analyze_successful_operations()

        # Recent nodes
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.timestamp, reverse=True)
        recent_nodes = all_nodes[:GLOBAL_HISTORY_WINDOW]

        # Source statistics
        source_stats = {
            source.value: len(self.get_nodes_by_source(source))
            for source in OptimizationSource
        }

        # Generation distribution
        generation_stats = {
            gen: len(nodes) for gen, nodes in self.nodes_by_generation.items()
        }

        return {
            "total_nodes": len(self.nodes),
            "total_evaluations": self.total_evaluations,
            "current_generation": (
                max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0
            ),
            "best_nodes": [
                {
                    "id": node.id,
                    "score": node.selection_score(),
                    "generation": node.generation,
                    "source": node.source.value,
                    "prompt_preview": node.prompt_text[:100] + "...",
                }
                for node in best_nodes
            ],
            "successful_operations": successful_ops,
            "recent_nodes_count": len(recent_nodes),
            "source_distribution": source_stats,
            "generation_distribution": generation_stats,
        }

    def save(self, filepath: str):
        """Persist the full history to a JSON file."""
        data = {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_ids": self.root_ids,
            "total_evaluations": self.total_evaluations,
            "creation_time": self.creation_time.isoformat(),
            "save_time": datetime.now().isoformat(),
        }

        # Create directory if it does not exist
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"History saved to {filepath} ({len(self.nodes)} nodes)")

    def get_statistics(self) -> Dict[str, any]:
        """Return general statistics about the history."""
        evaluated = self.get_evaluated_nodes()

        return {
            "total_nodes": len(self.nodes),
            "evaluated_nodes": len(evaluated),
            "root_nodes": len(self.root_ids),
            "generations": len(self.nodes_by_generation),
            "max_generation": (
                max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0
            ),
            "best_score": (
                max(n.selection_score() for n in evaluated) if evaluated else 0.0
            ),
            "avg_score": (
                np.mean([n.selection_score() for n in evaluated]) if evaluated else 0.0
            ),
            "total_evaluations": self.total_evaluations,
        }

    def __len__(self) -> int:
        """Return the total number of nodes in history."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"HistoryManager(nodes={stats['total_nodes']}, gen={stats['max_generation']}, best={stats['best_score']:.3f})"
