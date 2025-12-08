"""
RecencyBank: Temporal Heuristic for Link Prediction

A simple baseline that predicts users will edit the pages they most recently edited.
Based on the observation that Wikipedia editors exhibit strong temporal locality.

Author: Lorenzo Domenico Attolico
Course: Special Lectures in Mathematical Informatics II
Dataset: tgbl-wiki from Temporal Graph Benchmark
"""

import numpy as np
from collections import defaultdict


def _default_dict_factory():
    """Factory function for nested defaultdict (needed for pickle)."""
    return defaultdict(int)


class RecencyBank:
    """
    RecencyBank maintains interaction history with temporal decay scoring.
    
    For each (source, destination) pair, tracks:
    - Last interaction timestamp
    - Interaction count
    
    Scoring combines recency decay with frequency bonus.
    
    Complexity:
        - Training: O(|E|) time, O(|E|) space
        - Query: O(1) per (src, dst) pair
    
    Attributes:
        last_time (dict): Maps (src, dst) pairs to most recent timestamp
        counts (dict): Maps (src, dst) pairs to interaction count
    """
    
    def __init__(self):
        """Initialize empty RecencyBank."""
        # Use factory function instead of lambda for pickle compatibility
        self.last_time = defaultdict(dict)  # last_time[src][dst] = timestamp
        self.counts = defaultdict(_default_dict_factory)  # counts[src][dst] = count
        self.num_updates = 0
    
    def update(self, source, destination, timestamp):
        """
        Update the model with a new interaction.
        
        Args:
            source: Source node (user in tgbl-wiki)
            destination: Destination node (page in tgbl-wiki)
            timestamp: Time of interaction
        """
        self.last_time[source][destination] = timestamp
        self.counts[source][destination] += 1
        self.num_updates += 1
    
    def score_one(self, source, destination, timestamp):
        """
        Score a single (source, destination, timestamp) query.
        
        Args:
            source: Source node to query
            destination: Destination node to score
            timestamp: Query timestamp
            
        Returns:
            float: Score combining recency decay and frequency
        """
        # Get last interaction time for this (src, dst) pair
        last_ts = self.last_time[source].get(destination)
        
        if last_ts is None:
            # Never seen this (src, dst) pair
            return 0.0
        
        # Time since last interaction
        delta_t = max(0, int(timestamp) - int(last_ts))
        
        # Recency decay: 1 / (1 + Δt)
        recency_score = 1.0 / (1.0 + float(delta_t))
        
        # Frequency bonus: small weight on interaction count
        frequency_bonus = 1e-6 * float(self.counts[source][destination])
        
        return recency_score + frequency_bonus
    
    def score_batch(self, sources, destinations, timestamps):
        """
        Score a batch of (source, destination, timestamp) queries.
        
        Args:
            sources: Array of source nodes
            destinations: Array of destination nodes
            timestamps: Array of timestamps
            
        Returns:
            np.ndarray: Array of scores
        """
        scores = np.zeros(len(sources), dtype=np.float32)
        
        for i in range(len(sources)):
            scores[i] = self.score_one(
                int(sources[i]),
                int(destinations[i]),
                int(timestamps[i])
            )
        
        return scores
    
    def get_statistics(self):
        """
        Get statistics about the current model state.
        
        Returns:
            dict: Statistics including number of sources tracked and total updates
        """
        num_sources = len(self.last_time)
        num_pairs = sum(len(dsts) for dsts in self.last_time.values())
        
        return {
            'num_sources': num_sources,
            'num_pairs': num_pairs,
            'num_updates': self.num_updates,
        }