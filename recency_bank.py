"""
RecencyBank: Temporal Heuristic for Link Prediction

A simple baseline that predicts users will edit the pages they most recently edited.
Based on the observation that Wikipedia editors exhibit strong temporal locality.

Author: [Your Name]
Course: Special Lectures in Mathematical Informatics II
Dataset: tgbl-wiki from Temporal Graph Benchmark
"""


class RecencyBank:
    """
    RecencyBank maintains the most recent destination for each source node.
    
    The algorithm is based on temporal locality: in collaborative editing systems
    like Wikipedia, users tend to repeatedly edit the same pages within short
    time windows due to topical specialization, maintenance work, and focused
    contribution patterns.
    
    Complexity:
        - Training: O(|E|) time, O(|V|) space
        - Query: O(1) time
    
    Attributes:
        last_interaction (dict): Maps each source to its most recent destination
        num_updates (int): Total number of updates processed
    """
    
    def __init__(self):
        """Initialize empty RecencyBank."""
        self.last_interaction = {}
        self.num_updates = 0
    
    def update(self, source, destination, timestamp):
        """
        Update the model with a new interaction.
        
        Args:
            source: Source node (user in tgbl-wiki)
            destination: Destination node (page in tgbl-wiki)
            timestamp: Time of interaction (used for maintaining temporal order)
        
        Note:
            This implementation only stores the most recent destination per source.
            The timestamp is accepted for compatibility but not stored, as only
            the latest interaction matters for prediction.
        """
        self.last_interaction[source] = destination
        self.num_updates += 1
    
    def query(self, source):
        """
        Query the most recent destination for a source.
        
        Args:
            source: Source node to query
            
        Returns:
            Most recent destination for the source, or None if source
            has no history (cold start case)
        """
        return self.last_interaction.get(source, None)
    
    def get_statistics(self):
        """
        Get statistics about the current model state.
        
        Returns:
            dict: Statistics including number of sources tracked and total updates
        """
        return {
            'num_sources': len(self.last_interaction),
            'num_updates': self.num_updates,
        }
