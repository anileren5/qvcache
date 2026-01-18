"""
Python implementation of a bruteforce backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple


class BruteforceBackend:
    """
    A simple bruteforce backend that loads all vectors into memory
    and performs linear search.
    """
    
    def __init__(self, data_path: str, metric: str = "l2"):
        """
        Initialize the backend by loading data from a binary file.
        
        Args:
            data_path: Path to the binary data file (DiskANN format)
            metric: Distance metric to use - "l2", "cosine", or "inner_product" (default: "l2")
        """
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            # Read all vectors
            self.data = np.frombuffer(f.read(num_vectors * dim * 4), dtype=np.float32)
            self.data = self.data.reshape(num_vectors, dim)
        
        self.num_vectors = num_vectors
        self.dim = dim
        self.metric = metric.lower()
        
        if self.metric not in ["l2", "cosine", "inner_product", "innerproduct"]:
            raise ValueError(f"Unsupported metric: {metric}. Must be 'l2', 'cosine', or 'inner_product'")
        
        print(f"BruteforceBackend initialized with metric: {self.metric}")
        print(f"Loaded {self.num_vectors} vectors of dimension {self.dim} from {data_path}")
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using bruteforce distance.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing distances
                - For L2: squared L2 distances
                - For cosine: cosine distances (1 - cosine_similarity), matching DiskANN formula
                - For inner_product: negative inner product (-dot(query, data[i])), matching DiskANN/Greator formula
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must be 1D array of shape ({self.dim},), got {query.shape}")
        
        if self.metric == "inner_product" or self.metric == "innerproduct":
            # Implement exact DiskANN/Greator inner product distance formula:
            # inner_product_distance = -dot(a, b)
            # This matches DiskANN's AVXDistanceInnerProductFloat::compare() and Greator's implementation
            # Returns negative inner product so that smaller distances are better (for MIPS)
            
            # Compute dot products for all vectors
            dot_products = np.dot(self.data, query)  # dot(data[i], query) for each i
            
            # Return negative inner product (for distance, smaller is better)
            distances = -dot_products.astype(np.float32)
        
        elif self.metric == "cosine":
            # Implement exact DiskANN cosine distance formula:
            # cosine_distance = 1.0 - (dot(a,b) / (sqrt(||a||^2) * sqrt(||b||^2)))
            # This matches DiskANN's DistanceCosineFloat::compare() on Linux
            
            # Compute magnitudes and dot products for all vectors
            query_squared = np.sum(query ** 2)  # ||query||^2
            data_squared = np.sum(self.data ** 2, axis=1)  # ||data[i]||^2 for each i
            dot_products = np.dot(self.data, query)  # dot(data[i], query) for each i
            
            # Compute cosine distances using DiskANN's exact formula
            # Avoid division by zero by checking magnitudes
            query_mag = np.sqrt(query_squared)
            data_mags = np.sqrt(data_squared)
            
            # Handle zero vectors (should not happen, but be safe)
            valid_mask = (query_mag > 0) & (data_mags > 0)
            distances = np.full(self.num_vectors, np.finfo(np.float32).max, dtype=np.float32)
            
            if query_mag > 0:
                # Compute cosine similarity: dot(a,b) / (||a|| * ||b||)
                cosine_similarities = np.zeros(self.num_vectors, dtype=np.float32)
                cosine_similarities[valid_mask] = dot_products[valid_mask] / (query_mag * data_mags[valid_mask])
                
                # Cosine distance = 1 - cosine_similarity (matching DiskANN)
                distances[valid_mask] = 1.0 - cosine_similarities[valid_mask]
        
        else:  # L2
            # Compute L2 distances to all vectors
            # query shape: (dim,), data shape: (num_vectors, dim)
            # We want to compute ||query - data[i]||^2 for each i
            diff = self.data - query  # Broadcasting: (num_vectors, dim)
            distances = np.sum(diff ** 2, axis=1)  # (num_vectors,)
        
        # Get top K indices
        top_k_indices = np.argpartition(distances, K)[:K]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        # Extract tags and distances - create new arrays (not views) to ensure proper memory layout
        # Use .copy() to ensure we have independent arrays with their own memory
        tags = top_k_indices.astype(np.uint32, copy=True)
        top_k_distances = distances[top_k_indices].astype(np.float32, copy=True)
        
        # Ensure arrays are contiguous and have the correct shape
        tags = np.ascontiguousarray(tags.reshape(-1))
        top_k_distances = np.ascontiguousarray(top_k_distances.reshape(-1))
        
        return tags, top_k_distances
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs.
        
        Args:
            ids: List of vector IDs (0-indexed)
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        result = []
        for vec_id in ids:
            if 0 <= vec_id < self.num_vectors:
                result.append(self.data[vec_id].copy())
            else:
                # Return zero vector for invalid IDs
                result.append(np.zeros(self.dim, dtype=np.float32))
        return result

