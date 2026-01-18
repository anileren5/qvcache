"""
Python implementation of a FAISS backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple
import faiss
import os


class FaissBackend:
    """
    A FAISS backend that stores vectors in FAISS and performs approximate nearest neighbor search.
    """
    
    def __init__(self, index_path: str = None, dimension: int = None, 
                 data_path: str = None,
                 index_type: str = "HNSW",
                 recreate_index: bool = False,
                 hnsw_m: int = 32,
                 hnsw_ef_construction: int = 200):
        """
        Initialize the FAISS backend.
        
        Args:
            index_path: Path to save/load FAISS index (default: ./faiss_index.bin)
            dimension: Dimension of the vectors (required if creating new index)
            data_path: Optional path to binary data file to load vectors from (DiskANN format)
            index_type: Type of index ("HNSW" or "Flat") - default: "HNSW" for better performance
            recreate_index: If True, recreate the index even if it exists
            hnsw_m: HNSW parameter M (number of bi-directional links) - default: 32
            hnsw_ef_construction: HNSW parameter ef_construction - default: 200
        """
        if index_path is None:
            index_path = "./faiss_index.bin"
        
        self.index_path = index_path
        self.dim = dimension
        self.index_type = index_type
        self.vectors = None  # Store vectors for fetch_vectors_by_ids
        
        # Check if index exists
        index_exists = os.path.exists(index_path) and os.path.getsize(index_path) > 0
        
        if recreate_index and index_exists:
            os.remove(index_path)
            index_exists = False
        
        if not index_exists:
            # Create new index
            if dimension is None:
                raise ValueError("dimension must be provided when creating a new index")
            
            if index_type == "HNSW":
                # HNSW index for approximate search (better performance, good recall)
                # IndexHNSWFlat constructor: (dimension, M, metric_type)
                # Use 3-arg version with explicit metric type (METRIC_L2 = 0 for L2 distance)
                self.index = faiss.IndexHNSWFlat(int(dimension), int(hnsw_m), faiss.METRIC_L2)
                self.index.hnsw.efConstruction = hnsw_ef_construction
                self.index.hnsw.efSearch = max(200, hnsw_ef_construction // 2)  # ef_search for queries
            else:
                # Flat index for exact search (slower but perfect recall)
                self.index = faiss.IndexFlatL2(dimension)
            
            print(f"Created FAISS {index_type} index with dimension {dimension}")
            
            # Load data if provided
            if data_path and os.path.exists(data_path):
                self._load_data_from_file(data_path)
            
            # Save index
            faiss.write_index(self.index, index_path)
            print(f"Saved FAISS index to {index_path}")
        else:
            # Load existing index
            self.index = faiss.read_index(index_path)
            self.dim = self.index.d
            print(f"Loaded FAISS index from {index_path}")
            print(f"Index contains {self.index.ntotal} vectors")
            
            # Load vectors if needed for fetch_vectors_by_ids
            # For HNSW, we need to store vectors separately
            if isinstance(self.index, faiss.IndexHNSWFlat):
                # Try to load vectors from a separate file
                vectors_path = index_path.replace(".bin", "_vectors.bin")
                if os.path.exists(vectors_path):
                    self.vectors = np.fromfile(vectors_path, dtype=np.float32)
                    num_vectors = self.index.ntotal
                    self.vectors = self.vectors.reshape(num_vectors, self.dim)
                    print(f"Loaded {num_vectors} vectors from {vectors_path}")
                elif data_path and os.path.exists(data_path):
                    # Load vectors from data file
                    self._load_vectors_for_fetch(data_path)
        
        print(f"FaissBackend initialized with {self.index.ntotal} vectors")
    
    def _load_data_from_file(self, data_path: str):
        """
        Load vectors from a binary file (DiskANN format) into FAISS.
        
        Args:
            data_path: Path to the binary data file
        """
        print(f"Loading vectors from {data_path} into FAISS...")
        
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            if dim != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {dim}")
            
            print(f"Loading {num_vectors} vectors of dimension {dim}...")
            
            # Read all vectors
            vectors = np.frombuffer(f.read(num_vectors * dim * 4), dtype=np.float32)
            vectors = vectors.reshape(num_vectors, dim)
            
            # Normalize vectors (FAISS L2 index requires normalized vectors for best results)
            # Actually, FAISS IndexFlatL2 and IndexHNSWFlat work with unnormalized vectors
            # But we'll keep them as-is for consistency with other backends
            
            # Add vectors to index
            self.index.add(vectors.astype(np.float32))
            
            # Store vectors for fetch_vectors_by_ids (needed for HNSW)
            if isinstance(self.index, faiss.IndexHNSWFlat):
                self.vectors = vectors.astype(np.float32)
                # Save vectors to separate file
                vectors_path = self.index_path.replace(".bin", "_vectors.bin")
                self.vectors.tofile(vectors_path)
                print(f"Saved vectors to {vectors_path}")
            
            print(f"Loaded {self.index.ntotal} vectors into FAISS")
    
    def _load_vectors_for_fetch(self, data_path: str):
        """
        Load vectors from data file for fetch_vectors_by_ids (needed for HNSW index).
        
        Args:
            data_path: Path to the binary data file
        """
        print(f"Loading vectors from {data_path} for fetch_vectors_by_ids...")
        
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            vectors = np.frombuffer(f.read(num_vectors * dim * 4), dtype=np.float32)
            self.vectors = vectors.reshape(num_vectors, dim).astype(np.float32)
            
            # Save vectors to separate file for future use
            vectors_path = self.index_path.replace(".bin", "_vectors.bin")
            self.vectors.tofile(vectors_path)
            print(f"Saved {num_vectors} vectors to {vectors_path}")
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using FAISS.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing squared L2 distances
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must be 1D array of shape ({self.dim},), got {query.shape}")
        
        query = query.astype(np.float32).reshape(1, -1)  # Reshape to (1, dim) for FAISS
        
        # Set ef_search for HNSW if applicable
        if isinstance(self.index, faiss.IndexHNSWFlat):
            # Use higher ef_search for better recall
            self.index.hnsw.efSearch = max(K * 2, 200)
        
        # Perform search
        # FAISS returns squared L2 distances directly (matches BruteforceBackend format)
        distances, indices = self.index.search(query, K)
        
        # Extract results
        tags = indices[0].astype(np.uint32)  # Convert to uint32
        dists = distances[0].astype(np.float32)  # Already squared L2
        
        # Ensure we have exactly K results (pad if needed)
        if len(tags) < K:
            padded_tags = np.full(K, np.iinfo(np.uint32).max, dtype=np.uint32)
            padded_distances = np.full(K, np.finfo(np.float32).max, dtype=np.float32)
            padded_tags[:len(tags)] = tags
            padded_distances[:len(dists)] = dists
            tags = padded_tags
            dists = padded_distances
        elif len(tags) > K:
            tags = tags[:K]
            dists = dists[:K]
        
        return tags, dists
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs from FAISS.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        if not ids:
            return []
        
        if self.vectors is None:
            # For Flat index, we can reconstruct from index
            # But for HNSW, we need stored vectors
            raise RuntimeError("Vectors not available for fetch_vectors_by_ids. "
                             "Please provide data_path when initializing the backend.")
        
        result_vectors = []
        for vec_id in ids:
            if 0 <= vec_id < len(self.vectors):
                vector = self.vectors[vec_id].copy()
                if vector.shape[0] == self.dim:
                    result_vectors.append(vector)
                else:
                    result_vectors.append(np.zeros(self.dim, dtype=np.float32))
            else:
                # Invalid ID, return zero vector
                result_vectors.append(np.zeros(self.dim, dtype=np.float32))
        
        return result_vectors

