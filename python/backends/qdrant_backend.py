"""
Python implementation of a Qdrant backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os


class QdrantBackend:
    """
    A Qdrant backend that stores vectors in Qdrant and performs approximate nearest neighbor search.
    """
    
    def __init__(self, collection_name: str, dimension: int, 
                 qdrant_url: str = "http://localhost:6333",
                 data_path: str = None,
                 recreate_collection: bool = False):
        """
        Initialize the Qdrant backend.
        
        Args:
            collection_name: Name of the Qdrant collection
            dimension: Dimension of the vectors
            qdrant_url: URL of the Qdrant service (default: http://localhost:6333)
            data_path: Optional path to binary data file to load vectors from (DiskANN format)
            recreate_collection: If True, recreate the collection even if it exists
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.dim = dimension
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if recreate_collection or not collection_exists:
            # Create or recreate collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.EUCLID  # L2 distance
                )
            )
            print(f"Created Qdrant collection '{collection_name}' with dimension {dimension}")
            
            # Load data if provided
            if data_path and os.path.exists(data_path):
                self._load_data_from_file(data_path)
        else:
            collection_info = self.client.get_collection(collection_name)
            print(f"Using existing Qdrant collection '{collection_name}'")
            print(f"Collection contains {collection_info.points_count} vectors")
        
        print(f"QdrantBackend initialized with collection '{collection_name}'")
    
    def _load_data_from_file(self, data_path: str, batch_size: int = 1000):
        """
        Load vectors from a binary file (DiskANN format) into Qdrant.
        
        Args:
            data_path: Path to the binary data file
            batch_size: Number of vectors to insert per batch
        """
        print(f"Loading vectors from {data_path} into Qdrant...")
        
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            if dim != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {dim}")
            
            print(f"Loading {num_vectors} vectors of dimension {dim}...")
            
            # Load vectors in batches
            points = []
            for i in range(num_vectors):
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                vector = vector.astype(np.float32).tolist()
                
                points.append(PointStruct(
                    id=i,
                    vector=vector
                ))
                
                # Insert batch when full
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
                    print(f"Loaded {i + 1}/{num_vectors} vectors...", end='\r')
            
            # Insert remaining vectors
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            print(f"\nLoaded {num_vectors} vectors into Qdrant")
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using Qdrant.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing L2 distances
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must be 1D array of shape ({self.dim},), got {query.shape}")
        
        # Convert query to list and perform search
        query_vector = query.astype(np.float32).tolist()
        
        # Use HTTP API directly for compatibility across versions
        try:
            from qdrant_client.http import models as rest_models
            search_request = rest_models.SearchRequest(
                vector=query_vector,
                limit=K,
                with_payload=False,
                with_vectors=False
            )
            search_response = self.client.http.collections_api.search_points(
                collection_name=self.collection_name,
                search_request=search_request
            )
            search_results = search_response.result
        except Exception as e:
            # If HTTP API fails, try client methods
            try:
                if hasattr(self.client, 'search'):
                    search_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=K
                    )
                elif hasattr(self.client, 'query_points'):
                    search_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=K
                    )
                else:
                    raise AttributeError("No search method found on QdrantClient")
            except Exception as e2:
                raise RuntimeError(f"Failed to search Qdrant collection. HTTP API error: {e}, Client API error: {e2}")
        
        # Extract tags and distances
        # IMPORTANT: Qdrant returns actual Euclidean distance (with square root), but BruteforceBackend
        # returns squared L2 distance (L2^2). We need to square Qdrant distances to match.
        tags = []
        distances = []
        
        # Handle different response formats
        # HTTP API returns a list of ScoredPoint directly
        if isinstance(search_results, list):
            # Standard format: list of ScoredPoint objects
            for result in search_results:
                if hasattr(result, 'id') and hasattr(result, 'score'):
                    tags.append(result.id)
                    # Square the distance to match BruteforceBackend format (L2^2)
                    distances.append(result.score * result.score)
        elif hasattr(search_results, 'points'):
            # New API format (query_points returns QueryResponse with points attribute)
            for point in search_results.points:
                if hasattr(point, 'id') and hasattr(point, 'score'):
                    tags.append(point.id)
                    # Square the distance to match BruteforceBackend format (L2^2)
                    distances.append(point.score * point.score)
        elif hasattr(search_results, 'result'):
            # HTTP API might return result wrapped
            for point in search_results.result:
                if hasattr(point, 'id') and hasattr(point, 'score'):
                    tags.append(point.id)
                    # Square the distance to match BruteforceBackend format (L2^2)
                    distances.append(point.score * point.score)
        else:
            # Try iterating directly
            try:
                for result in search_results:
                    if hasattr(result, 'id') and hasattr(result, 'score'):
                        tags.append(result.id)
                        # Square the distance to match BruteforceBackend format (L2^2)
                        distances.append(result.score * result.score)
            except (TypeError, AttributeError) as e:
                raise RuntimeError(f"Could not parse Qdrant search results. Type: {type(search_results)}, Error: {e}")
        
        # Convert to numpy arrays
        tags = np.ascontiguousarray(np.array(tags, dtype=np.uint32))
        distances = np.ascontiguousarray(np.array(distances, dtype=np.float32))
        
        # Ensure we have exactly K results (pad if needed)
        # Note: We pad with invalid IDs (max uint32) instead of 0, since 0 might be a valid vector ID
        if len(tags) < K:
            padded_tags = np.full(K, np.iinfo(np.uint32).max, dtype=np.uint32)  # Use max uint32 as invalid marker
            padded_distances = np.full(K, np.finfo(np.float32).max, dtype=np.float32)
            padded_tags[:len(tags)] = tags
            padded_distances[:len(distances)] = distances
            tags = padded_tags
            distances = padded_distances
        
        return tags, distances
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs from Qdrant.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        if not ids:
            return []
        
        # Retrieve vectors from Qdrant using HTTP API for compatibility
        try:
            # Try HTTP API first
            from qdrant_client.http import models as rest_models
            retrieve_request = rest_models.PointIdsList(
                points=ids
            )
            retrieve_response = self.client.http.points_api.get_points(
                collection_name=self.collection_name,
                point_request=retrieve_request,
                with_vectors=True,
                with_payload=False
            )
            points = retrieve_response.result
        except Exception as e:
            # Fallback to client method if HTTP API fails
            try:
                if hasattr(self.client, 'retrieve'):
                    points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=ids,
                        with_vectors=True,
                        with_payload=False
                    )
                else:
                    raise AttributeError("No retrieve method found on QdrantClient")
            except Exception as e2:
                raise RuntimeError(f"Failed to retrieve vectors from Qdrant. HTTP API error: {e}, Client API error: {e2}")
        
        # Create a mapping of ID to vector
        # Handle different response formats
        id_to_vector = {}
        if isinstance(points, list):
            # Standard format: list of Record objects
            for point in points:
                if hasattr(point, 'id') and hasattr(point, 'vector') and point.vector is not None:
                    id_to_vector[point.id] = np.array(point.vector, dtype=np.float32)
        elif hasattr(points, 'result'):
            # HTTP API might return result wrapped
            for point in points.result:
                if hasattr(point, 'id') and hasattr(point, 'vector') and point.vector is not None:
                    id_to_vector[point.id] = np.array(point.vector, dtype=np.float32)
        else:
            # Try iterating directly
            try:
                for point in points:
                    if hasattr(point, 'id') and hasattr(point, 'vector') and point.vector is not None:
                        id_to_vector[point.id] = np.array(point.vector, dtype=np.float32)
            except (TypeError, AttributeError) as e:
                raise RuntimeError(f"Could not parse Qdrant retrieve results. Type: {type(points)}, Error: {e}")
        
        # Return vectors in the same order as requested IDs
        result = []
        for vec_id in ids:
            if vec_id in id_to_vector:
                result.append(id_to_vector[vec_id])
            else:
                # Return zero vector for invalid IDs
                result.append(np.zeros(self.dim, dtype=np.float32))
        
        return result

