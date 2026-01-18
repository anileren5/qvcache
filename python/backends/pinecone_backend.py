"""
Python implementation of a Pinecone backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple
from pinecone import Pinecone
import os

# Try to import spec classes (required for index creation)
try:
    from pinecone import ServerlessSpec
    HAS_SERVERLESS_SPEC = True
except ImportError:
    ServerlessSpec = None
    HAS_SERVERLESS_SPEC = False

try:
    from pinecone import PodSpec
    HAS_POD_SPEC = True
except ImportError:
    PodSpec = None
    HAS_POD_SPEC = False


class PineconeBackend:
    """
    A Pinecone backend that stores vectors in Pinecone and performs approximate nearest neighbor search.
    """
    
    def __init__(self, index_name: str, dimension: int, 
                 api_key: str = None,
                 environment: str = None,
                 host: str = None,
                 data_path: str = None,
                 recreate_index: bool = False):
        """
        Initialize the Pinecone backend.
        
        Args:
            index_name: Name of the Pinecone index
                For cloud: Check your Pinecone dashboard for the exact index name
            dimension: Dimension of the vectors
            api_key: Pinecone API key
                For cloud: Your API key (starts with "pcsk_")
                For local: Use "pclocal" or "local"
                Default: from PINECONE_API_KEY env var
            environment: Pinecone environment/region
                For cloud: Required! (e.g., "us-east-1", "us-west-2", "eu-west-1")
                For local: Not needed
            host: Pinecone host
                For cloud: Not needed (leave as None)
                For local Docker: Use service name like "pinecone"
            data_path: Optional path to binary data file to load vectors from (DiskANN format)
            recreate_index: If True, recreate the index even if it exists
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("PINECONE_API_KEY", "pclocal")  # Use "pclocal" for local Pinecone
        
        # Determine if this is local or cloud Pinecone
        # Local Pinecone uses "pclocal" or "local" as API key
        # Cloud Pinecone uses actual API keys (usually starting with "pcsk_")
        self.is_local = api_key in ["local", "pclocal"]
        
        if self.is_local:
            # For local Pinecone, convert "local" to "pclocal"
            if api_key == "local":
                api_key = "pclocal"
            # Store host for local Pinecone (for Docker networking)
            self.pinecone_host = host if host else "localhost"
        else:
            # Cloud Pinecone - host parameter is not used
            self.pinecone_host = None
        
        # Initialize Pinecone client
        if self.is_local:
            # For local Pinecone, initialize client
            # The client will default to localhost, but we can configure Index separately
            self.pc = Pinecone(api_key=api_key)
        else:
            # Cloud Pinecone - use the provided API key
            self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        # Ensure dimension is a Python int (not numpy type) for JSON serialization
        self.dim = int(dimension)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        index_exists = index_name in existing_indexes
        
        if recreate_index and index_exists:
            self.pc.delete_index(index_name)
            index_exists = False
        
        if not index_exists:
            # Create index
            # Pinecone API now requires a spec parameter
            try:
                # Pinecone API requires spec parameter
                if not HAS_SERVERLESS_SPEC and not HAS_POD_SPEC:
                    raise RuntimeError("Pinecone spec classes not available. Please ensure pinecone package is properly installed.")
                
                if self.is_local:
                    # Local Pinecone - use ServerlessSpec (local Pinecone accepts this)
                    # Local Pinecone will ignore the cloud/region values
                    if HAS_SERVERLESS_SPEC:
                        self.pc.create_index(
                            name=index_name,
                            dimension=self.dim,
                            metric="euclidean",
                            spec=ServerlessSpec(
                                cloud="aws",  # Local Pinecone accepts any value
                                region="us-east-1"  # Local Pinecone accepts any value
                            )
                        )
                    elif HAS_POD_SPEC:
                        # Fallback to PodSpec
                        self.pc.create_index(
                            name=index_name,
                            dimension=self.dim,
                            metric="euclidean",
                            spec=PodSpec(
                                environment="local",
                                pod_type="p1.x1",
                                pods=1
                            )
                        )
                    else:
                        raise RuntimeError("No spec classes available for local Pinecone.")
                else:
                    # Cloud Pinecone - use ServerlessSpec with provided environment/region
                    if not environment:
                        raise RuntimeError("For cloud Pinecone, 'environment' parameter (region) is required. Example: 'us-east-1', 'us-west-2'")
                    
                    if HAS_SERVERLESS_SPEC:
                        self.pc.create_index(
                            name=index_name,
                            dimension=self.dim,
                            metric="euclidean",  # L2 distance
                            spec=ServerlessSpec(
                                cloud="aws",
                                region=environment
                            )
                        )
                    else:
                        raise RuntimeError("ServerlessSpec not available. Please install pinecone package with cloud support.")
            except Exception as e:
                raise RuntimeError(f"Failed to create Pinecone index: {e}")
            
            print(f"Created Pinecone index '{index_name}' with dimension {self.dim}")
        else:
            print(f"Using existing Pinecone index '{index_name}'")
        
        # Connect to the index (must be done before loading data)
        self.index = self.pc.Index(index_name)
        
        # For local Pinecone in Docker, patch the connection to use service name
        if self.is_local and self.pinecone_host and self.pinecone_host != "localhost":
            # Try to patch the index's internal HTTP client to use the Docker service name
            try:
                # The Pinecone Index uses an internal API client
                # Try to find and patch the base URL
                if hasattr(self.index, '_api'):
                    api_obj = self.index._api
                    # Try different possible attribute names for the base URL
                    if hasattr(api_obj, 'base_url'):
                        original_url = api_obj.base_url
                        # Replace localhost/127.0.0.1 with the service name
                        new_url = original_url.replace('localhost', self.pinecone_host)
                        new_url = new_url.replace('127.0.0.1', self.pinecone_host)
                        new_url = new_url.replace('0.0.0.0', self.pinecone_host)
                        api_obj.base_url = new_url
                    elif hasattr(api_obj, '_base_url'):
                        original_url = api_obj._base_url
                        new_url = original_url.replace('localhost', self.pinecone_host)
                        new_url = new_url.replace('127.0.0.1', self.pinecone_host)
                        new_url = new_url.replace('0.0.0.0', self.pinecone_host)
                        api_obj._base_url = new_url
                    # Also try to patch the HTTP client if it exists
                    if hasattr(api_obj, '_client') and hasattr(api_obj._client, 'base_url'):
                        original_url = api_obj._client.base_url
                        new_url = original_url.replace('localhost', self.pinecone_host)
                        new_url = new_url.replace('127.0.0.1', self.pinecone_host)
                        new_url = new_url.replace('0.0.0.0', self.pinecone_host)
                        api_obj._client.base_url = new_url
            except (AttributeError, Exception) as e:
                # If patching doesn't work, log but continue
                # The connection might still work if Docker DNS resolves correctly
                print(f"Warning: Could not patch Pinecone host to {self.pinecone_host}: {e}")
        
        # Get index stats to check if data needs to be loaded
        try:
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"Index contains {vector_count} vectors")
        except Exception as e:
            print(f"Could not get index stats: {e}")
            vector_count = 0
        
        # Load data if provided (after index is connected)
        # Load data if: index was just created OR data_path is provided and index is empty
        if data_path and os.path.exists(data_path):
            if not index_exists:
                # New index - always load data
                print(f"Loading data into new index...")
                self._load_data_from_file(data_path)
            elif vector_count == 0:
                # Existing index but empty - load data
                print(f"Index exists but is empty. Loading data...")
                self._load_data_from_file(data_path)
            else:
                # Index exists and has data - skip loading
                print(f"Index already contains {vector_count} vectors. Skipping data load.")
                print(f"To reload data, use --recreate flag or delete the index manually.")
        
        print(f"PineconeBackend initialized with index '{index_name}'")
    
    def _load_data_from_file(self, data_path: str, batch_size: int = 100):
        """
        Load vectors from a binary file (DiskANN format) into Pinecone.
        
        Args:
            data_path: Path to the binary data file
            batch_size: Number of vectors to insert per batch (Pinecone recommends 100-1000)
        """
        print(f"Loading vectors from {data_path} into Pinecone...")
        
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            
            if dim != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {dim}")
            
            print(f"Loading {num_vectors} vectors of dimension {dim}...")
            
            # Load vectors in batches
            vectors_to_upsert = []
            for i in range(num_vectors):
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                vector = vector.astype(np.float32).tolist()
                
                # Pinecone format: (id, vector, metadata)
                vectors_to_upsert.append((str(i), vector))
                
                # Insert batch when full
                if len(vectors_to_upsert) >= batch_size:
                    try:
                        upsert_response = self.index.upsert(vectors=vectors_to_upsert)
                        # Log batch insertion success
                        if (i + 1) % (batch_size * 10) == 0 or i == num_vectors - 1:
                            print(f"✓ Inserted batch: {i + 1}/{num_vectors} vectors (batch size: {len(vectors_to_upsert)})")
                        vectors_to_upsert = []
                        print(f"Progress: {i + 1}/{num_vectors} vectors ({100.0 * (i + 1) / num_vectors:.1f}%)...", end='\r')
                    except Exception as e:
                        print(f"\n✗ ERROR: Failed to insert batch at vector {i + 1}: {e}")
                        raise
            
            # Insert remaining vectors
            if vectors_to_upsert:
                try:
                    upsert_response = self.index.upsert(vectors=vectors_to_upsert)
                    print(f"\n✓ Inserted final batch: {len(vectors_to_upsert)} vectors")
                except Exception as e:
                    print(f"\n✗ ERROR: Failed to insert final batch: {e}")
                    raise
            
            print(f"\n✓ Successfully loaded {num_vectors} vectors into Pinecone")
            
            # Verify insertion by checking index stats
            try:
                stats = self.index.describe_index_stats()
                actual_count = stats.get('total_vector_count', 0)
                if actual_count == num_vectors:
                    print(f"✓ Verification: Index contains {actual_count:,} vectors (matches expected)")
                else:
                    print(f"⚠ WARNING: Index contains {actual_count:,} vectors, expected {num_vectors:,}")
            except Exception as e:
                print(f"⚠ WARNING: Could not verify vector count: {e}")
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using Pinecone.
        
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
        
        try:
            # Perform search
            results = self.index.query(
                vector=query_vector,
                top_k=K,
                include_metadata=False,
                include_values=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to search Pinecone index: {e}")
        
        # Extract tags and distances
        # IMPORTANT: Pinecone returns squared Euclidean distance, but BruteforceBackend
        # also returns squared L2 distance (L2^2). They should match, but let's verify.
        tags = []
        distances = []
        
        # Pinecone query returns a QueryResponse dataclass, not a dict
        # Access matches using attribute access
        if results and hasattr(results, 'matches'):
            for match in results.matches:
                # Pinecone IDs are strings, convert to int then to uint32
                try:
                    # Match is also a dataclass, access id and score as attributes
                    vec_id = int(match.id)
                    tags.append(vec_id)
                    # Pinecone returns squared distance for euclidean metric
                    distances.append(match.score)
                except (ValueError, AttributeError, TypeError) as e:
                    continue
        elif results and isinstance(results, dict) and 'matches' in results:
            # Fallback: handle dict format if it exists
            for match in results['matches']:
                try:
                    if isinstance(match, dict):
                        vec_id = int(match['id'])
                        tags.append(vec_id)
                        distances.append(match['score'])
                    else:
                        # Match is a dataclass
                        vec_id = int(match.id)
                        tags.append(vec_id)
                        distances.append(match.score)
                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    continue
        
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
        Fetch vectors by their IDs from Pinecone.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        if not ids:
            return []
        
        try:
            # Pinecone fetch requires string IDs
            id_strings = [str(id) for id in ids]
            results = self.index.fetch(ids=id_strings)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve vectors from Pinecone: {e}")
        
        # Create a mapping of ID to vector
        # Pinecone fetch returns a FetchResponse dataclass, not a dict
        id_to_vector = {}
        
        # Handle dataclass format (new API)
        if hasattr(results, 'vectors'):
            vectors_dict = results.vectors
            for vec_id_str, vector_data in vectors_dict.items():
                try:
                    vec_id = int(vec_id_str)
                    # Vector data might be a dataclass or dict
                    if hasattr(vector_data, 'values'):
                        # Dataclass format
                        id_to_vector[vec_id] = np.array(vector_data.values, dtype=np.float32)
                    elif isinstance(vector_data, dict) and 'values' in vector_data:
                        # Dict format
                        id_to_vector[vec_id] = np.array(vector_data['values'], dtype=np.float32)
                    else:
                        # Vector data might be directly the values (list)
                        id_to_vector[vec_id] = np.array(vector_data, dtype=np.float32)
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    continue
        # Fallback: handle dict format
        elif isinstance(results, dict) and 'vectors' in results:
            for vec_id_str, vector_data in results['vectors'].items():
                try:
                    vec_id = int(vec_id_str)
                    if isinstance(vector_data, dict) and 'values' in vector_data:
                        id_to_vector[vec_id] = np.array(vector_data['values'], dtype=np.float32)
                    elif hasattr(vector_data, 'values'):
                        # Dataclass in dict
                        id_to_vector[vec_id] = np.array(vector_data.values, dtype=np.float32)
                    else:
                        # Vector data might be directly the values
                        id_to_vector[vec_id] = np.array(vector_data, dtype=np.float32)
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    continue
        
        # Return vectors in the same order as requested IDs
        result_vectors = []
        for vec_id in ids:
            if vec_id in id_to_vector:
                result_vectors.append(id_to_vector[vec_id])
            else:
                # Return zero vector for invalid IDs
                result_vectors.append(np.zeros(self.dim, dtype=np.float32))
        
        return result_vectors


