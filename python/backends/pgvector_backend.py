"""
Python implementation of a pgvector backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple
import psycopg2
from psycopg2.extras import execute_values
import os


class PgVectorBackend:
    """
    A pgvector backend that stores vectors in PostgreSQL with pgvector extension
    and performs approximate nearest neighbor search.
    """
    
    def __init__(self, table_name: str, dimension: int, 
                 db_host: str = "localhost",
                 db_port: int = 5432,
                 db_name: str = "postgres",
                 db_user: str = "postgres",
                 db_password: str = "postgres",
                 data_path: str = None,
                 recreate_table: bool = False,
                 metric: str = "l2",
                 hnsw_m: int = 16,
                 hnsw_ef_construction: int = 64,
                 rebuild_hnsw: bool = False):
        """
        Initialize the pgvector backend.
        
        Args:
            table_name: Name of the PostgreSQL table to store vectors
            dimension: Dimension of the vectors
            db_host: Host of the PostgreSQL service (default: localhost)
            db_port: Port of the PostgreSQL service (default: 5432)
            db_name: Database name (default: postgres)
            db_user: Database user (default: postgres)
            db_password: Database password (default: postgres)
            data_path: Optional path to binary data file to load vectors from (DiskANN format)
            recreate_table: If True, recreate the table even if it exists
            metric: Distance metric to use - "l2" or "cosine" (default: "l2")
            hnsw_m: HNSW max connections per node (default: 16)
            hnsw_ef_construction: HNSW construction candidate list size (default: 64)
            rebuild_hnsw: If True, drop existing HNSW indexes and rebuild
        """
        # Store connection parameters
        self.table_name = table_name
        self.dim = int(dimension)  # Ensure dimension is a Python int
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        
        # Normalize metric to lowercase
        metric_lower = metric.lower()
        if metric_lower not in ["l2", "cosine"]:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics: 'l2', 'cosine'")
        self.metric = metric_lower
        
        # Connect to PostgreSQL with retry logic
        import time
        max_retries = 10
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_password
                )
                self.conn.autocommit = True
                # Test connection
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect to PostgreSQL at {db_host}:{db_port} after {max_retries} attempts: {e}")
        
        # Enable pgvector extension
        with self.conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                print(f"Warning: Could not create vector extension (it may already exist): {e}")
        
        # Check if table exists
        table_exists = False
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            table_exists = cur.fetchone()[0]
        
        if recreate_table and table_exists:
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            table_exists = False
        
        if not table_exists:
            # Create table with pgvector column
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE {table_name} (
                        id BIGINT PRIMARY KEY,
                        vector vector({self.dim})
                    )
                """)
                # Note: Index will be created after data is loaded for better accuracy
                # IVFFlat index should be created after data insertion for optimal recall
            
            print(f"Created PostgreSQL table '{table_name}' with dimension {dimension}")
            
            # Load data if provided
            if data_path and os.path.exists(data_path):
                self._load_data_from_file(data_path)
        else:
            # Check current row count
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                num_entities = cur.fetchone()[0]
            print(f"Using existing PostgreSQL table '{table_name}'")
            print(f"Table contains {num_entities} vectors")
            
            # Load data if table is empty
            if num_entities == 0 and data_path and os.path.exists(data_path):
                print(f"Table exists but is empty. Loading data...")
                self._load_data_from_file(data_path)
            elif rebuild_hnsw:
                self.rebuild_hnsw_index()
        
        print(f"PgVectorBackend initialized with table '{table_name}' (metric: {self.metric})")
    
    def _load_data_from_file(self, data_path: str, batch_size: int = 10000):
        """
        Load vectors from a binary file (DiskANN format) into PostgreSQL.
        
        Args:
            data_path: Path to the binary data file
            batch_size: Number of vectors to insert per batch
        """
        print(f"Loading vectors from {data_path} into PostgreSQL...")
        
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        file_size = os.path.getsize(data_path)
        with open(data_path, 'rb') as f:
            num_vectors = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            dim = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            
            if dim != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {dim}")

            payload = file_size - 8
            elem_size = payload // (num_vectors * dim)
            if elem_size == 1:
                np_dtype = np.int8
            elif elem_size == 4:
                np_dtype = np.float32
            else:
                raise ValueError(
                    f"Cannot infer DiskANN element size from {data_path}: "
                    f"payload={payload} npts={num_vectors} dim={dim}"
                )
            print(f"Loading {num_vectors} vectors of dimension {dim} (source dtype={np_dtype.__name__})...")
            with self.conn.cursor() as cur:
                cur.execute("SET synchronous_commit TO OFF")

            loaded = 0
            remaining = num_vectors
            while remaining > 0:
                n = min(batch_size, remaining)
                raw = np.fromfile(f, dtype=np_dtype, count=n * dim)
                if raw.size != n * dim:
                    raise ValueError(
                        f"Truncated DiskANN payload in {data_path}: "
                        f"expected {n * dim} elements, got {raw.size}"
                    )
                batch = np.ascontiguousarray(raw.reshape(n, dim), dtype=np.float32)
                rows = [(int(loaded + j), batch[j].tolist()) for j in range(n)]
                with self.conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"INSERT INTO {self.table_name} (id, vector) VALUES %s",
                        rows,
                        template="(%s, %s::vector)",
                        page_size=batch_size,
                    )
                loaded += n
                remaining -= n
                print(f"Loaded {loaded}/{num_vectors} vectors...", end='\r')
            
            self._create_hnsw_index()
            print(f"\nLoaded {num_vectors} vectors into PostgreSQL")

    def _vector_ops(self) -> str:
        return "vector_cosine_ops" if self.metric == "cosine" else "vector_l2_ops"

    def _create_hnsw_index(self) -> None:
        ops = self._vector_ops()
        print(
            f"Creating HNSW index using {ops} "
            f"(m={self.hnsw_m}, ef_construction={self.hnsw_ef_construction})..."
        )
        with self.conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '8GB'")
            cur.execute("SET max_parallel_maintenance_workers = 8")
            cur.execute(f"""
                CREATE INDEX ON {self.table_name}
                USING hnsw (vector {ops})
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
            """)
        print(
            f"Created HNSW index with m={self.hnsw_m}, "
            f"ef_construction={self.hnsw_ef_construction} using {ops}"
        )

    def rebuild_hnsw_index(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = %s
                  AND indexdef ILIKE '%%USING hnsw%%'
                """,
                (self.table_name,),
            )
            names = [row[0] for row in cur.fetchall()]
            for name in names:
                print(f"Dropping HNSW index {name} on {self.table_name}")
                cur.execute(f'DROP INDEX IF EXISTS "{name}"')
        self._create_hnsw_index()
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using pgvector.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing distances (L2^2 for l2, cosine distance for cosine)
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must be 1D array of shape ({self.dim},), got {query.shape}")
        
        # Convert query to list format for pgvector
        # Ensure query is contiguous and float32
        query_contiguous = np.ascontiguousarray(query, dtype=np.float32)
        query_vector = query_contiguous.tolist()
        
        # Validate query vector
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dim}, got {len(query_vector)}")
        
        # Check for NaN or Inf in query
        if not np.all(np.isfinite(query_contiguous)):
            raise ValueError("Query vector contains NaN or Inf values")
        
        try:
            # Perform search using pgvector distance metric
            # For HNSW index, set ef_search parameter for better recall
            # Higher ef_search = better recall but slower queries
            # Check if HNSW index exists and set ef_search if needed
            cur_check = self.conn.cursor()
            has_hnsw = False
            try:
                cur_check.execute("""
                    SELECT COUNT(*) 
                    FROM pg_indexes 
                    WHERE tablename = %s AND indexdef LIKE '%%hnsw%%'
                """, (self.table_name,))
                result = cur_check.fetchone()
                has_hnsw = result is not None and result[0] > 0
            except Exception:
                pass
            finally:
                cur_check.close()
            
            cur = self.conn.cursor()
            try:
                if has_hnsw:
                    # Set ef_search for HNSW index (session-level, works with autocommit=True)
                    # ef_search should be >= K for good recall, typically 2-4x K
                    ef_search = max(K * 2, 200)  # At least 2x K, minimum 200
                    cur.execute(f"SET hnsw.ef_search = {ef_search}")
                
                # Format vector as string that pgvector expects: '[1.0,2.0,3.0]'
                vector_str = '[' + ','.join(f'{float(x):.6f}' for x in query_vector) + ']'
                
                # Use appropriate operator based on metric
                # <-> for L2 distance, <=> for cosine distance
                if self.metric == "cosine":
                    distance_op = "<=>"
                else:  # l2
                    distance_op = "<->"
                
                # Execute query
                query_sql = f"""
                    SELECT id, vector {distance_op} '{vector_str}'::vector AS distance
                    FROM {self.table_name}
                    ORDER BY vector {distance_op} '{vector_str}'::vector
                    LIMIT {K}
                """
                cur.execute(query_sql)
                results = cur.fetchall()
            finally:
                cur.close()
        except Exception as e:
            raise RuntimeError(f"Failed to search PostgreSQL table: {e}")
        
        # Extract tags and distances
        # For L2: pgvector's <-> returns actual L2 distance (with square root)
        # For cosine: pgvector's <=> returns cosine distance (1 - cosine similarity)
        tags = []
        distances = []
        
        # Process results
        for row in results:
            if len(row) < 2:
                continue
            vec_id, distance = row
            try:
                vec_id_int = int(vec_id)
                distance_float = float(distance)
                
                # Validate - distances should be finite and non-negative
                if not np.isfinite(distance_float) or distance_float < 0:
                    continue
                
                tags.append(vec_id_int)
                
                # Handle distance based on metric
                if self.metric == "cosine":
                    # Cosine distance: pgvector returns 1 - cosine_similarity
                    # Return as-is (cosine distance)
                    distances.append(float(distance_float))
                else:  # l2
                    # Square the L2 distance so it matches sum of squared differences.
                    squared_distance = distance_float * distance_float
                    distances.append(float(squared_distance))
            except (ValueError, TypeError, OverflowError):
                continue
        
        # Convert to numpy arrays
        tags = np.ascontiguousarray(np.array(tags, dtype=np.uint32))
        distances = np.ascontiguousarray(np.array(distances, dtype=np.float32))
        
        # Ensure we have exactly K results (pad if needed)
        # Note: We pad with invalid IDs (max uint32) instead of 0, since 0 might be a valid vector ID
        if len(tags) < K:
            padded_tags = np.full(K, np.iinfo(np.uint32).max, dtype=np.uint32)
            padded_distances = np.full(K, np.finfo(np.float32).max, dtype=np.float32)
            padded_tags[:len(tags)] = tags
            padded_distances[:len(distances)] = distances
            tags = padded_tags
            distances = padded_distances
        elif len(tags) > K:
            tags = tags[:K]
            distances = distances[:K]
        
        return tags, distances
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs from PostgreSQL.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        if not ids:
            return []
        
        try:
            # Retrieve vectors from PostgreSQL
            with self.conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(ids))
                cur.execute(f"""
                    SELECT id, vector::text
                    FROM {self.table_name}
                    WHERE id IN ({placeholders})
                """, tuple(ids))
                
                results = cur.fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve vectors from PostgreSQL: {e}")
        
        # Create a mapping of ID to vector
        id_to_vector = {}
        for row in results:
            vec_id, vector_data = row
            # pgvector returns vector as a list or array-like object
            if isinstance(vector_data, (list, tuple)):
                vector_list = vector_data
            else:
                # If it's a string, parse it
                vector_str = str(vector_data)
                vector_list = [float(x) for x in vector_str.strip('[]').split(',')]
            id_to_vector[int(vec_id)] = np.array(vector_list, dtype=np.float32)
        
        # Return vectors in the same order as requested IDs
        result_vectors = []
        for vec_id in ids:
            if vec_id in id_to_vector:
                result_vectors.append(id_to_vector[vec_id])
            else:
                # Return zero vector for invalid IDs
                result_vectors.append(np.zeros(self.dim, dtype=np.float32))
        
        return result_vectors

