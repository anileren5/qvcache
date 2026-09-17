# QVCache

**QVCache: A Query-Aware Vector Cache**
VLDB Artifact Submission

This repository provides a fully dockerized experimental environment for reproducing the results of **QVCache**, including all different backend databases and benchmarking scripts.

---

## 1. Environment Setup

We provide a Docker-based environment that allows rapid deployment and experimentation with QVCache. The following command launches:

* The main QVCache container
* A `pgvector` backend container
* A `Qdrant` backend container

```bash
docker compose up -d
```

---

## 2. Enter the QVCache Container

After the containers are created, enter the QVCache container where all experiments and scripts are executed:

```bash
docker exec -it qvcache /bin/bash
```

---

## 3. Build the Project

Inside the container, build all required components:

```bash
./build.sh
```

---

## 4. Datasets

Datasets must be placed under the `data/` directory using the following structure:

```text
data/
  dataset_name/
    dataset_base.bin
    dataset_query.bin
    dataset_groundtruth.bin
```

All benchmark scripts operate on `.bin` formatted datasets.

### Supported Conversions

If your datasets are in `.fvecs` or `.ivecs` format, use the provided utilities:

```bash
./build/utils/fvecs_to_bin <float/int8/uint8> input_vecs output_bin
./build/utils/ivecs_to_bin input_ivecs output_bin
```

A sample dataset (`siftsmall`) is included in the repository for illustration and quick testing.

> **Note**: All shell scripts are self-documented and parameterized. Users can easily modify them to evaluate different datasets, system configurations, and experimental settings.

---

## 5. Index Construction

Use the following scripts to build indexes for each backend:

* **DiskANN**

  ```bash
  ./scripts/qvcache/diskann/build_index.sh
  ```

* **FAISS**

  ```bash
  ./python/scripts/build_index/build_faiss_index.sh
  ```

* **Qdrant**

  ```bash
  ./python/scripts/build_index/build_qdrant_index.sh
  ```

* **pgvector**

  ```bash
  ./python/scripts/build_index/build_pgvector_index.sh
  ```

* **Pinecone**

  ```bash
  ./python/scripts/build_index/build_pinecone_index.sh
  ```

  *(Requires a valid Pinecone API key from [https://www.pinecone.io/](https://www.pinecone.io/))*

* **SPANN (SPTAG)**

  ```bash
  ./backends/SPTAG/build.sh
  ./backends/SPTAG/build_index.sh
  ```

---

## 6. Query Execution and Benchmarking

Each backend can be evaluated **with** and **without QVCache**.

### DiskANN

* Without QVCache:

  ```bash
  ./scripts/qvcache/backend_benchmark_diskann.sh
  ```
* With QVCache:

  ```bash
  ./scripts/qvcache/qvcache_benchmark_diskann.sh
  ```

### FAISS

* Without QVCache:

  ```bash
  ./python/scripts/benchmark/backend_benchmark_faiss_backend.sh
  ```
* With QVCache:

  ```bash
  ./python/scripts/benchmark/qvcache_benchmark_faiss_backend.sh
  ```

### Qdrant

* Without QVCache:

  ```bash
  ./python/scripts/benchmark/backend_benchmark_qdrant_backend.sh
  ```
* With QVCache:

  ```bash
  ./python/scripts/benchmark/qvcache_benchmark_qdrant_backend.sh
  ```

*(Qdrant container is automatically started during environment setup.)*

### pgvector

* Without QVCache:

  ```bash
  ./python/scripts/benchmark/backend_benchmark_pgvector_backend.sh
  ```
* With QVCache:

  ```bash
  ./python/scripts/benchmark/qvcache_benchmark_pgvector_backend.sh
  ```

*(pgvector container is automatically started during environment setup.)*

### Pinecone

* Without QVCache:

  ```bash
  ./python/scripts/benchmark/backend_benchmark_pinecone_backend.sh
  ```
* With QVCache:

  ```bash
  ./python/scripts/benchmark/qvcache_benchmark_pinecone_backend.sh
  ```

### SPANN (SPTAG)

Start the SPANN server first:

```bash
./backends/SPTAG/start_server.sh
```

Then run benchmarks:

* Without QVCache:

  ```bash
  ./scripts/qvcache/backend_benchmark_sptag.sh
  ```
* With QVCache:

  ```bash
  ./scripts/qvcache/qvcache_benchmark_sptag.sh
  ```

---

## 7. Integrating a Custom Backend

To test QVCache with a new backend database, you can create a Python backend class that implements the required interface. The backend must provide two main methods:

### Required Interface

1. **`search(query, K)`**: Performs a K-nearest neighbor search
   - Input: `query` (numpy array), `K` (int)
   - Output: Tuple of `(tags, distances)` where both are numpy arrays of shape `(K,)`

2. **`fetch_vectors_by_ids(ids)`**: Retrieves vectors by their IDs
   - Input: `ids` (list of integers)
   - Output: List of numpy arrays, each representing a vector

### Minimal Example

Here's a minimal template for creating a custom backend:

```python
import numpy as np
from typing import List, Tuple

class CustomBackend:
    def __init__(self, data_path: str, metric: str = "l2"):
        """
        Initialize the backend.
        
        Args:
            data_path: Path to your data/index file
            metric: Distance metric ("l2", "cosine", or "inner_product")
        """
        # Load your data/index here
        self.metric = metric
        # ... your initialization code ...
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing distances
        """
        # Implement your search logic here
        # ... your search code ...
        return tags, distances
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        # Implement your fetch logic here
        # ... your fetch code ...
        return vectors
```

### Complete Example

For a complete, working example that demonstrates proper implementation of all methods including metric handling (L2, cosine, inner product), see [`python/backends/bruteforce_backend.py`](python/backends/bruteforce_backend.py). This file shows:

- Proper data loading from binary format
- Implementation of multiple distance metrics (L2, cosine, inner product)
- Correct handling of numpy array types and memory layout
- Error handling and validation

### Integration with QVCache

Once your backend class is implemented, you can integrate it with QVCache by following the pattern in [`python/benchmarks/qvcache_benchmark_faiss_backend.py`](python/benchmarks/qvcache_benchmark_faiss_backend.py):

```python
import qvcache as qvc
from backends.custom_backend import CustomBackend

# Initialize your backend
backend = CustomBackend(data_path="data/dataset/dataset_base.bin", metric="l2")

# Create QVCache with your backend
qvcache = qvc.QVCache(
    data_path="data/dataset/dataset_base.bin",  # Base data file used to construct the PCA transformation matrix
    pca_prefix="pca_index",
    R=64,
    memory_L=128,
    B=8,
    M=8,
    alpha=1.2,
    build_threads=8,
    search_threads=24,
    use_reconstructed_vectors=False,
    p=0.9,
    deviation_factor=0.25,
    memory_index_max_points=100000,
    beamwidth=2,
    use_regional_theta=True,
    pca_dim=16,
    buckets_per_dim=8,
    max_regions=18446744073709551615,  # Unlimited by default
    n_async_insert_threads=16,
    lazy_theta_updates=True,
    number_of_mini_indexes=4,
    search_mini_indexes_in_parallel=False,
    max_search_threads=32,
    backend=backend  # Pass your backend here
)

# Set search strategy (optional)
qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
```

---

## Reproducibility

All experiments are fully script-driven and containerized, ensuring reproducibility of results for artifact evaluation. Configuration parameters, datasets, and backend settings can be modified directly through the provided scripts to support extended experimental analysis.
