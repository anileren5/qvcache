"""
Utility functions shared across benchmark scripts.
"""

import time
import json
import numpy as np
from typing import List, Tuple

try:
    import qvcache as qvc
except ImportError:
    qvc = None


def hybrid_search(
    qvcache: 'qvc.QVCache',
    queries: np.ndarray,
    K: int,
    search_threads: int,
    data_path: str
) -> Tuple[List[bool], List[float], np.ndarray, dict]:
    """
    Perform hybrid search on queries.
    
    Args:
        qvcache: QVCache instance (QVCache object)
        queries: Query vectors as numpy array (query_num, dim)
        K: Number of nearest neighbors
        search_threads: Number of search threads (for logging)
        data_path: Path to data (for logging)
        
    Returns:
        Tuple of (hit_results, latencies_ms, query_result_tags, metrics_dict)
        where metrics_dict contains hit_ratio and latency metrics
    """
    query_num = queries.shape[0]
    query_dim = queries.shape[1]
    
    query_result_tags = np.zeros((query_num, K), dtype=np.uint32)
    query_result_dists = np.zeros((query_num, K), dtype=np.float32)
    hit_results = [False] * query_num
    latencies_ms = [0.0] * query_num
    
    global_start = time.time()
    hit_count = 0
    
    # Search each query
    for i in range(query_num):
        start = time.time()
        query = queries[i].astype(np.float32)
        
        hit, tags, dists = qvcache.search(query, K)
        
        hit_results[i] = hit
        if hit:
            hit_count += 1
        
        # Copy results (handle case where tags/dists might be shorter than K)
        result_len = min(len(tags), K)
        query_result_tags[i, :result_len] = tags[:result_len]
        query_result_dists[i, :result_len] = dists[:result_len]
        
        # DEBUG: Log tag format for first few queries to diagnose issues
        # This helps identify if cache hits return tags in different format than backend misses
        if i < 3 and len(tags) > 0:
            tag_min, tag_max = int(tags[0]), int(tags[min(2, len(tags)-1)])
            # Tags should be in range [1, num_vectors] after QVCache processing
            # If we see tags in range [0, num_vectors-1], there's a format mismatch
            if tag_min == 0 or (tag_max < 100 and tag_min < 100):
                import warnings
                warnings.warn(f"Query {i} (hit={hit}): Suspicious tag range [{tag_min}, {tag_max}]. "
                            f"Expected range [1, ~1000000]. First 3 tags: {tags[:min(3, len(tags))]}")
        
        end = time.time()
        latencies_ms[i] = (end - start) * 1000.0  # Convert to ms
    
    # Calculate statistics
    total_hit_latency_ms = sum(lat for i, lat in enumerate(latencies_ms) if hit_results[i])
    actual_hit_count = sum(hit_results)
    avg_hit_latency_ms = total_hit_latency_ms / actual_hit_count if actual_hit_count > 0 else 0.0
    
    final_ratio = hit_count / query_num if query_num > 0 else 0.0
    
    global_end = time.time()
    total_time_ms = (global_end - global_start) * 1000.0
    total_time_sec = total_time_ms / 1000.0
    avg_latency_ms = sum(latencies_ms) / query_num if query_num > 0 else 0.0
    qps = query_num / total_time_sec if total_time_sec > 0 else 0.0
    qps_per_thread = qps / search_threads if search_threads > 0 else 0.0
    
    sorted_latencies = sorted(latencies_ms)
    get_percentile = lambda p: sorted_latencies[int(np.ceil(p * query_num)) - 1] if query_num > 0 else 0.0
    p50 = get_percentile(0.50) if query_num > 0 else 0.0
    p90 = get_percentile(0.90) if query_num > 0 else 0.0
    p95 = get_percentile(0.95) if query_num > 0 else 0.0
    p99 = get_percentile(0.99) if query_num > 0 else 0.0
    
    # Get memory stats from qvcache
    num_mini_indexes = qvcache.get_number_of_mini_indexes()
    mini_index_counts = {}
    for i in range(num_mini_indexes):
        mini_index_counts[f"index_{i}_vectors"] = qvcache.get_index_vector_count(i)
    
    # Build metrics dictionary
    metrics = {
        "hit_ratio": final_ratio,
        "hits": hit_count,
        "total_queries": query_num,
        "threads": search_threads,
        "avg_latency_ms": avg_latency_ms,
        "avg_hit_latency_ms": avg_hit_latency_ms,
        "qps": qps,
        "qps_per_thread": qps_per_thread,
        "memory_active_vectors": qvcache.get_number_of_vectors_in_memory_index(),
        "memory_max_points": qvcache.get_number_of_max_points_in_memory_index(),
        "pca_active_regions": qvcache.get_number_of_active_pca_regions(),
        **mini_index_counts,
        "tail_latency_ms": {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99
        }
    }
    
    return hit_results, latencies_ms, query_result_tags, metrics


def calculate_recall(K: int, groundtruth_ids: np.ndarray, query_result_tags: np.ndarray,
                     query_num: int, groundtruth_dim: int) -> dict:
    """Calculate recall metric. Returns metrics dict instead of printing."""
    # Invalid ID marker (max uint32) used for padding
    INVALID_ID = np.iinfo(np.uint32).max
    
    total_recall = 0.0
    recall_by_query = []
    
    for i in range(query_num):
        groundtruth_set = set(groundtruth_ids[i, :K])
        # Filter out invalid IDs (padded results) before calculating recall
        valid_tags = query_result_tags[i, :K]
        valid_mask = valid_tags != INVALID_ID
        valid_tags_filtered = valid_tags[valid_mask]
        
        if len(valid_tags_filtered) > 0:
            # C++ version subtracts 1 from calculated tags to match groundtruth format
            # QVCache returns tags with +1 offset (1-1000000), so we subtract 1 to get 0-999999
            # CRITICAL: Validate tag range before subtracting to detect format issues
            calculated_tags = valid_tags_filtered - 1
            # Validate: tags should be in valid range [0, num_vectors-1] after subtraction
            # If we get negative values, there's a tag format issue (tags might already be 0-based)
            if np.any(calculated_tags < 0):
                # Tag format issue: tags are already 0-based, don't subtract
                # This can happen if cache hits return tags differently than backend misses
                calculated_tags = valid_tags_filtered
                if i == 0:  # Only log once to avoid spam
                    import warnings
                    warnings.warn(f"Tag format issue: tags appear to be 0-based already. "
                                f"First few tags: {valid_tags_filtered[:min(5, len(valid_tags_filtered))]}")
            
            calculated_set = set(calculated_tags)
            matching = len(groundtruth_set & calculated_set)
            # Calculate recall based on valid results, but normalize by K
            recall = matching / K
        else:
            # No valid results
            recall = 0.0
        
        recall_by_query.append(recall)
        total_recall += recall
    
    avg_recall = total_recall / query_num if query_num > 0 else 0.0
    
    # Additional diagnostics: check for queries with very low recall
    low_recall_count = sum(1 for r in recall_by_query if r < 0.5)
    very_low_recall_count = sum(1 for r in recall_by_query if r < 0.1)
    
    return {
        "recall_all": avg_recall,
        "K": K,
        "low_recall_queries": low_recall_count,  # Queries with recall < 0.5
        "very_low_recall_queries": very_low_recall_count  # Queries with recall < 0.1
    }


def calculate_hit_recall(K: int, groundtruth_ids: np.ndarray, query_result_tags: np.ndarray,
                        hit_results: List[bool], query_num: int, groundtruth_dim: int) -> dict:
    """Calculate recall for cache hits only. Returns metrics dict instead of printing."""
    # Invalid ID marker (max uint32) used for padding
    INVALID_ID = np.iinfo(np.uint32).max
    
    total_recall = 0.0
    hit_count = 0
    for i in range(query_num):
        if hit_results[i]:
            groundtruth_set = set(groundtruth_ids[i, :K])
            # Filter out invalid IDs (padded results) before calculating recall
            valid_tags = query_result_tags[i, :K]
            valid_mask = valid_tags != INVALID_ID
            valid_tags_filtered = valid_tags[valid_mask]
            
            if len(valid_tags_filtered) > 0:
                # C++ version subtracts 1 from calculated tags to match groundtruth format
                calculated_set = set(valid_tags_filtered - 1)
                matching = len(groundtruth_set & calculated_set)
                # Calculate recall based on valid results, but normalize by K
                recall = matching / K
            else:
                # No valid results
                recall = 0.0
            
            total_recall += recall
            hit_count += 1
    
    if hit_count > 0:
        avg_recall = total_recall / hit_count
        return {
            "recall_cache_hits": avg_recall,
            "cache_hit_count": hit_count
        }
    else:
        return {
            "recall_cache_hits": None,
            "cache_hit_count": 0
        }


def backend_search(
    backend,
    queries: np.ndarray,
    K: int,
    search_threads: int
) -> Tuple[np.ndarray, dict]:
    """
    Perform backend-only search on queries (without QVCache).
    
    Args:
        backend: Backend instance (FaissBackend, QdrantBackend, etc.)
        queries: Query vectors as numpy array (query_num, dim)
        K: Number of nearest neighbors
        search_threads: Number of search threads (for logging)
        
    Returns:
        Tuple of (query_result_tags, metrics_dict)
        where metrics_dict contains latency metrics (no hit_ratio)
    """
    query_num = queries.shape[0]
    query_dim = queries.shape[1]
    
    query_result_tags = np.zeros((query_num, K), dtype=np.uint32)
    latencies_ms = [0.0] * query_num
    
    global_start = time.time()
    
    # Search each query
    for i in range(query_num):
        start = time.time()
        query = queries[i].astype(np.float32)
        
        tags, dists = backend.search(query, K)
        
        # Copy results (handle case where tags/dists might be shorter than K)
        result_len = min(len(tags), K)
        query_result_tags[i, :result_len] = tags[:result_len]
        
        end = time.time()
        latencies_ms[i] = (end - start) * 1000.0  # Convert to ms
    
    global_end = time.time()
    total_time_ms = (global_end - global_start) * 1000.0
    total_time_sec = total_time_ms / 1000.0
    avg_latency_ms = sum(latencies_ms) / query_num if query_num > 0 else 0.0
    qps = query_num / total_time_sec if total_time_sec > 0 else 0.0
    qps_per_thread = qps / search_threads if search_threads > 0 else 0.0
    
    sorted_latencies = sorted(latencies_ms)
    get_percentile = lambda p: sorted_latencies[int(np.ceil(p * query_num)) - 1] if query_num > 0 else 0.0
    p50 = get_percentile(0.50) if query_num > 0 else 0.0
    p90 = get_percentile(0.90) if query_num > 0 else 0.0
    p95 = get_percentile(0.95) if query_num > 0 else 0.0
    p99 = get_percentile(0.99) if query_num > 0 else 0.0
    
    # Build metrics dictionary (backend-only, no QVCache metrics)
    metrics = {
        "total_queries": query_num,
        "threads": search_threads,
        "avg_latency_ms": avg_latency_ms,
        "qps": qps,
        "qps_per_thread": qps_per_thread,
        "tail_latency_ms": {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99
        }
    }
    
    return query_result_tags, metrics


def calculate_backend_recall(K: int, groundtruth_ids: np.ndarray, query_result_tags: np.ndarray,
                              query_num: int, groundtruth_dim: int) -> dict:
    """Calculate recall metric for backend-only results. Backends return 0-indexed tags."""
    # Invalid ID marker (max uint32) used for padding
    INVALID_ID = np.iinfo(np.uint32).max
    
    total_recall = 0.0
    recall_by_query = []
    
    for i in range(query_num):
        groundtruth_set = set(groundtruth_ids[i, :K])
        # Filter out invalid IDs (padded results) before calculating recall
        valid_tags = query_result_tags[i, :K]
        valid_mask = valid_tags != INVALID_ID
        valid_tags_filtered = valid_tags[valid_mask]
        
        if len(valid_tags_filtered) > 0:
            # Backends return 0-indexed tags directly, so use them as-is
            calculated_set = set(valid_tags_filtered)
            matching = len(groundtruth_set & calculated_set)
            # Calculate recall based on valid results, but normalize by K
            recall = matching / K
        else:
            # No valid results
            recall = 0.0
        
        recall_by_query.append(recall)
        total_recall += recall
    
    avg_recall = total_recall / query_num if query_num > 0 else 0.0
    
    # Additional diagnostics: check for queries with very low recall
    low_recall_count = sum(1 for r in recall_by_query if r < 0.5)
    very_low_recall_count = sum(1 for r in recall_by_query if r < 0.1)
    
    return {
        "recall_all": avg_recall,
        "K": K,
        "low_recall_queries": low_recall_count,  # Queries with recall < 0.5
        "very_low_recall_queries": very_low_recall_count  # Queries with recall < 0.1
    }


def log_backend_split_metrics(metrics: dict, recall_all: dict, split_idx: int = None):
    """
    Log all metrics for a split in a single structured JSON line (backend-only).
    
    Args:
        metrics: Dictionary from backend_search containing latency metrics
        recall_all: Dictionary from calculate_backend_recall containing recall
        split_idx: Optional split index to include in the log
    """
    combined = {
        "event": "split_metrics",
        **metrics,
        **recall_all
    }
    if split_idx is not None:
        combined["split_idx"] = split_idx
    print(json.dumps(combined))


def log_split_metrics(metrics: dict, recall_all: dict, recall_hits: dict, split_idx: int = None):
    """
    Log all metrics for a split in a single structured JSON line.
    
    Args:
        metrics: Dictionary from hybrid_search containing hit_ratio and latency metrics
        recall_all: Dictionary from calculate_recall containing recall for all queries
        recall_hits: Dictionary from calculate_hit_recall containing recall for cache hits
        split_idx: Optional split index to include in the log
    """
    combined = {
        "event": "split_metrics",
        **metrics,
        **recall_all,
        **recall_hits
    }
    if split_idx is not None:
        combined["split_idx"] = split_idx
    print(json.dumps(combined))


def log_window_metrics(metrics: dict, recall_all: dict, recall_hits: dict = None, window_idx: int = None, repeat_idx: int = None):
    """
    Log all metrics for a window in a single structured JSON line.
    
    Args:
        metrics: Dictionary from hybrid_search or backend_search containing metrics
        recall_all: Dictionary from calculate_recall or calculate_backend_recall containing recall for all queries
        recall_hits: Optional dictionary from calculate_hit_recall containing recall for cache hits (for QVCache)
        window_idx: Optional window index to include in the log
        repeat_idx: Optional repeat index to include in the log
    """
    combined = {
        "event": "window_metrics",
        **metrics,
        **recall_all
    }
    if recall_hits is not None:
        combined.update(recall_hits)
    if window_idx is not None:
        combined["window_idx"] = window_idx
    if repeat_idx is not None:
        combined["repeat_idx"] = repeat_idx
    print(json.dumps(combined))


def log_backend_window_metrics(metrics: dict, recall_all: dict, window_idx: int = None, repeat_idx: int = None):
    """
    Log all metrics for a window in a single structured JSON line (backend-only).
    
    Args:
        metrics: Dictionary from backend_search containing latency metrics
        recall_all: Dictionary from calculate_backend_recall containing recall
        window_idx: Optional window index to include in the log
        repeat_idx: Optional repeat index to include in the log
    """
    combined = {
        "event": "window_metrics",
        **metrics,
        **recall_all
    }
    if window_idx is not None:
        combined["window_idx"] = window_idx
    if repeat_idx is not None:
        combined["repeat_idx"] = repeat_idx
    print(json.dumps(combined))

