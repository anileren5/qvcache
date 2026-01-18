#!/usr/bin/env python3
"""
Python version of qvcache_benchmark.cpp using Pinecone backend

This script implements the windowed QVCache benchmark logic from the C++ version
but uses a Python-implemented Pinecone backend.
"""

import argparse
import numpy as np
import json
import sys
import random
import os

# Import the compiled qvcache module
try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.pinecone_backend import PineconeBackend
from benchmarks.utils import hybrid_search, calculate_recall, calculate_hit_recall, log_window_metrics


def experiment_benchmark(data_path: str,
    query_path: str,
    groundtruth_path: str,
    pca_prefix: str,
    R: int,
    memory_L: int,
    K: int,
    B: int,
    M: int,
    alpha: float,
    build_threads: int,
    search_threads: int,
    beamwidth: int,
    use_reconstructed_vectors: int,
    p: float,
    deviation_factor: float,
    memory_index_max_points: int,
    use_regional_theta: bool,
    pca_dim: int,
    buckets_per_dim: int,
    max_regions: int,
    n_splits: int,
    n_split_repeat: int,
    n_async_insert_threads: int,
    lazy_theta_updates: bool,
    number_of_mini_indexes: int,
    search_mini_indexes_in_parallel: bool,
    max_search_threads: int,
    search_strategy: str,
    backend: PineconeBackend,
    index_name: str,
    window_size: int,
    n_repeat: int,
    stride: int,
    n_round: int,
    api_key: str = None,
    environment: str = None
):
    """Run the windowed QVCache benchmark experiment with Pinecone backend."""
    # Create QVCache with Pinecone backend
    qvcache = qvc.QVCache(
        data_path=data_path,
        pca_prefix=pca_prefix,
        R=R,
        memory_L=memory_L,
        B=B,
        M=M,
        alpha=alpha,
        build_threads=build_threads,
        search_threads=search_threads,
        use_reconstructed_vectors=bool(use_reconstructed_vectors),
        p=p,
        deviation_factor=deviation_factor,
        memory_index_max_points=memory_index_max_points,
        beamwidth=beamwidth,
        use_regional_theta=use_regional_theta,
        pca_dim=pca_dim,
        buckets_per_dim=buckets_per_dim,
        max_regions=max_regions,
        n_async_insert_threads=n_async_insert_threads,
        lazy_theta_updates=lazy_theta_updates,
        number_of_mini_indexes=number_of_mini_indexes,
        search_mini_indexes_in_parallel=search_mini_indexes_in_parallel,
        max_search_threads=max_search_threads,
        backend=backend
    )
    
    # Set search strategy
    if search_strategy == "SEQUENTIAL_LRU_STOP_FIRST_HIT":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_STOP_FIRST_HIT)
    elif search_strategy == "SEQUENTIAL_LRU_ADAPTIVE":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
        qvcache.enable_adaptive_strategy(True)
        qvcache.set_hit_ratio_window_size(100)
        qvcache.set_hit_ratio_threshold(0.90)
    elif search_strategy == "SEQUENTIAL_ALL":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_ALL)
    elif search_strategy == "PARALLEL":
        qvcache.set_search_strategy(qvc.SearchStrategy.PARALLEL)
    else:
        print(f"Warning: Unknown search strategy '{search_strategy}', using default", file=sys.stderr)
    
    # Load ground truth
    groundtruth_ids, groundtruth_dists = qvc.load_ground_truth_data(groundtruth_path)
    n_groundtruth, groundtruth_dim = groundtruth_ids.shape

    # Load queries
    queries, query_dim, query_aligned_dim = qvc.load_aligned_binary_data(query_path)
    query_num = queries.shape[0]
    
    # Ensure queries are float32
    queries = queries.astype(np.float32)
    
    # Query file structure: all copies of split 0, then all copies of split 1, etc.
    # Each split has n_split_repeat copies, and each copy has queries_per_original_split queries
    queries_per_original_split = query_num // (n_splits * n_split_repeat)
    
    # Validate window parameters
    # Ensure window parameters are integers
    window_size = int(window_size)
    n_repeat = int(n_repeat)
    stride = int(stride)
    n_round = int(n_round)
    min_split_repeat = (window_size // stride) * n_repeat * n_round
    if n_split_repeat < min_split_repeat:
        print(f"Error: n_split_repeat ({n_split_repeat}) must be >= (window_size / stride) * n_repeat * n_round = {min_split_repeat}", file=sys.stderr)
        sys.exit(1)
    
    # Random number generator for shuffling
    rng = random.Random()
    
    # Process windows in rounds; each round ends when the last split of a window reaches the last global split.
    window_idx = 0
    for round_num in range(n_round):
        print(json.dumps({
            "event": "round_start",
            "round": round_num
        }))
        
        for window_start in range(0, n_splits - window_size + 1, stride):
            window_end = window_start + window_size - 1
            print(json.dumps({
                "event": "window_start",
                "window_idx": window_idx,
                "window_start_split": window_start,
                "window_end_split": window_end
            }))
            
            # Process each repeat separately within this window
            for repeat_idx in range(min(n_repeat, n_split_repeat)):
                # Collect queries from this repeat across all splits in the window
                query_infos = []
                
                for offset in range(window_size):
                    split_idx = window_start + offset
                    split_offset = split_idx * n_split_repeat * queries_per_original_split
                    copy_offset = repeat_idx * queries_per_original_split
                    query_start = split_offset + copy_offset
                    query_end = min(query_start + queries_per_original_split, query_num)
                    
                    if query_start < query_end:
                        query_infos.append({
                            "query_offset": query_start,
                            "query_size": query_end - query_start,
                            "gt_offset": split_offset + copy_offset
                        })
                
                # Shuffle query_infos to randomize order
                rng.shuffle(query_infos)
                
                # Collect all queries and groundtruth in shuffled order
                total_repeat_queries = sum(info["query_size"] for info in query_infos)
                
                if total_repeat_queries == 0:
                    continue
                
                # Allocate buffers for shuffled queries and groundtruth
                shuffled_queries = np.zeros((total_repeat_queries, query_aligned_dim), dtype=np.float32)
                shuffled_groundtruth = np.zeros((total_repeat_queries, groundtruth_dim), dtype=np.uint32)
                
                current_idx = 0
                for info in query_infos:
                    # Copy queries
                    shuffled_queries[current_idx:current_idx + info["query_size"]] = queries[info["query_offset"]:info["query_offset"] + info["query_size"]]
                    # Copy groundtruth
                    shuffled_groundtruth[current_idx:current_idx + info["query_size"]] = groundtruth_ids[info["gt_offset"]:info["gt_offset"] + info["query_size"]]
                    current_idx += info["query_size"]
                
                # Perform search using backend
                hit_results, _, query_result_tags, metrics = hybrid_search(
                    qvcache,
                    shuffled_queries,
                    K,
                    search_threads,
                    data_path
                )
                
                # Calculate recall using shuffled groundtruth
                recall_all = calculate_recall(
                    K, shuffled_groundtruth, 
                    query_result_tags, total_repeat_queries, groundtruth_dim
                )
                recall_hits = calculate_hit_recall(
                    K, shuffled_groundtruth,
                    query_result_tags, hit_results, total_repeat_queries, groundtruth_dim
                )
                
                log_window_metrics(metrics, recall_all, recall_hits, window_idx=window_idx, repeat_idx=repeat_idx)
            
            print(json.dumps({
                "event": "window_end",
                "window_idx": window_idx
            }))
            window_idx += 1
        
        print(json.dumps({
            "event": "round_end",
            "round": round_num
        }))

    
    # Give async insert threads time to complete before cleanup
    # QVCache uses async insert threads that might still be processing
    import time
    time.sleep(2)  # Wait 2 seconds for async operations to complete
    
    # Explicitly delete qvcache to trigger cleanup
    del qvcache


def main():
    parser = argparse.ArgumentParser(description="QVCache benchmark experiment with Pinecone backend")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query data file")
    parser.add_argument("--groundtruth_path", type=str, required=True, help="Path to groundtruth file")
    parser.add_argument("--pca_prefix", type=str, required=True, help="PCA index prefix")
    parser.add_argument("--index_name", type=str, default="vectors",
                       help="Pinecone index name (default: vectors). "
                            "For cloud: Check your Pinecone dashboard for the exact index name")
    parser.add_argument("--api_key", type=str, default=None,
                       help="Pinecone API key (default: from PINECONE_API_KEY env var). "
                            "For cloud: Your API key (starts with 'pcsk_'). "
                            "For local: Use 'pclocal' or 'local'")
    parser.add_argument("--environment", type=str, default=None,
                       help="Pinecone environment/region. "
                            "For cloud: REQUIRED! (e.g., 'us-east-1', 'us-west-2', 'eu-west-1'). "
                            "For local: Not needed")
    parser.add_argument("--host", type=str, default=None,
                       help="Pinecone host. "
                            "For cloud: Not needed (leave as None). "
                            "For local Docker: Use service name like 'pinecone'")
    
    # QVCache parameters
    parser.add_argument("--R", type=int, default=64, help="R parameter")
    parser.add_argument("--memory_L", type=int, default=128, help="Memory L parameter")
    parser.add_argument("--K", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--B", type=int, default=8, help="B parameter")
    parser.add_argument("--M", type=int, default=8, help="M parameter")
    parser.add_argument("--alpha", type=float, default=1.2, help="Alpha parameter")
    parser.add_argument("--build_threads", type=int, default=8, help="Build threads")
    parser.add_argument("--search_threads", type=int, default=24, help="Search threads")
    parser.add_argument("--beamwidth", type=int, default=2, help="Beamwidth")
    parser.add_argument("--use_reconstructed_vectors", type=int, default=0, help="Use reconstructed vectors")
    parser.add_argument("--p", type=float, default=0.9, help="P parameter")
    parser.add_argument("--deviation_factor", type=float, default=0.025, help="Deviation factor")
    parser.add_argument("--memory_index_max_points", type=int, default=200000, help="Max points in memory index")
    parser.add_argument("--use_regional_theta", type=bool, default=True, help="Use regional theta")
    parser.add_argument("--pca_dim", type=int, default=16, help="PCA dimension")
    parser.add_argument("--buckets_per_dim", type=int, default=8, help="Buckets per dimension")
    parser.add_argument("--max_regions", type=int, default=18446744073709551615, help="Maximum number of regions for regional theta (default: unlimited)")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_split_repeat", type=int, required=True, help="Number of repeats per split pattern")
    parser.add_argument("--n_async_insert_threads", type=int, default=16, help="Async insert threads")
    parser.add_argument("--lazy_theta_updates", type=bool, default=True, help="Lazy theta updates")
    parser.add_argument("--number_of_mini_indexes", type=int, default=4, help="Number of mini indexes")
    parser.add_argument("--search_mini_indexes_in_parallel", type=bool, default=False, help="Search mini indexes in parallel")
    parser.add_argument("--max_search_threads", type=int, default=32, help="Max search threads")
    parser.add_argument("--search_strategy", type=str, default="SEQUENTIAL_LRU_ADAPTIVE",
                       choices=["SEQUENTIAL_LRU_STOP_FIRST_HIT", "SEQUENTIAL_LRU_ADAPTIVE",
                               "SEQUENTIAL_ALL", "PARALLEL"],
                       help="Search strategy")
    parser.add_argument("--data_type", type=str, default="float", help="Data type")
    
    # Window parameters
    parser.add_argument("--window_size", type=int, required=True, help="Window size (number of splits per window)")
    parser.add_argument("--n_repeat", type=int, required=True, help="N_repeat (number of copies per split in window)")
    parser.add_argument("--stride", type=int, required=True, help="Stride (step size for window advancement)")
    parser.add_argument("--n_round", type=int, default=1, help="Number of times to cycle windows over splits (wrapping)")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("PINECONE_API_KEY", "local")  # "local" will be converted to "pclocal" in backend
    
    # Print parameters
    params = {
        "event": "params",
        "backend": "Pinecone",
        "index_name": args.index_name,
        "api_key": api_key[:10] + "..." if len(api_key) > 10 else "***",
        "environment": args.environment,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "pca_prefix": args.pca_prefix,
        "R": args.R,
        "memory_L": args.memory_L,
        "K": args.K,
        "B": args.B,
        "M": args.M,
        "build_threads": args.build_threads,
        "search_threads": args.search_threads,
        "alpha": args.alpha,
        "use_reconstructed_vectors": args.use_reconstructed_vectors,
        "beamwidth": args.beamwidth,
        "p": args.p,
        "deviation_factor": args.deviation_factor,
        "use_regional_theta": args.use_regional_theta,
        "pca_dim": args.pca_dim,
        "buckets_per_dim": args.buckets_per_dim,
        "max_regions": args.max_regions,
        "memory_index_max_points": args.memory_index_max_points,
        "n_splits": args.n_splits,
        "n_split_repeat": args.n_split_repeat,
        "n_async_insert_threads": args.n_async_insert_threads,
        "lazy_theta_updates": args.lazy_theta_updates,
        "number_of_mini_indexes": args.number_of_mini_indexes,
        "search_mini_indexes_in_parallel": args.search_mini_indexes_in_parallel,
        "max_search_threads": args.max_search_threads,
        "search_strategy": args.search_strategy,
        "window_size": args.window_size,
        "n_repeat": args.n_repeat,
        "stride": args.stride,
        "n_round": args.n_round
    }
    print(json.dumps(params))
    
    # Get vector dimension from data file
    with open(args.data_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dimension = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])  # Convert to Python int
    
    # Create Pinecone backend (assumes index already exists)
    print(f"\nConnecting to Pinecone...", file=sys.stderr)
    backend = PineconeBackend(
        index_name=args.index_name,
        dimension=dimension,
        api_key=api_key,
        environment=args.environment,
        host=args.host,
        data_path=None,  # Don't load data here, should already be indexed
        recreate_index=False
    )
    
    # Run experiment
    experiment_benchmark(
        args.data_path, args.query_path, args.groundtruth_path, args.pca_prefix,
        args.R, args.memory_L, args.K, args.B, args.M, args.alpha,
        args.build_threads, args.search_threads, args.beamwidth,
        args.use_reconstructed_vectors, args.p, args.deviation_factor,
        args.memory_index_max_points,
        args.use_regional_theta, args.pca_dim, args.buckets_per_dim, args.max_regions,
        args.n_splits, args.n_split_repeat,
        args.n_async_insert_threads,
        args.lazy_theta_updates, args.number_of_mini_indexes,
        args.search_mini_indexes_in_parallel, args.max_search_threads,
        args.search_strategy, backend, args.index_name,
        args.window_size, args.n_repeat, args.stride, args.n_round,
        api_key, args.environment
    )


if __name__ == "__main__":
    main()

