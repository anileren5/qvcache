#!/usr/bin/env python3
"""
Windowed backend-only benchmark using FAISS backend (without QVCache)

This script benchmarks the FAISS backend directly, processing queries in windows.
"""

import argparse
import numpy as np
import json
import sys
import random

try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.faiss_backend import FaissBackend
from benchmarks.utils import backend_search, calculate_backend_recall, log_backend_window_metrics


def experiment_benchmark(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    K: int,
    search_threads: int,
    n_splits: int,
    n_split_repeat: int,
    backend: FaissBackend,
    index_path: str,
    window_size: int,
    n_repeat: int,
    stride: int,
    n_round: int
):
    """Run the windowed backend-only benchmark experiment with FAISS backend."""
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
                query_result_tags, metrics = backend_search(
                    backend,
                    shuffled_queries,
                    K,
                    search_threads
                )
                
                # Calculate recall using shuffled groundtruth
                recall_all = calculate_backend_recall(
                    K, shuffled_groundtruth, 
                    query_result_tags, total_repeat_queries, groundtruth_dim
                )
                
                log_backend_window_metrics(metrics, recall_all, window_idx=window_idx, repeat_idx=repeat_idx)
            
            print(json.dumps({
                "event": "window_end",
                "window_idx": window_idx
            }))
            window_idx += 1
        
        print(json.dumps({
            "event": "round_end",
            "round": round_num
        }))


def main():
    parser = argparse.ArgumentParser(description="Windowed backend-only benchmark experiment with FAISS backend")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query data file")
    parser.add_argument("--groundtruth_path", type=str, required=True, help="Path to groundtruth file")
    parser.add_argument("--index_path", type=str, default="./faiss_index.bin",
                       help="Path to FAISS index (default: ./faiss_index.bin)")
    
    # Search parameters
    parser.add_argument("--K", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--search_threads", type=int, default=24, help="Search threads")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_split_repeat", type=int, required=True, help="Number of repeats per split pattern")
    parser.add_argument("--data_type", type=str, default="float", help="Data type")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine", "inner_product"],
                       help="Distance metric: l2, cosine, or inner_product")
    
    # Window parameters
    parser.add_argument("--window_size", type=int, required=True, help="Window size (number of splits per window)")
    parser.add_argument("--n_repeat", type=int, required=True, help="N_repeat (number of copies per split in window)")
    parser.add_argument("--stride", type=int, required=True, help="Stride (step size for window advancement)")
    parser.add_argument("--n_round", type=int, default=1, help="Number of times to cycle windows over splits (wrapping)")
    
    args = parser.parse_args()
    
    # Print parameters
    params = {
        "event": "params",
        "backend": "FAISS",
        "index_path": args.index_path,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "K": args.K,
        "search_threads": args.search_threads,
        "n_splits": args.n_splits,
        "n_split_repeat": args.n_split_repeat,
        "metric": args.metric,
        "window_size": args.window_size,
        "n_repeat": args.n_repeat,
        "stride": args.stride,
        "n_round": args.n_round
    }
    print(json.dumps(params))
    
    # Get vector dimension from data file
    with open(args.data_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dimension = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    
    # Create FAISS backend (assumes index already exists)
    print(f"\nLoading FAISS index from {args.index_path}...", file=sys.stderr)
    backend = FaissBackend(
        index_path=args.index_path,
        dimension=dimension,
        data_path=args.data_path,  # Needed for fetch_vectors_by_ids
        recreate_index=False
    )
    
    # Run experiment
    experiment_benchmark(
        args.data_path, args.query_path, args.groundtruth_path,
        args.K, args.search_threads,
        args.n_splits, args.n_split_repeat,
        backend, args.index_path,
        args.window_size, args.n_repeat, args.stride, args.n_round
    )


if __name__ == "__main__":
    main()

