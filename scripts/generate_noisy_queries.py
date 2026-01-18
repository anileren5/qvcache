#!/usr/bin/env python3
"""
Generate noisy queries by splitting base queries and interpolating with random queries.

This script:
1. Loads base queries from a dataset
2. Splits queries into n_split chunks
3. For each split, generates n_split_repeat copies (including the original split)
4. For noisy copies, interpolates each query with a random query from all queries
5. Writes all splits to a single query bin file with parameters embedded in filename
"""

import argparse
import numpy as np
import struct
import os
import sys


def read_bin_vectors(input_file, dtype="float"):
    """Read vectors from a .bin file (DiskANN format)."""
    with open(input_file, "rb") as f:
        num_vectors = struct.unpack("I", f.read(4))[0]
        dim = struct.unpack("I", f.read(4))[0]
        total_values = num_vectors * dim
        
        if dtype == "float":
            vector_data = struct.unpack(f"{total_values}f", f.read(total_values * 4))
            vectors = np.array(vector_data, dtype=np.float32).reshape(num_vectors, dim)
        elif dtype == "int8":
            vector_data = struct.unpack(f"{total_values}b", f.read(total_values * 1))
            vectors = np.array(vector_data, dtype=np.int8).reshape(num_vectors, dim)
        elif dtype == "uint8":
            vector_data = struct.unpack(f"{total_values}B", f.read(total_values * 1))
            vectors = np.array(vector_data, dtype=np.uint8).reshape(num_vectors, dim)
        else:
            raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")
        
        return vectors, num_vectors, dim


def write_bin_vectors(vectors, output_file, dtype="float"):
    """Write vectors to a .bin file (DiskANN format)."""
    num_vectors, dim = vectors.shape
    with open(output_file, "wb") as f:
        f.write(struct.pack("I", num_vectors))
        f.write(struct.pack("I", dim))
        
        if dtype == "float":
            vectors_flat = vectors.astype(np.float32).flatten()
        elif dtype == "int8":
            vectors_flat = vectors.astype(np.int8).flatten()
        elif dtype == "uint8":
            vectors_flat = vectors.astype(np.uint8).flatten()
        else:
            raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")
        
        # Use tofile for efficient writing of large arrays
        vectors_flat.tofile(f)


def generate_noisy_queries(
    dataset_name,
    n_split,
    n_split_repeat,
    noise_ratio,
    random_seed=42,
    data_dir="data",
    dtype="float"
):
    """
    Generate noisy queries by splitting and interpolating.
    
    Args:
        dataset_name: Name of the dataset (used to find query file)
        n_split: Number of splits to create
        n_split_repeat: Number of copies per split (including original)
        noise_ratio: Noise ratio for interpolation (0-1)
        random_seed: Random seed for reproducibility
        data_dir: Base directory for data files
        dtype: Data type - float, int8, or uint8 (default: float)
    
    Returns:
        Path to the generated query file
    """
    if noise_ratio < 0 or noise_ratio > 1:
        raise ValueError(f"noise_ratio must be between 0 and 1, got {noise_ratio}")
    
    if n_split_repeat < 1:
        raise ValueError(f"n_split_repeat must be at least 1, got {n_split_repeat}")
    
    if dtype not in ["float", "int8", "uint8"]:
        raise ValueError(f"dtype must be float, int8, or uint8, got {dtype}")
    
    np.random.seed(random_seed)
    
    # Construct paths
    dataset_dir = os.path.join(data_dir, dataset_name)
    query_file = os.path.join(dataset_dir, f"{dataset_name}_query.bin")
    
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    # Read base queries
    print(f"Reading queries from {query_file} (dtype: {dtype})...")
    queries, num_queries, dim = read_bin_vectors(query_file, dtype)
    print(f"Loaded {num_queries} queries of dimension {dim}")
    
    # Split queries into n_split chunks
    split_arrays = np.array_split(queries, n_split, axis=0)
    print(f"Split queries into {n_split} chunks")
    
    # Generate copies for each split
    all_copies = []
    for split_idx, split_queries in enumerate(split_arrays):
        num_in_split = split_queries.shape[0]
        print(f"Processing split {split_idx + 1}/{n_split} ({num_in_split} queries)...")
        
        # First copy: original split (no noise)
        all_copies.append(split_queries)
        
        # Generate n_split_repeat - 1 noisy copies
        for copy_idx in range(1, n_split_repeat):
            # For each query in the split, interpolate with a random query from ALL queries
            # Convert to float for interpolation, then convert back to original dtype
            noisy_split = np.zeros_like(split_queries)
            split_queries_float = split_queries.astype(np.float32)
            
            for i in range(num_in_split):
                # Pick a random query from all queries
                random_idx = np.random.randint(0, num_queries)
                random_query_float = queries[random_idx].astype(np.float32)
                
                # Interpolate: (1-noise_ratio) * query + noise_ratio * random_query
                interpolated = (1.0 - noise_ratio) * split_queries_float[i] + noise_ratio * random_query_float
                
                # Convert back to original dtype
                if dtype == "float":
                    noisy_split[i] = interpolated.astype(np.float32)
                elif dtype == "int8":
                    noisy_split[i] = np.clip(interpolated, -128, 127).astype(np.int8)
                elif dtype == "uint8":
                    noisy_split[i] = np.clip(interpolated, 0, 255).astype(np.uint8)
            
            all_copies.append(noisy_split)
            
            if (copy_idx + 1) % 10 == 0 or copy_idx == n_split_repeat - 1:
                print(f"  Generated {copy_idx + 1}/{n_split_repeat - 1} noisy copies")
    
    # Concatenate all copies
    result = np.vstack(all_copies)
    total_queries = result.shape[0]
    print(f"\nTotal queries generated: {total_queries}")
    print(f"  ({n_split} splits × {n_split_repeat} copies = {n_split * n_split_repeat} total)")
    
    # Generate output filename with embedded parameters
    # Format noise_ratio to avoid unnecessary trailing zeros (e.g., 0.1 instead of 0.1000)
    noise_str = f"{noise_ratio:.10f}".rstrip('0').rstrip('.')
    output_filename = (
        f"{dataset_name}_query_nsplit-{n_split}_"
        f"nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    output_file = os.path.join(dataset_dir, output_filename)
    
    # Write to file
    print(f"\nWriting queries to {output_file} (dtype: {dtype})...")
    write_bin_vectors(result, output_file, dtype)
    
    # Verify file was written correctly
    file_size = os.path.getsize(output_file)
    print(f"File written successfully ({file_size / (1024**2):.2f} MB)")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy queries by splitting and interpolating"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (used to find query file: data/{dataset}/{dataset}_query.bin)"
    )
    parser.add_argument(
        "--n_split",
        type=int,
        required=True,
        help="Number of splits to create"
    )
    parser.add_argument(
        "--n_split_repeat",
        type=int,
        required=True,
        help="Number of copies per split (including original)"
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Noise ratio for interpolation (0-1)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for data files (default: data)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float",
        choices=["float", "int8", "uint8"],
        help="Data type - float, int8, or uint8 (default: float)"
    )
    
    args = parser.parse_args()
    
    try:
        output_file = generate_noisy_queries(
            dataset_name=args.dataset,
            n_split=args.n_split,
            n_split_repeat=args.n_split_repeat,
            noise_ratio=args.noise_ratio,
            random_seed=args.random_seed,
            data_dir=args.data_dir,
            dtype=args.dtype
        )
        print(f"\n✓ Success! Generated noisy queries: {output_file}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

