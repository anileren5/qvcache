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
import os
import sys

import numpy as np

_DTYPE_MAP = {
    "float": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def read_bin_vectors(input_file, dtype="float"):
    """Read vectors from a .bin file (DiskANN format)."""
    np_dtype = _DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")

    with open(input_file, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"Invalid bin header in {input_file}")
        num_vectors = int(header[0])
        dim = int(header[1])
        vectors = np.fromfile(f, dtype=np_dtype, count=num_vectors * dim)
        if vectors.size != num_vectors * dim:
            raise ValueError(
                f"Truncated bin file {input_file}: expected {num_vectors * dim} values, got {vectors.size}"
            )
        return vectors.reshape(num_vectors, dim), num_vectors, dim


def write_bin_vectors(vectors, output_file, dtype="float"):
    """Write vectors to a .bin file (DiskANN format)."""
    np_dtype = _DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")

    num_vectors, dim = vectors.shape
    out = np.ascontiguousarray(vectors, dtype=np_dtype)
    with open(output_file, "wb") as f:
        np.array([num_vectors, dim], dtype=np.uint32).tofile(f)
        out.tofile(f)


def _cast_interpolated(interpolated, dtype):
    if dtype == "float":
        return interpolated.astype(np.float32, copy=False)
    if dtype == "int8":
        return np.clip(interpolated, -128, 127).astype(np.int8)
    return np.clip(interpolated, 0, 255).astype(np.uint8)


def generate_noisy_queries(
    dataset_name,
    n_split,
    n_split_repeat,
    noise_ratio,
    random_seed=42,
    data_dir="data",
    dtype="float",
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

    if n_split < 1:
        raise ValueError(f"n_split must be at least 1, got {n_split}")

    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be float, int8, or uint8, got {dtype}")

    np.random.seed(random_seed)

    dataset_dir = os.path.join(data_dir, dataset_name)
    query_file = os.path.join(dataset_dir, f"{dataset_name}_query.bin")

    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")

    print(f"Reading queries from {query_file} (dtype: {dtype})...")
    queries, num_queries, dim = read_bin_vectors(query_file, dtype)
    print(f"Loaded {num_queries} queries of dimension {dim}")

    if n_split > num_queries:
        raise ValueError(f"n_split ({n_split}) cannot exceed number of queries ({num_queries})")

    split_sizes = [
        split.shape[0] for split in np.array_split(np.arange(num_queries), n_split)
    ]
    total_queries = num_queries * n_split_repeat
    result = np.empty((total_queries, dim), dtype=_DTYPE_MAP[dtype])

    queries_float = queries.astype(np.float32, copy=False)
    keep_ratio = np.float32(1.0 - noise_ratio)
    noise = np.float32(noise_ratio)

    print(f"Split queries into {n_split} chunks")

    write_offset = 0
    query_offset = 0
    for split_idx, split_size in enumerate(split_sizes):
        split_queries = queries[query_offset : query_offset + split_size]
        split_float = queries_float[query_offset : query_offset + split_size]
        print(f"Processing split {split_idx + 1}/{n_split} ({split_size} queries)...")

        result[write_offset : write_offset + split_size] = split_queries
        write_offset += split_size

        for copy_idx in range(1, n_split_repeat):
            # Same RNG consumption as the old per-query np.random.randint loop.
            random_idx = np.random.randint(0, num_queries, size=split_size)
            interpolated = keep_ratio * split_float + noise * queries_float[random_idx]
            result[write_offset : write_offset + split_size] = _cast_interpolated(
                interpolated, dtype
            )
            write_offset += split_size

            if (copy_idx + 1) % 10 == 0 or copy_idx == n_split_repeat - 1:
                print(f"  Generated {copy_idx + 1}/{n_split_repeat - 1} noisy copies")

        query_offset += split_size

    print(f"\nTotal queries generated: {total_queries}")
    print(f"  ({n_split} splits × {n_split_repeat} copies = {n_split * n_split_repeat} total)")

    noise_str = f"{noise_ratio:.10f}".rstrip("0").rstrip(".")
    output_filename = (
        f"{dataset_name}_query_nsplit-{n_split}_"
        f"nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    output_file = os.path.join(dataset_dir, output_filename)

    print(f"\nWriting queries to {output_file} (dtype: {dtype})...")
    write_bin_vectors(result, output_file, dtype)

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
        help="Dataset name (used to find query file: data/{dataset}/{dataset}_query.bin)",
    )
    parser.add_argument(
        "--n_split",
        type=int,
        required=True,
        help="Number of splits to create",
    )
    parser.add_argument(
        "--n_split_repeat",
        type=int,
        required=True,
        help="Number of copies per split (including original)",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Noise ratio for interpolation (0-1)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for data files (default: data)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float",
        choices=["float", "int8", "uint8"],
        help="Data type - float, int8, or uint8 (default: float)",
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
            dtype=args.dtype,
        )
        print(f"\n✓ Success! Generated noisy queries: {output_file}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
