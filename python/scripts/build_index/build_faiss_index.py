#!/usr/bin/env python3
"""
Helper script to build a FAISS index from binary data files.

This script loads vectors from a binary file (DiskANN format) and builds a FAISS index.
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from backends.faiss_backend import FaissBackend


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--index_path", type=str, default="./faiss_index.bin",
                       help="Path to save FAISS index (default: ./faiss_index.bin)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate index even if it exists")
    parser.add_argument("--dimension", type=int, default=None,
                       help="Vector dimension (will be read from file if not provided)")
    parser.add_argument("--index_type", type=str, default="HNSW",
                       choices=["HNSW", "Flat"],
                       help="Type of FAISS index: HNSW (approximate, faster) or Flat (exact, slower)")
    parser.add_argument("--hnsw_m", type=int, default=32,
                       help="HNSW parameter M (number of bi-directional links, default: 32)")
    parser.add_argument("--hnsw_ef_construction", type=int, default=200,
                       help="HNSW parameter ef_construction (default: 200)")
    
    args = parser.parse_args()
    
    # Read metadata to get dimension
    if args.dimension is None:
        with open(args.data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dimension = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        print(f"Detected {num_vectors} vectors of dimension {dimension} in {args.data_path}")
    else:
        dimension = args.dimension
    
    # Initialize FAISS backend (this will create index and load data)
    print(f"Building FAISS {args.index_type} index...")
    backend = FaissBackend(
        index_path=args.index_path,
        dimension=dimension,
        data_path=args.data_path,
        index_type=args.index_type,
        recreate_index=args.recreate,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction
    )
    
    # Verify the index
    print(f"\nIndex built successfully!")
    print(f"Index contains {backend.index.ntotal} vectors")
    print(f"Vector dimension: {backend.dim}")
    print(f"Index type: {args.index_type}")


if __name__ == "__main__":
    main()

