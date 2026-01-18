#!/usr/bin/env python3
"""
Helper script to build a Qdrant index from binary data files.

This script loads vectors from a binary file (DiskANN format) and uploads them to Qdrant.
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from backends.qdrant_backend import QdrantBackend
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def main():
    parser = argparse.ArgumentParser(description="Build Qdrant index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--collection_name", type=str, default="vectors",
                       help="Name of the Qdrant collection (default: vectors)")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                       help="URL of the Qdrant service (default: http://localhost:6333)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate collection even if it exists")
    parser.add_argument("--dimension", type=int, default=None,
                       help="Vector dimension (will be read from file if not provided)")
    
    args = parser.parse_args()
    
    # Read metadata to get dimension
    if args.dimension is None:
        with open(args.data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dimension = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        print(f"Detected {num_vectors} vectors of dimension {dimension} in {args.data_path}")
    else:
        dimension = args.dimension
    
    # Initialize Qdrant backend (this will create collection and load data)
    print(f"Connecting to Qdrant at {args.qdrant_url}...")
    backend = QdrantBackend(
        collection_name=args.collection_name,
        dimension=dimension,
        qdrant_url=args.qdrant_url,
        data_path=args.data_path,
        recreate_collection=args.recreate
    )
    
    # Verify the collection
    client = QdrantClient(url=args.qdrant_url)
    collection_info = client.get_collection(args.collection_name)
    print(f"\nIndex built successfully!")
    print(f"Collection '{args.collection_name}' contains {collection_info.points_count} vectors")
    print(f"Vector dimension: {collection_info.config.params.vectors.size}")
    print(f"Distance metric: {collection_info.config.params.vectors.distance}")


if __name__ == "__main__":
    main()

