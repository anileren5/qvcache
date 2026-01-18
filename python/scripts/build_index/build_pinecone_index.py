#!/usr/bin/env python3
"""
Helper script to build a Pinecone index from binary data files.

This script loads vectors from a binary file (DiskANN format) and uploads them to Pinecone.
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from backends.pinecone_backend import PineconeBackend


def main():
    parser = argparse.ArgumentParser(description="Build Pinecone index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--index_name", type=str, default="vectors",
                       help="Name of the Pinecone index (default: vectors). "
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
                       help="Pinecone host (for local Docker, use service name like 'pinecone')")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate index even if it exists")
    parser.add_argument("--dimension", type=int, default=None,
                       help="Vector dimension (will be read from file if not provided)")
    
    args = parser.parse_args()
    
    # Read metadata to get dimension
    if args.dimension is None:
        with open(args.data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dimension = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])  # Convert to Python int
        print(f"Detected {num_vectors} vectors of dimension {dimension} in {args.data_path}")
    else:
        dimension = int(args.dimension)  # Ensure it's a Python int
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("PINECONE_API_KEY", "local")  # "local" will be converted to "pclocal" in backend
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Get file size for verification
    file_size = os.path.getsize(args.data_path)
    print(f"Data file size: {file_size:,} bytes")
    
    # Calculate expected vectors from file size
    # Format: 4 bytes (num_vectors) + 4 bytes (dim) + num_vectors * dim * 4 bytes
    expected_min_size = 8 + dimension * 4  # At least 1 vector
    print(f"Expected minimum file size for {dimension}D vectors: {expected_min_size:,} bytes")
    
    # Initialize Pinecone backend (this will create index and load data)
    print(f"\n{'='*60}")
    print(f"Connecting to Pinecone...")
    print(f"Index name: {args.index_name}")
    print(f"Dimension: {dimension}")
    if args.environment:
        print(f"Environment: {args.environment}")
    if args.host:
        print(f"Host: {args.host}")
    print(f"Recreate index: {args.recreate}")
    print(f"{'='*60}\n")
    
    try:
        backend = PineconeBackend(
            index_name=args.index_name,
            dimension=dimension,
            api_key=api_key,
            environment=args.environment,
            host=args.host,
            data_path=args.data_path,
            recreate_index=args.recreate
        )
        print(f"\n✓ PineconeBackend initialized successfully")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to initialize PineconeBackend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify the index with detailed stats
    print(f"\n{'='*60}")
    print(f"Verifying index...")
    print(f"{'='*60}")
    try:
        stats = backend.index.describe_index_stats()
        print(f"\nIndex stats:")
        print(f"  Total vector count: {stats.get('total_vector_count', 0):,}")
        print(f"  Dimension: {dimension}")
        
        # Check namespaces if available
        if 'namespaces' in stats:
            print(f"  Namespaces: {list(stats['namespaces'].keys())}")
            for ns_name, ns_stats in stats['namespaces'].items():
                print(f"    Namespace '{ns_name}': {ns_stats.get('vector_count', 0):,} vectors")
        
        num_entities = stats.get('total_vector_count', 0)
        
        # Read expected number of vectors from file
        with open(args.data_path, 'rb') as f:
            expected_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        
        print(f"\n{'='*60}")
        print(f"Verification Summary:")
        print(f"  Expected vectors: {expected_vectors:,}")
        print(f"  Actual vectors in index: {num_entities:,}")
        
        if num_entities == expected_vectors:
            print(f"  ✓ SUCCESS: All vectors inserted correctly!")
        elif num_entities == 0:
            print(f"  ✗ ERROR: No vectors found in index!")
            print(f"  ✗ The index may be empty or vectors were not inserted.")
        elif num_entities < expected_vectors:
            print(f"  ⚠ WARNING: Only {num_entities:,} out of {expected_vectors:,} vectors found!")
            print(f"  ⚠ Some vectors may not have been inserted.")
        else:
            print(f"  ⚠ WARNING: Index contains more vectors ({num_entities:,}) than expected ({expected_vectors:,})!")
            print(f"  ⚠ The index may have been populated from a previous run.")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to verify index stats: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n⚠ WARNING: Could not verify vector count, but index may still be valid.")
        print(f"  Try checking the index manually in the Pinecone dashboard.")


if __name__ == "__main__":
    main()

