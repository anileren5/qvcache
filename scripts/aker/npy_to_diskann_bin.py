#!/usr/bin/env python3
"""Aker: convert pgvector-bench .npy vectors to DiskANN .bin (npts, dim, packed data)."""

from __future__ import annotations

import argparse
import glob
import os
import struct

import numpy as np


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "float":
        return np.dtype(np.float32)
    if dtype == "int8":
        return np.dtype(np.int8)
    if dtype == "uint8":
        return np.dtype(np.uint8)
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_npy_matrix(path_or_glob: str) -> np.ndarray:
    if "*" in path_or_glob:
        files = sorted(glob.glob(path_or_glob))
        if not files:
            raise FileNotFoundError(f"No files matched: {path_or_glob}")
        parts = [np.load(f) for f in files]
        return np.concatenate(parts, axis=0)
    arr = np.load(path_or_glob, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array in {path_or_glob}, got shape {arr.shape}")
    return arr


def write_diskann_bin(vectors: np.ndarray, output_path: str, dtype: str) -> None:
    np_dtype = _numpy_dtype(dtype)
    npts, dim = vectors.shape
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    chunk = 100_000
    with open(output_path, "wb") as f:
        f.write(struct.pack("I", int(npts)))
        f.write(struct.pack("I", int(dim)))
        for start in range(0, npts, chunk):
            packed = np.ascontiguousarray(vectors[start : start + chunk], dtype=np_dtype)
            packed.tofile(f)
    print(f"Wrote {output_path}: npts={npts} dim={dim} dtype={dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aker: npy -> DiskANN bin")
    parser.add_argument("--input", required=True, help="npy path or glob (e.g. part_*.npy)")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--dtype", default="int8", choices=["float", "int8", "uint8"])
    args = parser.parse_args()
    write_diskann_bin(load_npy_matrix(args.input), args.output, args.dtype)


if __name__ == "__main__":
    main()
