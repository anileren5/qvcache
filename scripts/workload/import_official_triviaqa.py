#!/usr/bin/env python3
"""Turn Aker's released TriviaQA simZipf files into DiskANN query and truthset bins.

The Hugging Face files are the paper's simZipf streams: Zipf over question
groups, then a uniform draw from that question's frozen pool of paraphrases.
Query vectors are float16 [N, 768]. Each pickle record holds the same vector
plus exact top-100 ids and scores against SPHERE-10M.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import struct

import numpy as np


def write_query_bin(vectors: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    npts, dim = arr.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("II", int(npts), int(dim)))
        arr.tofile(f)


def write_truthset(ids: np.ndarray, dists: np.ndarray, path: str) -> None:
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    dists = np.ascontiguousarray(dists, dtype=np.float32)
    if ids.shape != dists.shape or ids.ndim != 2:
        raise ValueError(f"truthset shape mismatch: {ids.shape} vs {dists.shape}")
    npts, k = ids.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(npts), int(k)))
        ids.tofile(f)
        dists.tofile(f)


def load_pickle_truth(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    payloads = np.stack([np.asarray(rec["payload"], dtype=np.float16).reshape(-1) for rec in records])
    ids = np.stack([np.asarray(rec["gt_ids"]).reshape(-1) for rec in records])
    scores = np.stack([np.asarray(rec["gt_scores"]).reshape(-1) for rec in records])
    if np.any(ids < 0) or np.any(ids > np.iinfo(np.uint32).max):
        raise ValueError(f"ground-truth ids in {path} do not fit in uint32")
    return payloads, ids, scores


def import_one(npy_path: str, pkl_path: str, out_dir: str, skew: str) -> dict:
    npy = np.load(npy_path)
    payloads, ids, scores = load_pickle_truth(pkl_path)
    if npy.shape != payloads.shape:
        raise ValueError(f"{npy_path} shape {npy.shape} != pickle payloads {payloads.shape}")
    if not np.array_equal(npy.astype(np.float16, copy=False), payloads):
        # Released npy and pickle payloads are the same float16 matrix.
        n_mismatch = int(np.sum(np.any(npy.astype(np.float32) != payloads.astype(np.float32), axis=1)))
        raise ValueError(f"{n_mismatch} query rows differ between npy and pickle")

    query_path = os.path.join(out_dir, "query.bin")
    gt_path = os.path.join(out_dir, "groundtruth.bin")
    stats_path = os.path.join(out_dir, "stats.json")
    write_query_bin(npy, query_path)
    write_truthset(ids, scores, gt_path)
    unique_rows = len({row.tobytes() for row in np.ascontiguousarray(npy)})
    stats = {
        "construction": "aker_official_triviaqa_simzipf",
        "skew": skew,
        "n_stream": int(npy.shape[0]),
        "dim": int(npy.shape[1]),
        "dtype": "float32",
        "source_dtype": str(npy.dtype),
        "k": int(ids.shape[1]),
        "unique_rows": int(unique_rows),
        "exact_repeat_rows": int(npy.shape[0] - unique_rows),
        "npy": os.path.abspath(npy_path),
        "pkl": os.path.abspath(pkl_path),
        "query_bin": os.path.abspath(query_path),
        "groundtruth_bin": os.path.abspath(gt_path),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Import official TriviaQA simZipf npy/pkl")
    parser.add_argument("--npy", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skew", required=True)
    args = parser.parse_args()
    import_one(args.npy, args.pkl, args.out_dir, args.skew)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
