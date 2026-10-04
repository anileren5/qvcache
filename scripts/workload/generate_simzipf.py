#!/usr/bin/env python3
"""Generate an Aker-style simZipf stream: frozen 50-variant groups, then Zipf or uniform over groups.

Stage 1 follows the paper: for each base query q, 50 interpolations
q' = (1-λ)q + λp with λ ~ Unif(0, 0.5). Stage 2 draws a group (YCSB
scrambled Zipf or i.i.d. uniform) and then a member of that group uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from generate_simzipf_plus import (  # noqa: E402
    _cast_interpolated,
    choose_topics,
    read_bin_vectors,
    write_bin_vectors,
)


def build_frozen_groups(
    queries: np.ndarray,
    group_size: int,
    rng: np.random.Generator,
    dtype: str,
    lambda_max: float = 0.5,
) -> np.ndarray:
    n_topics, dim = queries.shape
    q_float = queries.astype(np.float32, copy=False)
    partners = rng.integers(0, n_topics, size=(n_topics, group_size), dtype=np.int64)
    topic_idx = np.arange(n_topics, dtype=np.int64)[:, None]
    same = partners == topic_idx
    while np.any(same):
        partners[same] = rng.integers(0, n_topics, size=int(same.sum()), dtype=np.int64)
        same = partners == topic_idx
    lam = rng.uniform(0.0, lambda_max, size=(n_topics, group_size)).astype(np.float32)
    mixed = (1.0 - lam)[:, :, None] * q_float[:, None, :] + lam[:, :, None] * q_float[partners]
    return _cast_interpolated(mixed.reshape(-1, dim), dtype).reshape(n_topics, group_size, dim)


def generate_simzipf(
    queries: np.ndarray,
    n_stream: int,
    rng: np.random.Generator,
    dtype: str,
    distribution: str = "uniform",
    skew: float = 0.99,
    group_size: int = 50,
    lambda_max: float = 0.5,
) -> tuple[np.ndarray, dict]:
    n_topics, dim = queries.shape
    groups = build_frozen_groups(queries, group_size, rng, dtype, lambda_max=lambda_max)
    topics = choose_topics(n_topics, n_stream, rng, distribution, skew)
    members = rng.integers(0, group_size, size=n_stream, dtype=np.int64)
    stream = groups[topics, members]
    unique_rows = len({row.tobytes() for row in stream})
    stats = {
        "n_topics": int(n_topics),
        "n_stream": int(n_stream),
        "group_size": int(group_size),
        "lambda_max": float(lambda_max),
        "distribution": distribution,
        "skew": None if distribution == "uniform" else float(skew),
        "dtype": dtype,
        "dim": int(dim),
        "unique_rows": int(unique_rows),
        "unique_topics_used": int(len(np.unique(topics))),
        "unique_group_members_emitted": int(len(np.unique(topics.astype(np.int64) * group_size + members))),
        "zipf_scrambled": distribution == "zipfian",
        "construction": "aker_simzipf_frozen_groups",
    }
    return stream, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Aker simZipf DiskANN query bin")
    parser.add_argument("--queryset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="int8", choices=["float", "int8", "uint8"])
    parser.add_argument("--distribution", default="uniform", choices=["zipfian", "uniform"])
    parser.add_argument("--skew", type=float, default=0.99)
    parser.add_argument("--n-stream", type=int, default=100000)
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-json", default="")
    args = parser.parse_args()

    if args.group_size < 1:
        raise SystemExit("group-size must be >= 1")
    if args.lambda_max <= 0 or args.lambda_max > 1:
        raise SystemExit("lambda-max must be in (0, 1]")
    if args.distribution == "zipfian" and (args.skew <= 0 or args.skew >= 1):
        raise SystemExit("skew (Zipf theta) must be in (0, 1), YCSB-style")

    queries = read_bin_vectors(args.queryset, args.dtype)
    rng = np.random.default_rng(args.seed)
    stream, stats = generate_simzipf(
        queries,
        args.n_stream,
        rng,
        args.dtype,
        distribution=args.distribution,
        skew=args.skew,
        group_size=args.group_size,
        lambda_max=args.lambda_max,
    )
    write_bin_vectors(stream, args.output, args.dtype)
    stats["queryset"] = os.path.abspath(args.queryset)
    stats["output"] = os.path.abspath(args.output)
    stats["seed"] = args.seed
    print(json.dumps(stats, indent=2))
    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
            f.write("\n")
    print(f"Wrote {args.output}: npts={stream.shape[0]} dim={stream.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
