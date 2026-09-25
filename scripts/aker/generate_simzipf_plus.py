#!/usr/bin/env python3
"""Generate a simZipf+ query stream (Zipf over topics, fresh interpolation per access).

Unlike Aker simZipf, variants are not drawn from a frozen pool of 50, so the
stream is not supposed to replay bit-identical embeddings. Default epsilon
matches the windowed SIFT generator (0.01).

simZipf+2 uses the same construction but draws ε ~ Unif(0, lambda_max) per
access (Aker Stage-1 weight, default 0.5) instead of a fixed ε.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

_DTYPE_MAP = {
    "float": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
}

FNV_OFFSET_BASIS64 = np.uint64(0xCBF29CE484222325)
FNV_PRIME64 = np.uint64(1099511628211)


def read_bin_vectors(path: str, dtype: str) -> np.ndarray:
    np_dtype = _DTYPE_MAP[dtype]
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"Invalid bin header in {path}")
        npts, dim = int(header[0]), int(header[1])
        data = np.fromfile(f, dtype=np_dtype, count=npts * dim)
        if data.size != npts * dim:
            raise ValueError(f"Truncated bin file {path}")
        return data.reshape(npts, dim)


def write_bin_vectors(vectors: np.ndarray, path: str, dtype: str) -> None:
    np_dtype = _DTYPE_MAP[dtype]
    npts, dim = vectors.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = np.ascontiguousarray(vectors, dtype=np_dtype)
    with open(path, "wb") as f:
        f.write(struct.pack("II", int(npts), int(dim)))
        out.tofile(f)


def fnv_hash64(values: np.ndarray) -> np.ndarray:
    hashed = np.full(values.shape, FNV_OFFSET_BASIS64, dtype=np.uint64)
    x = values.astype(np.uint64, copy=True)
    for _ in range(8):
        octet = x & np.uint64(0xFF)
        x >>= np.uint64(8)
        hashed ^= octet
        hashed *= FNV_PRIME64
    return hashed


def ycsb_zipfian_ranks(n_items: int, theta: float, rng: np.random.Generator, size: int) -> np.ndarray:
    """YCSB-C ZipfianGenerator::Next over items {0, ..., n_items-1}."""
    if n_items < 2:
        raise ValueError("Zipf needs at least 2 topics")
    i = np.arange(1, n_items + 1, dtype=np.float64)
    zeta_n = np.sum(np.power(i, -theta))
    zeta_2 = 1.0 + np.power(2.0, -theta)
    alpha = 1.0 / (1.0 - theta)
    eta = (1.0 - np.power(2.0 / n_items, 1.0 - theta)) / (1.0 - zeta_2 / zeta_n)
    u = rng.random(size)
    uz = u * zeta_n
    ranks = np.empty(size, dtype=np.int64)
    ranks[uz < 1.0] = 0
    second = (uz >= 1.0) & (uz < 1.0 + np.power(0.5, theta))
    ranks[second] = 1
    rest = ~(uz < 1.0) & ~second
    ranks[rest] = (n_items * np.power(eta * u[rest] - eta + 1.0, alpha)).astype(np.int64)
    return np.clip(ranks, 0, n_items - 1)


def scrambled_zipf_topics(n_items: int, theta: float, rng: np.random.Generator, size: int) -> np.ndarray:
    ranks = ycsb_zipfian_ranks(n_items, theta, rng, size)
    scrambled = fnv_hash64(ranks.astype(np.uint64)) % np.uint64(n_items)
    return scrambled.astype(np.int64)


def choose_topics(
    n_items: int,
    size: int,
    rng: np.random.Generator,
    distribution: str,
    skew: float,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.integers(0, n_items, size=size, dtype=np.int64)
    if distribution == "zipfian":
        if skew <= 0 or skew >= 1:
            raise ValueError("zipfian skew (YCSB theta) must be in (0, 1)")
        return scrambled_zipf_topics(n_items, skew, rng, size)
    raise ValueError(f"Unknown distribution: {distribution}")


def _cast_interpolated(interpolated: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float":
        return interpolated.astype(np.float32, copy=False)
    if dtype == "int8":
        return np.clip(interpolated, -128, 127).astype(np.int8)
    return np.clip(interpolated, 0, 255).astype(np.uint8)


def _sample_epsilons(
    n_stream: int,
    epsilon: float,
    rng: np.random.Generator,
    lambda_max: float | None,
) -> np.ndarray:
    if lambda_max is not None:
        return rng.uniform(0.0, lambda_max, size=n_stream).astype(np.float32)
    return np.full(n_stream, np.float32(epsilon), dtype=np.float32)


def generate_simzipf_plus(
    queries: np.ndarray,
    n_stream: int,
    epsilon: float,
    rng: np.random.Generator,
    dtype: str,
    distribution: str = "zipfian",
    skew: float = 0.99,
    max_resample: int = 32,
    lambda_max: float | None = None,
) -> tuple[np.ndarray, dict]:
    n_topics, dim = queries.shape
    topics = choose_topics(n_topics, n_stream, rng, distribution, skew)
    partners = rng.integers(0, n_topics, size=n_stream, dtype=np.int64)
    same = partners == topics
    while np.any(same):
        partners[same] = rng.integers(0, n_topics, size=int(same.sum()), dtype=np.int64)
        same = partners == topics

    q_float = queries.astype(np.float32, copy=False)
    eps = _sample_epsilons(n_stream, epsilon, rng, lambda_max)
    mixed = (1.0 - eps)[:, None] * q_float[topics] + eps[:, None] * q_float[partners]
    stream = _cast_interpolated(mixed, dtype)

    collisions_topic = 0
    collisions_prev = 0
    if dtype != "float":
        topic_cast = queries.astype(stream.dtype, copy=False)
        seen: set[bytes] = set()
        for t in range(n_stream):
            raw = stream[t].tobytes()
            tries = 0
            while (
                np.array_equal(stream[t], topic_cast[topics[t]]) or raw in seen
            ) and tries < max_resample:
                if np.array_equal(stream[t], topic_cast[topics[t]]):
                    collisions_topic += 1
                else:
                    collisions_prev += 1
                p = int(rng.integers(0, n_topics))
                while p == int(topics[t]):
                    p = int(rng.integers(0, n_topics))
                partners[t] = p
                if lambda_max is not None:
                    eps[t] = np.float32(rng.uniform(0.0, lambda_max))
                mixed_row = (1.0 - eps[t]) * q_float[topics[t]] + eps[t] * q_float[p]
                stream[t] = _cast_interpolated(mixed_row.reshape(1, dim), dtype)[0]
                raw = stream[t].tobytes()
                tries += 1
            seen.add(raw)
        unique_rows = len(seen)
    else:
        unique_rows = len({row.tobytes() for row in stream})

    plus2 = lambda_max is not None
    stats = {
        "n_topics": int(n_topics),
        "n_stream": int(n_stream),
        "distribution": distribution,
        "skew": None if distribution == "uniform" else float(skew),
        "epsilon": None if plus2 else float(epsilon),
        "lambda_max": float(lambda_max) if plus2 else None,
        "epsilon_mean": float(eps.mean()),
        "epsilon_min": float(eps.min()),
        "epsilon_max": float(eps.max()),
        "dtype": dtype,
        "dim": int(dim),
        "unique_rows": int(unique_rows),
        "unique_topics_used": int(len(np.unique(topics))),
        "collisions_equal_topic_resampled": int(collisions_topic),
        "collisions_duplicate_resampled": int(collisions_prev),
        "zipf_scrambled": distribution == "zipfian",
        "construction": (
            "simzipf_plus2_fresh_uniform_lambda" if plus2 else "simzipf_plus_fresh_fixed_epsilon"
        ),
    }
    return stream, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simZipf+ DiskANN query bin")
    parser.add_argument("--queryset", required=True, help="Unique queryset .bin (DiskANN)")
    parser.add_argument("--output", required=True, help="Output query .bin")
    parser.add_argument("--dtype", default="int8", choices=["float", "int8", "uint8"])
    parser.add_argument(
        "--distribution",
        default="zipfian",
        choices=["zipfian", "uniform"],
        help="Topic popularity: YCSB scrambled Zipf or i.i.d. uniform (Aker's fourth case)",
    )
    parser.add_argument("--skew", type=float, default=0.99, help="YCSB Zipf theta; ignored if --distribution uniform")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=0.0,
        help="If >0, draw ε ~ Unif(0, lambda-max) per access (simZipf+2). Overrides --epsilon.",
    )
    parser.add_argument("--n-stream", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-json", default="", help="Optional stats path")
    args = parser.parse_args()

    if args.epsilon < 0 or args.epsilon > 1:
        raise SystemExit("epsilon must be in [0, 1]")
    if args.lambda_max < 0 or args.lambda_max > 1:
        raise SystemExit("lambda-max must be in [0, 1]")
    if args.distribution == "zipfian" and (args.skew <= 0 or args.skew >= 1):
        raise SystemExit("skew (Zipf theta) must be in (0, 1), YCSB-style")

    lambda_max = args.lambda_max if args.lambda_max > 0 else None
    queries = read_bin_vectors(args.queryset, args.dtype)
    rng = np.random.default_rng(args.seed)
    stream, stats = generate_simzipf_plus(
        queries,
        args.n_stream,
        args.epsilon,
        rng,
        args.dtype,
        distribution=args.distribution,
        skew=args.skew,
        lambda_max=lambda_max,
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
