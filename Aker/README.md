![aker-logo](aker-logo-1.png)

# Aker: Density-Aware Approximate Caching for Vector Search

**Aker** is an approximate result cache that sits *above* an ANNS index (or vector database). It caches top-K
nearest-neighbor results for previous queries and serves future queries from cache when the query vector is
sufficiently similar to a cached representative.

This repository contains the **core C++ library** (`libaker.so`) plus optional integration/benchmark
components. This README focuses on the core library.

---

## Aker

**Aker is designed to:**

- Cache top-K results for repeated or nearby query vectors.
- Avoid calling the underlying index entirely on cache hit.
- Support an exact-hit fast path (same query_id) and an approximate-hit path (similar query vector).
- Provide a stable public API (C++ + C ABI wrappers) while keeping internals modular.

**Aker is not:**

- An ANN index by itself. It relies on your system (FAISS, pgvector, etc.) to produce results on cache miss.
- A persistence layer. Cache state is in-memory and process-local.
- A correctness guarantee. Approximate-hit serves cached results that may deviate from true top-K.

---

## Core idea: exact- + approximate-hit

Aker identifies cache entries by a caller-defined `query_id` (typically a stable hash of the query vector).
It then supports two lookup paths:

1) **Exact-hit**: direct lookup by `query_id`.
2) **Approximate-hit**: search a small *representative filter* (FAISS HNSW) over cached representatives,
   then verify the candidate by a distance threshold.

High-level flow:

```text
Caller (DB / index)                      Aker (cache)
-------------------                      -----------
getCacheEntry(query)
  |-- exact lookup by query_id --------> lookup_table
  |                                      |
  |                                      +-- hit -> return cached neighbors
  |
  |-- approx lookup -------------------> ApproxFilter (FAISS HNSW over representatives)
                                         |
                                         +-- candidate repr_id
                                         +-- verify: distance(query, repr_query) < threshold
                                         +-- hit -> return cached neighbors

miss:
  |-- run underlying index search
  |-- createCacheEntry(query, neighbors)
  `-- insertCacheEntry(query_id, entry)
```

Notes:

- Aker stores a fixed-size neighbor list per entry: `slot_list_size = in_topk + top_delta`.
  Most callers serve only the first `in_topk` neighbors; `top_delta` is extra slack for maintenance.
- The approximate filter indexes representative query vectors, not result vectors.
  This keeps the filter small and fast.

---

## Evaluation modes

Aker selects its threshold policy at **compile time** (exactly one mode must be enabled):

- **Standard mode** (default): adaptive per-entry threshold.
  - exact-hit tends to loosen per-entry threshold
  - approximate-hit tends to tighten per-entry threshold
  - includes write-log maintenance paths

- **Proximity mode** (`AKER_ENABLE_PROXIMITY_MODE=ON`): global fixed threshold.
  - threshold is constant throughout runtime (`tuning.global_thresh`)

- **Potluck mode** (`AKER_ENABLE_POTLUCK_MODE=ON`): global threshold tuning + dropout.
  - maintains a global threshold and may randomly drop approximate-hits to force revalidation

These two options are mutually exclusive:

```text
AKER_ENABLE_PROXIMITY_MODE  (ON/OFF)
AKER_ENABLE_POTLUCK_MODE    (ON/OFF)
=> if both OFF, Standard mode is used
```

---

## Directory layout

```text
.
├─ inc/                 # public headers (C++ + C ABI)
│  ├─ core/             # ANNSCache internals
│  └─ utils/            # config parser, helpers
├─ src/                 # library implementation (*.cc, *.c)
├─ bootstrap/           # example configs (INI / JSON)
├─ test/                # small utilities
├─ extern/              # external headers expected on include path
├─ apps/                # optional integrations (e.g., pgvector patch)
├─ docker/              # container build scripts
└─ pgvector-bench/      # optional benchmark harness (see its README)
```

For the core library, the main entry points are:

- C++: `inc/core/ak_anns_cache.hh`
- C ABI: `inc/ak_anns_cache_c_wrapper.h`

---

## Dependencies

### Toolchain

- Tested on Linux
- CMake >= 3.10
- C++17 compiler (this project sets `CMAKE_CXX_STANDARD=17`)

### Boost

The CMake default expects `BOOST_ROOT=/opt/boost_1_86` but you can override it.

Example build/install (one possible way):

```bash
./bootstrap
sudo ./b2 install --prefix=/opt/boost_1_86 --with=all

# If the dynamic linker cannot find Boost at runtime:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/boost_1_86/lib"
```

### FAISS

FAISS provides the approximate filter used for representative lookup.
Build/install example:

```bash
cmake -B build . \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_CUVS=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_C_API=ON \
  -DFAISS_ENABLE_MKL=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -C build -j faiss
make -C build -j faiss_avx2
make -C build -j faiss_avx512

sudo make -C build install

# If the dynamic linker cannot find FAISS at runtime:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
```

If you need the upstream source, see the official repository:
https://github.com/facebookresearch/faiss

### External headers

This repository includes `extern/` in the compiler include path.

- **xxHash (XXH3)**: required for `akerDefaultHash()`.
  - Expected include: `xxHash/xxh3.h`
- **YCSB-C headers**: used by the `ak_ycsb_gen` utility.
  - Expected include prefix: `YCSB-C/core/...`

---

## Build

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT=/opt/boost_1_86 \
  -DFAISS_ROOT=/usr/local

cmake --build build -j
```

Artifacts:

- `build/lib/libaker.so`
- `build/bin/ak_ycsb_gen` (utility target)

`ak_ycsb_gen` writes the Zipfian key sequence output files under:

- `apps/ak_ycsb_gen/sequence.csv`
- `apps/ak_ycsb_gen/sequence-freqs.csv`

### Build with a specific mode

Standard (default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAKER_ENABLE_PROXIMITY_MODE=OFF \
  -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build build -j
```

Proximity:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAKER_ENABLE_PROXIMITY_MODE=ON \
  -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build build -j
```

Potluck:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAKER_ENABLE_PROXIMITY_MODE=OFF \
  -DAKER_ENABLE_POTLUCK_MODE=ON
cmake --build build -j
```

---

## Bootstrap configuration (INI/JSON)

Aker loads a *bootstrap config* into `anns_cache_config_t`.

- Supported formats: **INI** and **JSON**
- Example configs: `bootstrap/aker-*.ini`, `bootstrap/aker-*.json`

Common fields:

- `vector_format.dimension` (uint32)
- `vector_format.vector_in_bytes` (bytes per vector payload)
- `capacity.pool_size` (vector slots in the pool)
- `capacity.in_topk`
- `capacity.top_delta`
- `tuning.*` (mode-dependent)
- `distance_metric` (`l2` default)

Example (JSON):

```json
{
  "vector_format": { "dimension": 128, "vector_in_bytes": 512 },
  "capacity": { "pool_size": 1000000, "in_topk": 10, "top_delta": 5 },
  "tuning": {
    "global_thresh": 0.0,
    "dropout": 0.0,
    "risk_thresh": 0.35,
    "alpha_tighten": 0.9,
    "alpha_loosen": 1.1
  },
  "distance_metric": "l2"
}
```

Mode notes:

- **Standard**: `global_thresh` is not used as the hit threshold (per-entry threshold is adaptive).
- **Proximity**: uses `tuning.global_thresh` as a fixed threshold.
- **Potluck**: uses `dropout` and maintains/tunes a global threshold.

Environment variable:

- Some integrations load the bootstrap path from `AKER_CONFIG_PATH`.
- The core library can load explicitly via `akerImportAnnsCacheConfig(path, ...)`.

---

## Programming guide

### C++ API (direct)

The primary class is `aker::ANNSCache`:

- `getCacheEntry()`
- `createCacheEntry()` / `insertCacheEntry()`
- `insertWriteLogEntry()` / `processWriteLogEntries()`
- `markVectorDeleted()`
- `exportTraceToFiles()`

See: `inc/core/ak_anns_cache.hh`

### C API

The C API wrappers are in `inc/ak_anns_cache_c_wrapper.h`.

Minimal flow (pseudo-code):

```c
// 1) Load config and create cache
anns_cache_parameter_c_t param;
akerImportAnnsCacheConfig("bootstrap/aker-standard.json", &param);
anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(param);

// 2) Build a query vector view
uint64_t query_id = akerDefaultHash((char*)query_bytes, query_bytes_len);
char* q_slot = akerCreateVectorSlot(query_id, query_bytes_len, (char*)query_bytes, 0, 0, 0.0f);

// transform_callback converts raw bytes -> float[dim] (for FAISS filter)
char* q_view = akerCreateVectorView(q_slot,
                                   param.vector_format.dimension,
                                   param.vector_format.vector_in_bytes,
                                   my_transform_callback);

// 3) Lookup
bool similar = false;
bool invalid = false;
char* entry = akerGetCacheEntry(cache, q_view, &similar, &invalid, my_distance_func);

if (entry != NULL) {
    // hit: consume results (typically first in_topk)
    // ...
    akerDestroyCacheEntry(entry);
} else {
    // miss: run underlying index search, then insert
    // - build VectorSlot list for results
    // - akerCreateCacheEntry(...)
    // - akerInsertCacheEntry(...)
}

akerDestroyVectorView(q_view);
akerDestroyVectorSlot(q_slot);
akerDestroyAnnsCache(cache);
```

Distance and transform callbacks:

- `transform_callback`: raw vector bytes -> `float[dimension]` for the representative filter.
- `distance_function`: used to verify similarity against the threshold.
  - Signature: `float (*)(uint8_t*, uint8_t*, size_t)`
  - If your raw vectors are float32, you can cast and reuse the helpers:

```c
float my_l2(uint8_t* a, uint8_t* b, size_t dim) {
    return akerL2Distance((float*)a, (float*)b, dim);
}
```

Ownership rules:

- `akerGetCacheEntry()` returns a deep-copied entry owned by the caller.
  You must free it via `akerDestroyCacheEntry()`.
- VectorSlot / VectorView wrappers created by the caller must also be destroyed.

---

## Tracing and telemetry

Aker can export trace snapshots under:

```text
/tmp/aker_trace_<timestamp>/
```

Export triggers:

- Explicit call: `ANNSCache::exportTraceToFiles()` or `akerExportTraceToFiles()`
- Destructor: exports a final snapshot if the cache had activity

Typical files include:

- `aker_trace_cache_summary.csv`
- `aker_trace_cache_status.txt`
- `aker_trace_parameters.csv`
- `aker_trace_write_log.csv`
- `aker_trace_latency_summary.csv`
- `aker_trace_cache_history.csv`
- `aker_trace_hit_ratio_history.csv`
- `aker_trace_approx_filter_history.csv`

## Note

- The codebase is being continuously reorganized and refined.
- The evaluation results reported in the paper were obtained using [9f0c6c2e](https://github.com/sjoon-oh/Aker/tree/9f0c6c2e91a8c6dc199a069f349f2a2c7fb01a78) commit.
