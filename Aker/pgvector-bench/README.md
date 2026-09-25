# pgvector-bench

This directory is a refactoring of the legacy `script/pgvector-*/2-runs/*.py` pipeline.

---

## Execution model (host client + container server)

- **Host** runs the benchmark driver (bash + Python client).
- **Docker container** runs PostgreSQL + pgvector (and Aker-integrated pgvector for Aker modes).
- PostgreSQL data directories (**PGDATA**) live on the **host** and are bind-mounted into the container.

ASCII overview:

```text
Host (pgvector-bench/)                           Docker container (server)
---------------------                            -------------------------
*.sh (pipeline glue)                             idle container (sleep)
  ├─ docker run/start (reuse)  ---------------->  (container stays up)
  ├─ docker exec numactl pg_ctl start ---------->  postmaster starts
  ├─ python -m ak_bench (client) --TCP--------->  127.0.0.1:<port> (host network)
  ├─ docker exec pg_ctl stop   ---------------->  postmaster stops
  ├─ docker cp /tmp/aker_trace_* <------------  /tmp traces copied out
  └─ docker stop (container kept) ------------>  container stopped (reusable)
```

---

## Requirements

- Linux + Docker (daemon running).
- Python 3 + venv (the scripts create `./.venv` automatically).

---

## Directory layout

```text
pgvector-bench/
├─ ak_bench/                 # python -m ak_bench ...
├─ configs/                  # INI files + postgresql.conf
├─ dataset/                  # Place datasets here (dataset/<DATASET>/...)
│  └─ SPACEV/                # example dataset name
├─ datastore/                # PostgreSQL data directories (PGDATA)
├─ output/                   # Results, merged /tmp traces, generated workloads
└─ *.sh                      # Bash entrypoints (run from here)
```

---

## Docker images

Build images from the project root:

```bash
# From project root:
./docker/scripts/build_images.sh
```

This builds:

- `aker_pgvector_vanilla:latest`
- `aker_pgvector_standard:latest`
- `aker_pgvector_potluck:latest`
- `aker_pgvector_proximity:latest`

Notes:

- These images compile PostgreSQL 16.2 from the official source tarball and install pgvector v0.8.0.
- Aker-integrated images additionally apply `apps/pgvector/pgvector.patch` and build Aker in the selected mode.

---

## Docker mode is required (how to select the image)

**You MUST specify a Docker image**, either by environment variable or in the INI.

### Option A: environment variable

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_vanilla:latest
```

### Option B: INI (`configs/*.ini`)

```ini
[docker]
image = aker_pgvector_vanilla:latest
network = host
numa_node = 0
remove_container_on_exit = 0
```

---

## Datastore path

The container does not “search” for PGDATA.

1. The host scripts resolve `postgres.datastore` to an absolute path under `pgvector-bench/datastore/...` (unless you provide an absolute path).
2. The benchmark container bind-mounts the **project root at the SAME absolute path** inside the container.
3. `pg_ctl` is executed in the container with `-D <that absolute path>`.

Example mapping:

```text
Host:      /home/user/aker/pgvector-bench/datastore/SPACEV_hnsw
Container: /home/user/aker/pgvector-bench/datastore/SPACEV_hnsw  (same path string)
```

This same-path bind-mount approach also applies to `postgresql.conf` and Aker bootstrap files.

---

## Workload

- **Search-workload** (legacy `workloada` trace format: `list[dict]`)
- **Stress-workload** (legacy `trace-stress.py` format: `dict{search:list[dict], insert:list[dict]}`)


---

## Aker bootstrap/config file handling

If your Aker-integrated pgvector reads `AKER_CONFIG_PATH` via `getenv()`, you can provide it in two ways:

### A) Pass via `--aker-config` (recommended)

```bash
./ak_bench_run_search_workload.sh --config configs/spacev-1m-small-test.ini \
  --aker-config ../bootstrap/aker-standard.json
```

This overrides any `AKER_CONFIG_PATH` set in `[env]`.

### B) Provide in INI `[env]`

```ini
[env]
AKER_CONFIG_PATH = /abs/path/to/bootstrap/aker-standard.json
```

Because the project root is bind-mounted at the same absolute path, absolute paths resolved on host remain valid inside the container.

---

## /tmp traces and “wait before shutdown”

Aker trace exporting can be expensive. To avoid shutting down the server before traces finish exporting:

- If `AK_BENCH_TRACE_EXPORT_WAIT_SEC` is set, the harness uses it (0 disables).
- Otherwise, if `AKER_CONFIG_PATH` is set, it defaults to **600 seconds**.

Examples:

```bash
export AK_BENCH_TRACE_EXPORT_WAIT_SEC=600   # default for Aker runs
export AK_BENCH_TRACE_EXPORT_WAIT_SEC=0     # disable waiting
```

After each run:

- Newly created `/tmp` traces are copied from the container:
  - per-run: `output/.../runs/<run_id>/tmp_traces/`
  - merged: `output/.../merged_tmp_traces/`
- The container `/tmp` traces are then removed to keep the container reusable.

---

## Container lifecycle

Containers are **not removed** automatically. They are stopped and kept for reuse.

- Default: stop container after each evaluation
- Optional: remove container on exit:
  - INI: `docker.remove_container_on_exit = 1`
  - Env: `export AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT=1`

To inspect leftover containers:

```bash
docker ps -a | grep akerbench
```

---

## Pipeline overview

```text
(A) Prepare storage once (base pgvector recommended)
    - initdb
    - create table
    - COPY vectors (contiguous IDs)
    - build index
    - create PGDATA-clean snapshot

(B) Generate workload (when storage exists)
    - restore clean snapshot (NO cache drop)
    - start postgres
    - generate *.pkl (legacy format)
    - fix-trace (legacy hole fixer)
    - stop postgres

(C) Run benchmark (repeatable)
    - restore clean snapshot
    - ALWAYS drop OS page cache
    - export [env] (and optional --aker-config override)
    - start postgres (container, NUMA pinned)
    - run benchmark under numactl (host)
    - wait for trace export (optional; default 600s if Aker)
    - collect /tmp traces from container
    - stop postgres
    - stop container (keep for reuse)
```

---

## Sample configs

Example INI files under `configs/`:

- `configs/search_workload_hnsw.ini`
- `configs/stress_workload_hnsw.ini`
- `configs/search_workload_ivfflat.ini`
- `configs/stress_workload_ivfflat.ini`

You must edit:

- `dataset.base`, `dataset.search` (place `.npy` files under `dataset/<DATASET>/`)
- `dataset.gt_trace` (recommended under `output/workloads/...`)
- `postgres.host/port/user/password/database`
- (recommended for reproducible repeated runs)
  - `postgres.datastore` / `postgres.datastore_clean`

### PostgreSQL configuration file (`postgresql.conf`)

The benchmark scripts pass `postgresql.conf` to PostgreSQL via:

- `pg_ctl -o "--config-file=..."`

Default shipped config:

- `configs/postgresql.conf`

Override per INI:

- `postgres.psql_config = <relative or absolute path>`

---

## How to run

### 0) Build images (from project root)

```bash
./docker/scripts/build_images.sh
cd pgvector-bench
```

### 1) Prepare storage (On vanilla pgvector)

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_vanilla:latest
./ak_bench_prepare_storage_pgvector.sh --config configs/spacev-1m-small-test.ini
```

### 2) Generate workload (run once per dataset/config)

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_vanilla:latest

# Optional: Select numpy as backend
export GT_BACKEND=numpy
export GT_NUMPY_WORKERS=16
export GT_NUMPY_BASE_CHUNK_ROWS=10000
export GT_NUMPY_QUERY_BATCH_SIZE=32

./ak_bench_generate_workload_from_storage.sh --config configs/spacev-1m-small-test.ini
```

### 3) Run Search-workload benchmark (repeatable per version)

Vanilla:

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_vanilla:latest
./ak_bench_run_search_workload.sh --config configs/spacev-1m-small-test.ini --output-dir output
```

Aker Standard:

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_standard:latest
./ak_bench_run_search_workload.sh --config configs/spacev-1m-small-test.ini --output-dir output \
  --aker-config configs/aker-standard-bootstrap.ini
```

Potluck:

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_potluck:latest
./ak_bench_run_search_workload.sh --config configs/spacev-1m-small-test.ini --output-dir output \
  --aker-config ../bootstrap/aker-potluck-mode.json
```

Proximity:

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_proximity:latest
./ak_bench_run_search_workload.sh --config configs/spacev-1m-small-test.ini --output-dir output \
  --aker-config ../bootstrap/aker-proximity-mode.json
```

### 4) Run Stress-workload benchmark

```bash
export AK_BENCH_DOCKER_IMAGE=aker_pgvector_standard:latest
./ak_bench_run_stress_workload.sh --config stress_workload_hnsw.ini --invalidate 0.10 --output-dir output \
  --aker-config ../bootstrap/aker-standard.json
```

---

## Outputs

Typical run outputs:

- `output/runs/<run_id>/report.csv`
- `output/runs/<run_id>/trace-extract-info.csv`
- `output/runs/<run_id>/search-results.pkl`
- `output/runs/<run_id>/tmp_traces/` (copied from container `/tmp`)
- `output/runs/<run_id>/docker_container.log`
- `output/merged_tmp_traces/` (accumulated traces)

---

## Scripts

### Bash entrypoints

- `ak_bench_prepare_storage_pgvector.sh`
  - initdb + COPY + build index + create `<datastore_clean>` snapshot
- `ak_bench_generate_workload_from_storage.sh`
  - start server + generate workload pkl + fix-trace + stop
- `ak_bench_run_search_workload.sh`
  - restore snapshot + drop cache + start + run search + wait + stop + collect traces + stop container
- `ak_bench_run_stress_workload.sh`
  - restore snapshot + run insert/search stress + invalidation + wait + stop + collect traces + stop container
- `ak_bench_pg_safeguard.sh`
  - INI parsing, Docker container lifecycle, pg_ctl start/stop (container), NUMA pin, cache drop, snapshot restore, trace collection

### Python modules (invoked as `python -m ak_bench ...`)

- `ak_bench_cli.py`
  - CLI subcommand routing (create-table, copy-vectors, build-index, generate workloads, run workloads)
- `ak_bench_pgvector_admin.py`
  - table creation, COPY, index build
- `ak_bench_workload_gen.py`
  - workload generation + ground truth creation
- `ak_bench_runner_search.py`
  - timed search queries + recall computation + report outputs
- `ak_bench_runner_stress.py`
  - insert/search phases + invalidation + report outputs

---

## Troubleshooting

If you run on non-Linux Docker environments (Docker Desktop), `--network host` behavior differs. Prefer Linux for reproducible performance benchmarking.