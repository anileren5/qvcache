# Docker scaffolding for Aker + pgvector benchmarking

This directory provides a minimal, **sample** container setup for running four benchmark targets:

- Vanilla pgvector (baseline)
- Aker-integrated pgvector (Standard)
- Aker-integrated pgvector (Potluck)
- Aker-integrated pgvector (Proximity)

The benchmark workload format and pipeline are **not** changed by these files. The goal is to let you build
separate images and run PostgreSQL inside each container.

## Repository assumptions (matching your plan)

The project is expected to have:

- `apps/pgvector/pgvector.patch` : patch applied for Aker-integrated images

This scaffolding downloads and builds **unmodified PostgreSQL 16.2** from the official
source tarball, and installs **pgvector v0.8.0** from the upstream Git repository.

## Build images

From the project root:

```bash
./docker/scripts/build_images.sh
```

This builds:

- `aker_pgvector_pg16_base:16`
- `aker_pgvector_vanilla:latest`
- `aker_pgvector_standard:latest`
- `aker_pgvector_potluck:latest`
- `aker_pgvector_proximity:latest`

## Run one container (idle)

Run a container that stays alive but does not automatically start PostgreSQL:

```bash
./docker/scripts/run_container.sh aker_pgvector_standard:latest aker_std 5432
```

The script bind-mounts the **project root** into the container at the **same absolute path**
to keep benchmark paths consistent.

## Start / stop PostgreSQL inside the container

Use the helper script (parses the same INI configs used by the benchmark):

```bash
./docker/scripts/exec_pg_ctl.sh aker_std start pgvector-bench/configs/search_workload_hnsw.ini 0
./docker/scripts/exec_pg_ctl.sh aker_std status pgvector-bench/configs/search_workload_hnsw.ini
./docker/scripts/exec_pg_ctl.sh aker_std stop  pgvector-bench/configs/search_workload_hnsw.ini
```

The last argument is an optional NUMA node (e.g., `0`), applied via `numactl`.

## Notes / TODOs

- This is a scaffolding layer. You will still need to decide how the benchmark scripts
  coordinate server start/stop when PostgreSQL lives inside containers.
- `pgvector.patch` is required for Aker-integrated images and is expected to be placed under
  `apps/pgvector/pgvector.patch`.
