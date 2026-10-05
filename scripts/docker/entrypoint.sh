#!/bin/bash
# First start: install missing image deps, then build Aker + QVCache if binaries are absent.
set -euo pipefail
cd /app

if [[ ! -f /usr/include/postgresql/libpq-fe.h ]]; then
  echo "[qvcache] installing libpq-dev"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq libpq-dev
fi

need_build=0
for bin in \
    qvcache_pgvector_search \
    qvcache_diskann_search \
    aker_pgvector_search \
    aker_diskann_search \
    diskann_build_index \
    compute_groundtruth
do
  if [[ ! -x "build/benchmarks/${bin}" ]]; then
    need_build=1
    break
  fi
done
if [[ ! -e /usr/local/lib/libaker.so && ! -e /app/external/Aker/build/lib/libaker.so ]]; then
  need_build=1
fi

if [[ "${need_build}" == 1 ]]; then
  echo "[qvcache] building Aker + QVCache"
  ./build.sh
fi

exec "$@"
