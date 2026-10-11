#!/bin/bash
# Wait for the one-time sphere_1m HNSW build, dump table+index, then run
# refresh workloads by restoring that snapshot instead of rebuilding.
set -u
DUMP=/local/home/agoecer/qvcache/sphere_1m_hnsw.dump
QUEUE_LOG=/home/agoecer/qvcache/logs/refresh/sphere-1m/queue.log
PY_PID=408080

echo "$(date -Is) WAIT snapshot build pid=${PY_PID}" >> "$QUEUE_LOG"

ready=0
for _ in $(seq 1 360); do
  if docker exec postgres psql -U postgres -tAc \
      "SELECT 1 FROM pg_indexes WHERE tablename='sphere_1m' AND indexdef ILIKE '%USING hnsw%' LIMIT 1;" \
      | grep -q 1 \
    && ! docker exec postgres psql -U postgres -tAc \
      "SELECT count(*) FROM pg_stat_progress_create_index;" | grep -q '[1-9]'; then
    ready=1
    break
  fi
  if ! docker exec qvcache bash -c "ps -p ${PY_PID} >/dev/null"; then
    sleep 5
    if docker exec postgres psql -U postgres -tAc \
        "SELECT 1 FROM pg_indexes WHERE tablename='sphere_1m' AND indexdef ILIKE '%USING hnsw%' LIMIT 1;" \
        | grep -q 1; then
      ready=1
    fi
    break
  fi
  sleep 20
done

if [ "$ready" != 1 ]; then
  echo "$(date -Is) ABORT snapshot build did not finish" >> "$QUEUE_LOG"
  exit 1
fi

rows=$(docker exec postgres psql -U postgres -tAc "SELECT count(*) FROM sphere_1m;")
echo "$(date -Is) DUMP sphere_1m rows=${rows} -> ${DUMP}" >> "$QUEUE_LOG"
if ! docker exec postgres pg_dump -U postgres -Fc --no-owner --no-privileges -t sphere_1m > "$DUMP"; then
  echo "$(date -Is) ABORT pg_dump failed" >> "$QUEUE_LOG"
  exit 1
fi
echo "$(date -Is) DUMP done bytes=$(stat -c %s "$DUMP")" >> "$QUEUE_LOG"

restore_index() {
  echo "$(date -Is) RESTORE sphere_1m" >> "$QUEUE_LOG"
  if ! docker exec -i postgres pg_restore -U postgres -d postgres --clean --if-exists --no-owner --no-privileges < "$DUMP"; then
    if ! docker exec postgres psql -U postgres -tAc \
        "SELECT 1 FROM pg_indexes WHERE tablename='sphere_1m' AND indexdef ILIKE '%USING hnsw%' LIMIT 1;" \
        | grep -q 1; then
      echo "$(date -Is) ABORT restore failed" >> "$QUEUE_LOG"
      exit 1
    fi
  fi
  echo "$(date -Is) RESTORE done" >> "$QUEUE_LOG"
}

run_one() {
  local engine="$1" stream="$2" del="$3"
  local script log
  if [ "$engine" = aker ]; then
    script=./scripts/aker/pgvector_refresh.sh
    log="logs/refresh/sphere-1m/${stream}/del${del}/aker.log"
  else
    script=./scripts/qvcache/pgvector_refresh.sh
    log="logs/refresh/sphere-1m/${stream}/del${del}/qvcache.log"
  fi
  echo "$(date -Is) START ${engine} ${stream} insert=0.05 delete=${del}" >> "$QUEUE_LOG"
  docker exec -w /app qvcache bash -c \
    "export PYTHONUNBUFFERED=1; DATASET=sphere-1m DATA_TYPE=float TABLE_NAME=sphere_1m QUERY_STREAM=${stream} INSERT_FRAC=0.05 DELETE_RATE=${del} EF_SEARCH=512 REBUILD_INDEX=0 ${script} > ${log} 2>&1"
  echo "$(date -Is) END ${engine} ${stream} delete=${del} rc=$?" >> "$QUEUE_LOG"
}

first=1
for del in 0.05 0.15 0.25; do
  for stream in sim-100k-0.99 simplus-100k-0.99; do
    for engine in aker qvcache; do
      if [ "$first" = 1 ]; then
        first=0
      else
        restore_index
      fi
      run_one "$engine" "$stream" "$del"
    done
  done
done
echo "$(date -Is) ALL_DONE" >> "$QUEUE_LOG"
