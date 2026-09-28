# Resolve STREAM / QUERY_STREAM / SKEW into QUERY_STREAM (file tag).
# Source from a refresh script after DATASET is set.
#   STREAM=simZipf DELETE_RATE=0.05 ./scripts/qvcache/qvcache_benchmark_refresh_pgvector.sh
#   STREAM=simZipf+ SKEW=0.99 ./scripts/aker/aker_benchmark_refresh_pgvector.sh
#   QUERY_STREAM=simplus-100k-0.99 still works as a raw file tag.
SKEW="${SKEW:-0.99}"
if [ "$SKEW" = "uniform" ]; then
  _skew_tag="uniform"
else
  _skew_tag="$SKEW"
fi
if [ -n "${STREAM:-}" ]; then
  case "$STREAM" in
    simZipf|simzipf|sim)
      QUERY_STREAM="sim-100k-${_skew_tag}"
      ;;
    simZipf+|simzipf+|simplus)
      QUERY_STREAM="simplus-100k-${_skew_tag}"
      ;;
    simZipf+2|simzipf+2|simplus2)
      QUERY_STREAM="simplus2-100k-${_skew_tag}"
      ;;
    *)
      QUERY_STREAM="$STREAM"
      ;;
  esac
else
  QUERY_STREAM="${QUERY_STREAM:-sim-100k-${_skew_tag}}"
fi
unset _skew_tag
