# Resolve DiskANN bin paths for DATASET + QUERY_STREAM.
# Layout:
#   data/<dataset>/base.bin
#   data/<dataset>/query.bin              # topic queryset (QUERY_STREAM=query)
#   data/<dataset>/groundtruth.bin
#   data/<dataset>/queries/<stream>/query.bin
#   data/<dataset>/queries/<stream>/groundtruth.bin
#   data/<dataset>/queries/<stream>/stats.json
#
# Source after DATASET and QUERY_STREAM are set.

DATASET_DIR="${DATASET_DIR:-data/${DATASET}}"
DATA_PATH="${DATA_PATH:-${DATASET_DIR}/base.bin}"
if [ "${QUERY_STREAM:-query}" = "query" ]; then
  STREAM_DIR="${DATASET_DIR}"
else
  STREAM_DIR="${DATASET_DIR}/queries/${QUERY_STREAM}"
fi
QUERY_PATH="${QUERY_PATH:-${STREAM_DIR}/query.bin}"
GROUNDTRUTH_PATH="${GROUNDTRUTH_PATH:-${STREAM_DIR}/groundtruth.bin}"
