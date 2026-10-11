#!/bin/bash
# TriviaQA simZipf+ : Mistral rephrases plus the encoder that matches Aker's queries.
# Submit from a Clariden login node:
#   sbatch /iopsstor/scratch/cscs/$USER/triviaqa-simzipf/run_triviaqa_simzipf_plus.sbatch
#SBATCH --job-name=triviaqa_simplus
#SBATCH --account=lsaiews26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/triviaqa-simzipf/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/triviaqa-simzipf/logs/%x_%j.err

set -eo pipefail

ROOT=/iopsstor/scratch/cscs/${USER}/triviaqa-simzipf
export HF_HOME=${ROOT}/hf
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=0

echo "[sbatch] host=$(hostname) root=${ROOT}"
mkdir -p "${ROOT}/logs" "${ROOT}/out" "${HF_HOME}"

# The NGC PyTorch image has torch but not transformers. Install it once
# into a venv on scratch. The login node cannot run that venv, so the
# readiness check is a stamp written inside the container.
VENV=${ROOT}/venv
if [[ ! -f ${VENV}/.transformers_ready ]]; then
  srun --mpi=pmix --network=disable_rdzv_get --environment=alps \
    bash -c "python -m venv --system-site-packages '${VENV}' && \
      '${VENV}/bin/python' -m pip install --disable-pip-version-check \
        'transformers>=4.46' sentencepiece protobuf && \
      '${VENV}/bin/python' -c 'import transformers' && \
      touch '${VENV}/.transformers_ready'"
fi

# Regular CDN. The Xet bridge has been dropping long downloads.
export HF_HUB_DISABLE_XET=1

srun --mpi=pmix --network=disable_rdzv_get --environment=alps \
  "${VENV}/bin/python" "${ROOT}/build_triviaqa_simzipf_plus.py" \
    --questions "${ROOT}/triviaqa.txt" \
    --official "${ROOT}/triviaqa-sim-100k-0.99.npy" \
    --out-dir "${ROOT}/out" \
    --n-stream 100000 \
    --seed 42 \
    --skew 0.3 0.6 0.99 \
    --batch-size 64 \
    --embed-batch-size 256

echo "[sbatch] finished"
