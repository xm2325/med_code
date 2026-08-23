#!/usr/bin/env bash
#SBATCH --job-name=adecoding-pt
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
umask 077

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
: "${ROIHU_PROJECT:?Set ROIHU_PROJECT to the CSC project allocation}"
: "${EXPERIMENT_LIMIT:=50}"
: "${EXPERIMENT_WORKERS:=4}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID}"
: "${EXPERIMENT_INPUT_FILE:=${EXPERIMENT_DIR}/input/adecoding_pt_all_inputs.csv}"
: "${EXPERIMENT_OUTPUT_PREFIX:=adecoding_pt}"
: "${REPO_ROOT:=${SLURM_SUBMIT_DIR}}"

MODEL_ID="Qwen/Qwen3.8-27B"
MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL_NAME="qwen3.8-27b-meddra"
DEPLOY_ROOT="${ROIHU_DEPLOY_ROOT:-/scratch/${ROIHU_PROJECT}/${USER}/med-code}"
HF_HOME="${DEPLOY_ROOT}/huggingface"
HF_HUB_CACHE="${HF_HOME}/hub"
XDG_CACHE_HOME="${DEPLOY_ROOT}/xdg-cache"
TORCHINDUCTOR_CACHE_DIR="${DEPLOY_ROOT}/torchinductor-cache"
PORT="${QWEN_PORT:-8000}"

export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module purge
module load python-vllm/0.19.1
python -c "import vllm; assert vllm.__version__ == '0.19.1', vllm.__version__"

export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME TORCHINDUCTOR_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/outputs/${EXPERIMENT_ID}"
chmod 700 "${EXPERIMENT_DIR}" "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/outputs" "${EXPERIMENT_DIR}/outputs/${EXPERIMENT_ID}"
test -f "${EXPERIMENT_INPUT_FILE}" || { echo "Missing experiment input" >&2; exit 1; }
test -f "${REPO_ROOT}/scripts/run_adecoding_pt_experiment.py" || { echo "Missing public experiment runner" >&2; exit 1; }

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}"
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

vllm serve "${MODEL_ID}" --revision "${MODEL_REVISION}" --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 --port "${PORT}" --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --generation-config vllm --gdn-prefill-backend triton \
  --disable-uvicorn-access-log \
  >"${EXPERIMENT_DIR}/logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

for attempt in $(seq 1 180); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM stopped before health check" >&2
    wait "${VLLM_PID}" || true
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if [[ "${attempt}" -eq 180 ]]; then
    echo "vLLM health timeout" >&2
    exit 1
  fi
  sleep 2
done

python "${REPO_ROOT}/scripts/run_adecoding_pt_experiment.py" \
  --input "${EXPERIMENT_INPUT_FILE}" \
  --vocabulary "${EXPERIMENT_DIR}/input/legacy_pt_vocabulary.csv" \
  --output-dir "${EXPERIMENT_DIR}/outputs/${EXPERIMENT_ID}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --limit "${EXPERIMENT_LIMIT}" --max-workers "${EXPERIMENT_WORKERS}" \
  --experiment-id "${EXPERIMENT_ID}" --output-prefix "${EXPERIMENT_OUTPUT_PREFIX}"
