#!/usr/bin/env bash
#SBATCH --job-name=med-code-live
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
umask 077
MODEL_ID="Qwen/Qwen3.8-27B"
MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL_NAME="qwen3.8-27b-meddra"
LLM_HOST="127.0.0.1"
LLM_PORT="${QWEN_PORT:-8000}"
API_HOST="127.0.0.1"
API_PORT="${CODING_API_PORT:-8010}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
: "${ROIHU_PROJECT:?Set ROIHU_PROJECT to the CSC project identifier.}"
case "${ROIHU_PROJECT}" in project_[0-9]*) ;; *) echo "Invalid ROIHU_PROJECT" >&2; exit 2;; esac
case "${MAX_MODEL_LEN}" in 8192|16384) ;; *) echo "Context must be 8192 or 16384" >&2; exit 2;; esac
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module purge
module load python-vllm/0.19.1
python -c "import vllm; assert vllm.__version__ == '0.19.1', vllm.__version__"
DEPLOY_ROOT="/scratch/${ROIHU_PROJECT}/${USER}/med-code"
HF_HOME="${DEPLOY_ROOT}/huggingface"
HF_HUB_CACHE="${HF_HOME}/hub"
XDG_CACHE_HOME="${DEPLOY_ROOT}/xdg-cache"
TORCHINDUCTOR_CACHE_DIR="${DEPLOY_ROOT}/torchinductor-cache"
CTCAE_XLSX="${DEPLOY_ROOT}/terminology/ctcae-v6.0.xlsx"
CTCAE_SHA256="4d7b4fcfdcb25c45a23b02c07fcb20eab7f284b23862e4e4e64bb8823c2f440b"
test -f "${CTCAE_XLSX}" || { echo "Missing ${CTCAE_XLSX}" >&2; exit 1; }
test "$(sha256sum "${CTCAE_XLSX}" | cut -d' ' -f1)" = "${CTCAE_SHA256}" || { echo "CTCAE workbook hash mismatch" >&2; exit 1; }
export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME TORCHINDUCTOR_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEDCODE_CTCAE_XLSX="${CTCAE_XLSX}"
mkdir -p "${DEPLOY_ROOT}/logs"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the checked-out med_code repository.}"
test -f "${REPO_ROOT}/scripts/run_real_coding_api.py" || { echo "Submit from the med_code repository root" >&2; exit 1; }
cd "${REPO_ROOT}"
python -c "import fastapi, uvicorn; print('FastAPI', fastapi.__version__, 'uvicorn', uvicorn.__version__)"
VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}"
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
vllm serve "${MODEL_ID}" --revision "${MODEL_REVISION}" --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${LLM_HOST}" --port "${LLM_PORT}" --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.88 --generation-config vllm --gdn-prefill-backend triton \
  --disable-uvicorn-access-log \
  >"${DEPLOY_ROOT}/logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
for attempt in $(seq 1 180); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then echo "vLLM stopped before becoming healthy" >&2; wait "${VLLM_PID}" || true; exit 1; fi
  if curl --fail --silent "http://${LLM_HOST}:${LLM_PORT}/health" >/dev/null; then break; fi
  if [[ "${attempt}" -eq 180 ]]; then echo "Timed out waiting for vLLM health" >&2; exit 1; fi
  sleep 2
done
python scripts/run_real_coding_api.py --host "${API_HOST}" --port "${API_PORT}" \
  --llm-base-url "http://${LLM_HOST}:${LLM_PORT}/v1"
