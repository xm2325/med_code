#!/usr/bin/env bash
#SBATCH --job-name=qwen38-cache
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
MODEL_ID="Qwen/Qwen3.8-27B"
MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
: "${ROIHU_PROJECT:?Set ROIHU_PROJECT to the CSC project identifier.}"
case "${ROIHU_PROJECT}" in project_[0-9]*) ;; *) echo "Invalid ROIHU_PROJECT" >&2; exit 2;; esac
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
set -u
module purge
module load python-vllm/0.19.1
DEPLOY_ROOT="/scratch/${ROIHU_PROJECT}/${USER}/med-code"
HF_HOME="${DEPLOY_ROOT}/huggingface"
HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HOME HF_HUB_CACHE
mkdir -p "${HF_HUB_CACHE}" "${DEPLOY_ROOT}/logs"
python -c "import vllm; assert vllm.__version__ == '0.19.1', vllm.__version__"
python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3])' \
  "${MODEL_ID}" "${MODEL_REVISION}" "${HF_HUB_CACHE}"
echo "Cached ${MODEL_ID}@${MODEL_REVISION} under ${HF_HUB_CACHE}"
