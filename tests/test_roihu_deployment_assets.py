from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")
def test_assets_lock_model_and_csc_runtime():
    for relative in ("configs/roihu_qwen38.yaml", "slurm/cache_qwen38.sbatch", "slurm/serve_qwen38.sbatch", "docs/ROIHU_QWEN38_DEPLOYMENT.md"):
        text = _read(relative)
        assert "Qwen/Qwen3.8-27B" in text and REVISION in text
        assert "python-vllm/0.19.1" in text and "VLLM_IMAGE" not in text
def test_slurm_uses_full_gh200_and_noninteractive_csc_init():
    for relative in ("slurm/cache_qwen38.sbatch", "slurm/serve_qwen38.sbatch"):
        text = _read(relative)
        assert "#SBATCH --partition=gpumedium" in text and "#SBATCH --gres=gpu:gh200:1" in text
        assert "CSC_ENV_INIT_NON_INTERACTIVE=yes" in text and "source /etc/profile.d/zz-csc-env.sh" in text
        assert "set +u" in text and "set -u" in text
def test_serve_is_localhost_only_bounded_and_manages_both_processes():
    serve = _read("slurm/serve_qwen38.sbatch")
    assert 'LLM_HOST="127.0.0.1"' in serve and 'API_HOST="127.0.0.1"' in serve
    assert "0.0.0.0" not in serve and "8192|16384" in serve
    assert "--disable-uvicorn-access-log" in serve and "--enable-log-requests" not in serve
    assert "--gdn-prefill-backend triton" in serve
    assert "TORCHINDUCTOR_CACHE_DIR" in serve and "XDG_CACHE_HOME" in serve
    assert "trap cleanup EXIT INT TERM" in serve
    assert "/health" in serve and "scripts/run_real_coding_api.py" in serve
    assert "MEDCODE_CTCAE_XLSX" in serve and "4d7b4fcfdcb25c45a23b02c07fcb20eab7f284b23862e4e4e64bb8823c2f440b" in serve
    assert "HF_HUB_OFFLINE=1" in serve
def test_assets_use_user_scoped_project_scratch_without_secrets():
    for relative in ("slurm/cache_qwen38.sbatch", "slurm/serve_qwen38.sbatch"):
        text = _read(relative)
        assert '/scratch/${ROIHU_PROJECT}/${USER}/med-code' in text
        assert "hf_token=" not in text.lower() and "api_key=" not in text.lower()
def test_smoke_disables_thinking_and_validates_json():
    smoke = _read("scripts/smoke_qwen_server.py")
    assert '"enable_thinking": False' in smoke
    assert '"response_format": {"type": "json_object"}' in smoke
    assert "json.loads(result" in smoke
