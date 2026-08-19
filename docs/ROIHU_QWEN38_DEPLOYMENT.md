# Roihu live deployment: Qwen3.8-27B and coding API

This runbook uses CSC's supported `python-vllm/0.19.1` module on an ARM/GH200
compute node; no custom container is required.

## Locked contract

- `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- CSC module `python-vllm/0.19.1`
- one full GH200 via `gpumedium` and `gpu:gh200:1`
- cache `/scratch/<project>/<user>/med-code/huggingface`
- vLLM/Torch compile caches under the same project scratch root
- vLLM `127.0.0.1:8000`; coding API `127.0.0.1:8010`
- 8,192 context initially; 16,384 only after a measured smoke run

Both jobs initialize CSC non-interactively:

```bash
export CSC_ENV_INIT_NON_INTERACTIVE=yes
source /etc/profile.d/zz-csc-env.sh
module purge
module load python-vllm/0.19.1
```

Set the project and cache the exact public snapshot:

```bash
export ROIHU_PROJECT=project_200xxxx
cd /scratch/$ROIHU_PROJECT/$USER/med-code/repo
sbatch --account="$ROIHU_PROJECT" slurm/cache_qwen38.sbatch
```

No credentials belong in the repository or sbatch scripts. Scratch is a
reproducible cache, not archival storage. Start the live stack with:

```bash
sbatch --account="$ROIHU_PROJECT" slurm/serve_qwen38.sbatch
```

The job backgrounds vLLM, waits for `/health`, and runs
`scripts/run_real_coding_api.py` in the foreground. A trap terminates vLLM when
the API exits. vLLM 0.19.1 is started without `--enable-log-requests` and with
`--disable-uvicorn-access-log`; the FastAPI runner also disables its access log.
The GDN prefill backend is pinned to Triton to avoid first-run FlashInfer JIT
stalling this short-lived demo service.
Submit both jobs from the checked-out repository root: the serve
script validates `$SLURM_SUBMIT_DIR`, the NCI workbook fingerprint, and imports
`fastapi`/`uvicorn` before allocating the listeners. Request logging is disabled
and both listeners are localhost only. The expected public workbook path is
`/scratch/<project>/<user>/med-code/terminology/ctcae-v6.0.xlsx`. A controlled
16K run uses:

```bash
QWEN_MAX_MODEL_LEN=16384 sbatch --account="$ROIHU_PROJECT" slurm/serve_qwen38.sbatch
```

Do not expose `0.0.0.0`, silently quantize, or change the model revision.
Forward compute-node port 8010 through the Roihu login host using CSC's current
documented multi-hop SSH method. Then check locally:

```bash
curl --fail http://127.0.0.1:8010/health
curl --fail http://127.0.0.1:8010/meta
```

For direct vLLM troubleshooting, forward port 8000 and run:

```bash
python scripts/smoke_qwen_server.py --base-url http://127.0.0.1:8000/v1
```

Confirm model revision, module version, one-GPU allocation, terminology
fingerprint, peak GPU memory and synthetic latency before real coding. Clinical
text must be de-identified and follow the approved project data policy.
