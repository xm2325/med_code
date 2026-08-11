#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_codiesp_icd_kb_ab as base

SCHEMA_VERSION = "codiesp-icd-kb-ab-resilient-v0.2"
TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}

PRED_COLUMNS = [
    "article_id", "mode", "gold", "pred", "accepted_json",
    "retrieved_decisions_json", "added_outside_retrieval_json",
    "retrieved_codes", "retrieval_gold_recall", "tp", "fp", "fn",
    "precision", "recall", "f1", "exact", "prompt_tokens",
    "completion_tokens", "api_attempts",
]
FAIL_COLUMNS = [
    "article_id", "mode", "error_type", "error", "attempt_log_json"
]
RETR_COLUMNS = ["article_id", "retrieved_json"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(text: str, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def call_deepseek_resilient(prompt: str, api_key: str, max_attempts: int = 4):
    body = json.dumps({
        "model": base.MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Return clinically grounded ICD-10 diagnosis coding as strict JSON. Think deeply but expose only concise auditable conclusions.",
            },
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "response_format": {"type": "json_object"},
        "max_tokens": 12000,
    }).encode("utf-8")

    attempt_log = []
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as response:
                raw = response.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            obj = json.loads(content)
            if not isinstance(obj, dict):
                raise ValueError("DeepSeek JSON response was not an object")
            attempt_log.append({
                "attempt": attempt,
                "status": "success",
                "elapsed_seconds": round(time.time() - started, 3),
            })
            return obj, payload.get("usage", {}), attempt_log

        except urllib.error.HTTPError as exc:
            last_exc = exc
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            retryable = exc.code in TRANSIENT_HTTP
            attempt_log.append({
                "attempt": attempt,
                "status": "http_error",
                "http_status": exc.code,
                "retryable": retryable,
                "message": _short(err_body or str(exc)),
                "elapsed_seconds": round(time.time() - started, 3),
            })
            if not retryable:
                break

        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_exc = exc
            attempt_log.append({
                "attempt": attempt,
                "status": "network_or_timeout",
                "retryable": True,
                "message": _short(str(exc)),
                "elapsed_seconds": round(time.time() - started, 3),
            })

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_exc = exc
            attempt_log.append({
                "attempt": attempt,
                "status": "response_parse_error",
                "retryable": True,
                "message": _short(str(exc)),
                "elapsed_seconds": round(time.time() - started, 3),
            })

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            attempt_log.append({
                "attempt": attempt,
                "status": "unexpected_error",
                "retryable": True,
                "message": _short(f"{type(exc).__name__}: {exc}"),
                "elapsed_seconds": round(time.time() - started, 3),
            })

        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 8))

    summary = "; ".join(
        f"attempt {x['attempt']}={x['status']}:{x.get('http_status', '')}:{x.get('message', '')}"
        for x in attempt_log
    )
    raise RuntimeError(
        f"DeepSeek request failed after {len(attempt_log)} attempt(s). {summary}. "
        f"Last exception={type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}"
    )


def write_checkpoint(
    out: Path,
    rows: list[dict],
    failures: list[dict],
    retrieved_rows: list[dict],
    manifest_base: dict,
    ids: list[str],
) -> None:
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(failures, columns=FAIL_COLUMNS).to_csv(out / "failures.csv", index=False)
    pd.DataFrame(retrieved_rows, columns=RETR_COLUMNS).to_csv(out / "retrieved_knowledge.csv", index=False)

    completed = [r["article_id"] for r in rows]
    failed = [r["article_id"] for r in failures]
    pending = [x for x in ids if x not in set(completed) | set(failed)]
    manifest = {
        **manifest_base,
        "checkpointed_at": _now(),
        "case_ids": ids,
        "completed_case_ids": completed,
        "failed_case_ids": failed,
        "pending_case_ids": pending,
        "n_success": len(rows),
        "n_failures": len(failures),
        "n_pending": len(pending),
        "complete": len(pending) == 0 and len(failures) == 0,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "rag"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--total-cases", type=int, default=50)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--retrieval-top-k", type=int, default=35)
    ap.add_argument("--max-api-attempts", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    texts, gold, corpus_manifest = base.load_codiesp_test()
    all_ids = sorted(set(texts) & set(gold))[: args.total_cases]
    ids = all_ids[args.shard_index :: args.num_shards]
    if not ids:
        raise RuntimeError("Empty shard")

    retriever = None
    kb_manifest = None
    if args.mode == "rag":
        kb, kb_manifest = base.build_kb()
        retriever = base.Retriever(kb)
        (out / "icd_kb_manifest.json").write_text(
            json.dumps(kb_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    prompt_text = (
        base.direct_prompt("[CLINICAL_TEXT]")
        if args.mode == "direct"
        else base.rag_prompt("[CLINICAL_TEXT]", [{"dynamic": "retrieved per case"}])
    )
    (out / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    manifest_base = {
        "schema_version": SCHEMA_VERSION,
        "base_runner_schema_version": base.SCHEMA_VERSION,
        "mode": args.mode,
        "model": base.MODEL,
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
        "prompt_version": base.DIRECT_PROMPT_VERSION if args.mode == "direct" else base.RAG_PROMPT_VERSION,
        "corpus": corpus_manifest,
        "icd_knowledge": kb_manifest,
        "retrieval": None if args.mode == "direct" else {
            "top_k": args.retrieval_top_k,
            "method": "hybrid TF-IDF over official descriptions + hierarchy + XML coding notes; full-document and sentence-level similarity",
            "weights": {"full_word": 0.25, "full_char": 0.10, "sentence_word_max": 0.45, "sentence_char_max": 0.20},
            "candidate_restriction": False,
        },
        "total_cases_protocol": args.total_cases,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_api_attempts": args.max_api_attempts,
        "checkpoint_after_each_case": True,
    }

    rows: list[dict] = []
    failures: list[dict] = []
    retrieved_rows: list[dict] = []
    write_checkpoint(out, rows, failures, retrieved_rows, manifest_base, ids)

    for pos, doc_id in enumerate(ids, start=1):
        retrieved = retriever.retrieve(texts[doc_id], args.retrieval_top_k) if retriever is not None else []
        if retrieved:
            retrieved_rows.append({
                "article_id": doc_id,
                "retrieved_json": json.dumps(retrieved, ensure_ascii=False),
            })
        prompt = base.direct_prompt(texts[doc_id]) if args.mode == "direct" else base.rag_prompt(texts[doc_id], retrieved)

        try:
            obj, usage, attempt_log = call_deepseek_resilient(
                prompt, api_key, max_attempts=args.max_api_attempts
            )
            pred = {base.norm_code(c) for c in obj.get("codes", []) if base.norm_code(c)}
            g = gold[doc_id]
            metrics = base.score_one(g, pred)
            accepted = obj.get("accepted", []) if isinstance(obj.get("accepted", []), list) else []
            retrieved_decisions = obj.get("retrieved_decisions", []) if isinstance(obj.get("retrieved_decisions", []), list) else []
            outside = obj.get("added_outside_retrieval", []) if isinstance(obj.get("added_outside_retrieval", []), list) else []
            rows.append({
                "article_id": doc_id,
                "mode": args.mode,
                "gold": "|".join(sorted(g)),
                "pred": "|".join(sorted(pred)),
                "accepted_json": json.dumps(accepted, ensure_ascii=False),
                "retrieved_decisions_json": json.dumps(retrieved_decisions, ensure_ascii=False),
                "added_outside_retrieval_json": json.dumps(outside, ensure_ascii=False),
                "retrieved_codes": "|".join(r["code"] for r in retrieved),
                "retrieval_gold_recall": len(g & {r["code"] for r in retrieved}) / len(g) if g else 0.0,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact": metrics["exact"],
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "api_attempts": len(attempt_log),
            })
            status = "success"
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "article_id": doc_id,
                "mode": args.mode,
                "error_type": type(exc).__name__,
                "error": _short(str(exc), 3000),
                "attempt_log_json": json.dumps([], ensure_ascii=False),
            })
            status = "failure"

        write_checkpoint(out, rows, failures, retrieved_rows, manifest_base, ids)
        print(
            f"{args.mode} shard={args.shard_index} {pos}/{len(ids)} article={doc_id} status={status} "
            f"success={len(rows)} failures={len(failures)}",
            flush=True,
        )

    if failures:
        raise RuntimeError(
            f"Shard completed with partial failures: {len(rows)} success, {len(failures)} failure(s). "
            "Checkpoint files were preserved for artifact upload."
        )


if __name__ == "__main__":
    main()
