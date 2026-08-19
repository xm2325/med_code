from __future__ import annotations

import json, os
from pathlib import Path
from typing import Any

from .local_llm import OpenAICompatibleQwenClient
from .real_coding import METHODS, RealCodingEngine

MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _records():
    return json.loads((Path(__file__).resolve().parents[2]/"examples"/"ra_adr_synthetic_50.json").read_text(encoding="utf-8"))


def _environment_terms():
    source=os.getenv("MEDCODE_CTCAE_XLSX")
    if not source: return None
    from .meddra_terminology import NCICTCAEV6Importer, MedDRAAsciiHierarchyImporter
    imported=NCICTCAEV6Importer(source).load(); records=imported.records
    paths=[os.getenv("MEDCODE_MEDDRA_LLT_ASC"),os.getenv("MEDCODE_MEDDRA_PT_ASC"),os.getenv("MEDCODE_MEDDRA_MDHIER_ASC")]
    if any(paths) and not all(paths): raise ValueError("all three MedDRA ASC paths are required")
    if all(paths):
        records=MedDRAAsciiHierarchyImporter(terminology_version="28.0",llt_path=paths[0],pt_path=paths[1],mdhier_path=paths[2]).enrich(records)
    return records


def create_app(engine: RealCodingEngine|None=None):
    try:
        from fastapi import FastAPI,HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc: raise RuntimeError("Install the api extra") from exc
    if engine is None:
        client=OpenAICompatibleQwenClient(base_url=os.getenv("MEDCODE_LLM_BASE_URL","http://127.0.0.1:8000/v1"),
            model=os.getenv("MEDCODE_LLM_MODEL","qwen3.8-27b-meddra"),
            model_revision=os.getenv("MEDCODE_LLM_REVISION",MODEL_REVISION),api_key=os.getenv("MEDCODE_LLM_API_KEY","local"))
        terms=_environment_terms(); engine=RealCodingEngine(client,terms) if terms is not None else RealCodingEngine(client)
    app=FastAPI(title="MedCode local Qwen API",version="1.0.0")
    demo=Path(__file__).resolve().parents[2]/"demo"

    @app.get("/",include_in_schema=False)
    def chinese_demo(): return FileResponse(demo/"ra_adr_meddra_demo.html")
    @app.get("/en",include_in_schema=False)
    def english_demo(): return FileResponse(demo/"ra_adr_meddra_demo.en.html")

    @app.get("/api/v1/health")
    @app.get("/health",include_in_schema=False)
    def health(): return {"status":"ok"}
    @app.get("/api/v1/meta")
    @app.get("/meta",include_in_schema=False)
    def meta(): return {"schema_version":"1.0","model":engine.client.model,"model_revision":engine.client.model_revision,
        "methods":list(METHODS),"meddra_version":"28.0","terminology":dict(engine.terminology),
        "terminology_sha256":engine.terminology_sha}
    @app.get("/api/v1/records")
    @app.get("/records",include_in_schema=False)
    def records(): return _records()
    @app.post("/api/v1/coding/run")
    @app.post("/coding/run",include_in_schema=False)
    def coding_run(request:dict[str,Any]):
        try:
            if request.get("schema_version")!="1.0": raise ValueError("schema_version must be 1.0")
            if request.get("meddra_version")!="28.0": raise ValueError("meddra_version must be 28.0")
            if request.get("data_classification") not in {"synthetic","public","deidentified","restricted"}:
                raise ValueError("unsupported data_classification")
            text=request.get("text")
            if not isinstance(text,str) or not 1<=len(text.strip())<=4000: raise ValueError("text length must be 1..4000")
            methods=request.get("methods",list(METHODS))
            if not isinstance(methods,list): raise ValueError("methods must be a list")
            return engine.run(text,record_id=request.get("record_id"),methods=[str(v) for v in methods],
                data_classification=str(request["data_classification"]))
        except (TypeError,ValueError) as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    return app
