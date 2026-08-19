#!/usr/bin/env python3
"""Lightweight API runner that avoids importing cohortcoder's eager package __init__."""
import argparse, importlib.util, os, sys, types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; PKG=ROOT/"src"/"cohortcoder"
package=types.ModuleType("cohortcoder"); package.__path__=[str(PKG)]; sys.modules["cohortcoder"]=package
def load(name):
    spec=importlib.util.spec_from_file_location(f"cohortcoder.{name}",PKG/f"{name}.py"); module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module; spec.loader.exec_module(module); return module
for dependency in ("clinical_context","adr_coding","local_llm","meddra_terminology","real_coding"):
    load(dependency)
api=load("api")

parser=argparse.ArgumentParser(); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=8010)
parser.add_argument("--llm-base-url",default="http://127.0.0.1:8000/v1"); parser.add_argument("--model",default="qwen3.8-27b-meddra")
parser.add_argument("--revision",default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
args=parser.parse_args(); os.environ["MEDCODE_LLM_BASE_URL"]=args.llm_base_url; os.environ["MEDCODE_LLM_MODEL"]=args.model; os.environ["MEDCODE_LLM_REVISION"]=args.revision
import uvicorn
uvicorn.run(api.create_app(),host=args.host,port=args.port,access_log=False)
