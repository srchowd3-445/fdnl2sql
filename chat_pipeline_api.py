#!/usr/bin/env python3
"""HTTP bridge for chat UI -> single-question NL2SQL pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
METHOD_DIR = ROOT / "method"
ORCH_SCRIPT = METHOD_DIR / "orchestrate_single_question_chat.py"
RUNS_DIR = ROOT / "results" / "chat_api_runs"


class ChatQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    item_id: Optional[str] = ""
    skip_exec: int = 0
    preview_rows: int = 20


class ChatQueryResponse(BaseModel):
    status: str
    item_id: Optional[str] = ""
    question: str
    final_sql: str
    confidence_overall: Optional[float] = None
    error: Optional[str] = None
    decomposed_queries: list[str] = []
    retrieval_top: list[Dict[str, Any]] = []
    execution_preview: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}


class SeedFeedbackRequest(BaseModel):
    question: str = Field(min_length=1)
    predicted_sql: str = Field(min_length=1)


app = FastAPI(title="MAYO chat pipeline API", version="1.0")


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CHAT_CORS_ALLOW_ORIGINS", "*")
    if not raw.strip():
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_cmd(req: ChatQueryRequest, response_json: Path, run_tag: str) -> list[str]:
    backend = os.getenv("CHAT_BACKEND", "openai_compat")
    api_base = os.getenv("CHAT_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("CHAT_API_KEY", os.getenv("OPENAI_API_KEY", "dummy"))
    model_name = os.getenv("CHAT_MODEL_NAME", "gpt-5-nano")
    temperature = os.getenv("CHAT_TEMPERATURE", "1")
    decompose_tokens = os.getenv("CHAT_DECOMPOSE_MAX_NEW_TOKENS", "2048")
    synth_tokens = os.getenv("CHAT_SYNTH_MAX_NEW_TOKENS", "2048")
    logprob_mode = os.getenv("CHAT_LOGPROB_MODE", "none")
    logprobs = os.getenv("CHAT_LOGPROBS", "0")
    embed_backend = os.getenv("CHAT_EMBED_BACKEND", "openai")
    embed_model = os.getenv("CHAT_EMBED_MODEL", "text-embedding-3-small")
    embed_api_base = os.getenv("CHAT_EMBED_API_BASE", "")
    embed_api_key = os.getenv("CHAT_EMBED_API_KEY", api_key)
    embed_batch_size = os.getenv("CHAT_EMBED_BATCH_SIZE", "128")

    cmd = [
        sys.executable,
        str(ORCH_SCRIPT),
        "--question",
        req.question,
        "--item-id",
        req.item_id or "",
        "--backend",
        backend,
        "--api-base",
        api_base,
        f"--api-key={api_key}",
        "--model-name",
        model_name,
        "--temperature",
        str(temperature),
        "--decompose-max-new-tokens",
        str(decompose_tokens),
        "--synth-max-new-tokens",
        str(synth_tokens),
        "--logprob-mode",
        str(logprob_mode),
        "--logprobs",
        str(logprobs),
        "--embed-backend",
        str(embed_backend),
        "--embed-model",
        str(embed_model),
        "--embed-batch-size",
        str(embed_batch_size),
        "--skip-exec",
        str(int(req.skip_exec)),
        "--preview-rows",
        str(int(req.preview_rows)),
        "--run-eval",
        "0",
        "--output-dir",
        str(RUNS_DIR),
        "--run-tag",
        run_tag,
        "--output-json",
        str(response_json),
        "--keep-artifacts",
        os.getenv("CHAT_KEEP_ARTIFACTS", "1"),
    ]

    # Optional candidate source overrides.
    candidate_json = os.getenv("CHAT_CANDIDATE_JSON", "").strip()
    candidate_sqlite = os.getenv("CHAT_CANDIDATE_SQLITE", "").strip()
    seed_json = os.getenv("CHAT_SEED_JSON", "").strip()
    if embed_api_base:
        cmd.extend(["--embed-api-base", embed_api_base])
    if embed_api_key:
        cmd.append(f"--embed-api-key={embed_api_key}")

    if candidate_sqlite:
        cmd.extend(["--candidate-sqlite", candidate_sqlite])
    elif candidate_json:
        cmd.extend(["--candidate-json", candidate_json])
    elif seed_json:
        cmd.extend(["--seed-json", seed_json])

    return cmd


def _append_seed_entry(path: Path, question: str, predicted_sql: str) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Seed file not found: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise RuntimeError(f"Seed file must be a JSON list: {path}")

    q = question.strip()
    s = predicted_sql.strip()
    if not q or not s:
        raise RuntimeError("Both question and predicted_sql are required.")

    q_norm = q.lower()
    for row in obj:
        if not isinstance(row, dict):
            continue
        oq = str(row.get("original_question") or "").strip().lower()
        if oq == q_norm:
            return {"appended": False, "reason": "duplicate_question", "path": str(path)}

    obj.append(
        {
            "original_question": q,
            "gt_sql": s,
            "decomposed_query": {
                "query_1": {
                    "question": q,
                    "sql": s,
                }
            },
        }
    )
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"appended": True, "reason": "added", "path": str(path)}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "orchestrator_script": str(ORCH_SCRIPT),
        "orchestrator_exists": ORCH_SCRIPT.exists(),
        "runs_dir": str(RUNS_DIR),
    }


@app.post("/api/chat-query", response_model=ChatQueryResponse)
def chat_query(req: ChatQueryRequest) -> Dict[str, Any]:
    if not ORCH_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Missing orchestrator script: {ORCH_SCRIPT}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    response_json = RUNS_DIR / f"{run_tag}.response.json"
    cmd = _build_cmd(req, response_json, run_tag)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = {
            "message": "Pipeline failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "cmd": cmd,
        }
        raise HTTPException(status_code=500, detail=err)

    if not response_json.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Pipeline completed but response JSON was not produced",
                "cmd": cmd,
                "stdout_tail": proc.stdout[-4000:],
            },
        )

    try:
        return json.loads(response_json.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to parse response JSON: {exc}") from exc


@app.post("/api/seed-feedback")
def seed_feedback(req: SeedFeedbackRequest) -> Dict[str, Any]:
    targets = [
        ROOT / "data" / "seed_questions.json",
        ROOT / "frontend" / "public" / "seed_questions.json",
    ]
    out = []
    for p in targets:
        out.append(_append_seed_entry(p, req.question, req.predicted_sql))
    return {"ok": True, "results": out}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CHAT_API_HOST", "127.0.0.1")
    port = int(os.getenv("CHAT_API_PORT", "8181"))
    uvicorn.run(app, host=host, port=port)