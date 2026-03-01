#!/usr/bin/env python3
"""Phase-based decompose -> SBERT retrieve -> synthesize -> eval -> seed-grow pipeline.

This script processes questions in phases (default: 3 phases x 500 questions):
1) Decompose each natural-language question with Gemma/OpenAI-compatible model
2) Retrieve top-k seed candidates using SBERT on decomposed queries
3) Synthesize final SQL using retrieved candidates
4) Evaluate phase output with eval_run_baselines_v2.py
5) Grow seed file using configurable acceptance rules from a .txt file

Outputs per phase are saved as:
- 500_phase1.json, 500_phase2.json, 500_phase3.json (default naming)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import retrieve_similar_queries as rsq
except Exception:
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import retrieve_similar_queries as rsq


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    default_out = root / "results" / "eval_baselines_v3" / "20260226_062742" / "decompose_method"

    ap = argparse.ArgumentParser(
        description=(
            "Run phased decomposition + SBERT retrieval + SQL synthesis + eval + adaptive seed growth."
        )
    )

    # Question source and phase controls
    ap.add_argument("--question-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--id-key", default="item_id")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--total-questions", type=int, default=1500)
    ap.add_argument("--phase-size", type=int, default=500)

    # Candidate/seed sources
    ap.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    ap.add_argument("--candidate-json", default="", help="Optional override for initial candidates")

    # Prompt context
    ap.add_argument("--prompt-file", default=str(root / "prompting" / "zero_shot_sql_expert.txt"))
    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument(
        "--decompose-examples-file",
        default=str(root / "prompting" / "decompose_good_bad_examples.txt"),
        help="Optional good/bad decomposition examples text file",
    )

    # Retrieval controls
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--retrieval-per-decomp", type=int, default=5)
    ap.add_argument("--max-decomposed-queries", type=int, default=5)
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-device", default=None)
    ap.add_argument("--sbert-batch-size", type=int, default=64)

    # Backend controls
    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")

    # OpenAI-compatible backend args
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="gemma-3-27b-local")
    ap.add_argument(
        "--decompose-model-name",
        default="",
        help="Optional model override for decomposition call when backend=openai_compat",
    )
    ap.add_argument(
        "--synthesis-model-name",
        default="",
        help="Optional model override for synthesis call when backend=openai_compat",
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)
    ap.add_argument("--logprobs", type=int, default=0)

    # Local vLLM backend args
    ap.add_argument(
        "--model-path",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/"
            "snapshots/005ad3404e59d6023443cb575daa05336842228a"
        ),
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)

    # Shared generation args
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--decompose-max-new-tokens", type=int, default=None)
    ap.add_argument("--synthesis-max-new-tokens", type=int, default=None)
    ap.add_argument("--decompose-temperature", type=float, default=None)
    ap.add_argument("--synthesis-temperature", type=float, default=None)
    ap.add_argument("--decompose-top-p", type=float, default=None)
    ap.add_argument("--synthesis-top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)

    # Throughput controls
    ap.add_argument("--gen-batch-size", type=int, default=32, help="Questions dispatched per chunk")
    ap.add_argument("--batch-concurrency", type=int, default=8, help="Concurrent question workers for openai_compat")

    # SQL preview
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=1)

    # Eval + acceptance config
    ap.add_argument("--run-eval", type=int, default=1)
    ap.add_argument("--eval-script", default=str(root / "eval_run_baselines_v2.py"))
    ap.add_argument("--compute-bertscore", type=int, default=0)
    ap.add_argument("--ast-weight-select", type=float, default=0.5)
    ap.add_argument("--ast-weight-where", type=float, default=0.4)
    ap.add_argument("--ast-weight-from", type=float, default=0.1)
    ap.add_argument(
        "--acceptance-config",
        default=str(root / "method" / "decompose_method" / "seed_acceptance_rules.txt"),
    )

    # Output
    ap.add_argument("--output-dir", default=str(default_out))

    return ap.parse_args()


def load_template_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_schema_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""

    raw = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return raw.strip()

    return json.dumps(obj, ensure_ascii=False, indent=2)


def render_base_prompt(base_prompt: str, schema_text: str, question: str) -> str:
    q = (question or "").strip()

    if base_prompt.strip():
        prompt = base_prompt
    else:
        prompt = (
            "You are an expert natural language to SQL generator. "
            "Return only one valid SQLite SQL statement ending with semicolon.\n\n"
            "Schema: {SCHEMA}\n\n"
            "Question: {QUESTION}\nGenerate SQL:"
        )

    if "{SCHEMA}" in prompt:
        prompt = prompt.replace("{SCHEMA}", schema_text)
    elif schema_text:
        prompt = f"{prompt}\n\nSchema:\n{schema_text}"

    if "{QUESTION}" in prompt:
        prompt = prompt.replace("{QUESTION}", q)
    elif "{Question}" in prompt:
        prompt = prompt.replace("{Question}", q)
    else:
        prompt = f"{prompt}\n\nQuestion: {q}\nGenerate SQL:"

    return prompt


def _is_retryable_provider_error(exc: Exception) -> bool:
    s = str(exc or "").lower()
    markers = [
        "rate limit",
        "too many requests",
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "429",
        "503",
        "service unavailable",
        "temporarily unavailable",
        "deadline exceeded",
        "timed out",
        "connection reset",
        "connection error",
    ]
    return any(m in s for m in markers)


def _parse_openai_text(completion: Any) -> str:
    choices = getattr(completion, "choices", []) or []
    c0 = choices[0] if choices else None
    if c0 is None:
        return ""
    message = getattr(c0, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if t is not None:
                    parts.append(str(t))
            else:
                t = getattr(item, "text", None)
                if t is not None:
                    parts.append(str(t))
        return "".join(parts).strip()
    return str(content or "").strip()


def _openai_confidence_from_logprobs(completion: Any) -> Optional[float]:
    try:
        choices = getattr(completion, "choices", []) or []
        c0 = choices[0] if choices else None
        if c0 is None:
            return None
        c0_logprobs = getattr(c0, "logprobs", None)
        content = getattr(c0_logprobs, "content", None) if c0_logprobs is not None else None
        if not isinstance(content, list):
            return None

        vals: List[float] = []
        for tok in content:
            lp = getattr(tok, "logprob", None)
            if isinstance(lp, (int, float)):
                vals.append(float(lp))
        if not vals:
            return None
        return float(math.exp(sum(vals) / len(vals)))
    except Exception:  # noqa: BLE001
        return None


def extract_sql(text: str) -> str:
    if not text:
        return ""

    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\b(SELECT|WITH)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start() :].strip()

    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"

    return t


def load_questions(path: str) -> List[Dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list at {path}")
    return [x for x in obj if isinstance(x, dict)]


def extract_question_from_row(row: Dict[str, Any], key: str) -> Tuple[str, Optional[str]]:
    q = row.get(key)
    if isinstance(q, str) and q.strip():
        return q.strip(), None

    for alt in ("natural_question", "question", "original_question"):
        alt_q = row.get(alt)
        if isinstance(alt_q, str) and alt_q.strip():
            return alt_q.strip(), None

    return "", f"MISSING_QUESTION_KEY:{key}"


def build_phase_tasks(
    rows: Sequence[Dict[str, Any]],
    question_key: str,
    id_key: str,
    global_start_index: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        q, err = extract_question_from_row(row, question_key)
        tasks.append(
            {
                "task_index": i,
                "global_index": global_start_index + i,
                "item_id": row.get(id_key),
                "question": q,
                "error": err,
                "raw_row": row,
            }
        )
    return tasks


def load_seed_rows(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Seed JSON must be a list: {path}")
    return [x for x in obj if isinstance(x, dict)]


def default_good_bad_examples(seed_rows: Sequence[Dict[str, Any]]) -> str:
    good_orig = "Which melanoma phase 3 trials used Pembrolizumab?"
    good_decomp = [
        "Which trials have Cancer type = Melanoma?",
        "Which trials have Trial phase = Phase 3?",
        "Which trials have Name of ICI = Pembrolizumab?",
    ]

    for row in seed_rows:
        dq = row.get("decomposed_query")
        oq = row.get("original_question") or row.get("question")
        if isinstance(dq, dict) and isinstance(oq, str) and oq.strip():
            parts: List[str] = []
            for qk in sorted(dq):
                one = dq.get(qk)
                if isinstance(one, dict):
                    q = one.get("question")
                    if isinstance(q, str) and q.strip():
                        parts.append(q.strip())
                if len(parts) >= 3:
                    break
            if parts:
                good_orig = oq.strip()
                good_decomp = parts
                break

    good_block = "\n".join([f"- {x}" for x in good_decomp])

    bad_block = "\n".join(
        [
            "- Find trials from the table.",
            "- Write SQL for this question.",
            "- Show me the answer.",
        ]
    )

    return (
        "Good decomposition example:\n"
        f"Original question: {good_orig}\n"
        "Decomposed queries:\n"
        f"{good_block}\n\n"
        "Bad decomposition example:\n"
        "Original question: Which colorectal trials had 3 or more arms and used multikinase inhibitor as control?\n"
        "Bad decomposed queries (too vague / not retrieval-friendly):\n"
        f"{bad_block}\n"
    )


def load_decomposition_examples(path: str, seed_rows: Sequence[Dict[str, Any]]) -> str:
    p = Path(path)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return default_good_bad_examples(seed_rows)


def build_decomposition_messages(
    *,
    question: str,
    base_prompt: str,
    schema_text: str,
    examples_text: str,
    max_queries: int,
) -> List[Dict[str, str]]:
    base = render_base_prompt(base_prompt=base_prompt, schema_text=schema_text, question=question)

    user = (
        f"{base}\n\n"
        "You are now performing query decomposition, not final SQL generation.\n"
        "Decompose the original question into short, retrieval-friendly atomic sub-questions.\n"
        "Each sub-question should map to one key constraint/entity.\n"
        f"Return between 1 and {max(1, int(max_queries))} sub-questions.\n\n"
        f"{examples_text}\n\n"
        "Output format requirements:\n"
        "- Output ONLY JSON\n"
        "- Use this exact schema: {\"decomposed_queries\": [\"...\", \"...\"]}\n"
        "- Do NOT include SQL, markdown, or explanation\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert clinical-trials SQL analyst and query decomposition specialist.",
        },
        {"role": "user", "content": user},
    ]


def _clean_decomposed_items(items: Sequence[Any], max_items: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = str(x or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def parse_decomposition(raw_text: str, max_items: int) -> List[str]:
    t = (raw_text or "").strip()
    if not t:
        return []

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            dq = obj.get("decomposed_queries")
            if isinstance(dq, list):
                cleaned = _clean_decomposed_items(dq, max_items=max_items)
                if cleaned:
                    return cleaned

            d1 = obj.get("decomposed_query")
            if isinstance(d1, str) and d1.strip():
                return _clean_decomposed_items([d1], max_items=max_items)
    except Exception:  # noqa: BLE001
        pass

    lines: List[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^[-*]\s+", "", s)
        s = re.sub(r"^\d+[\.)]\s+", "", s)
        s = s.strip()
        if s:
            lines.append(s)

    return _clean_decomposed_items(lines, max_items=max_items)


def match_score(match_obj: Any) -> float:
    if hasattr(match_obj, "sbert_score"):
        try:
            return float(getattr(match_obj, "sbert_score"))
        except Exception:  # noqa: BLE001
            pass
    try:
        return float(getattr(match_obj, "total_score"))
    except Exception:  # noqa: BLE001
        return 0.0


def build_match_result(rank: int, score: float, candidate: Any) -> Any:
    try:
        return rsq.MatchResult(
            rank=int(rank),
            total_score=float(score),
            sbert_score=float(score),
            candidate=candidate,
        )
    except TypeError:
        return rsq.MatchResult(
            rank=int(rank),
            total_score=float(score),
            lexical_score=float(score),
            char_score=0.0,
            literal_score=0.0,
            operator_score=0.0,
            column_score=0.0,
            candidate=candidate,
        )


def retrieve_from_decomposition(
    *,
    candidates: Sequence[Any],
    decomposed_queries: Sequence[str],
    top_k: int,
    per_query_k: int,
    sbert_model: str,
    sbert_device: Optional[str],
    sbert_batch_size: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    traces: List[Dict[str, Any]] = []
    best_by_candidate_id: Dict[str, Tuple[float, Any, str]] = {}

    parts = [q for q in decomposed_queries if (q or "").strip()]
    if not parts:
        return [], traces

    for part in parts:
        ranked_part = rsq.rank_candidates(
            question=part,
            candidates=candidates,
            top_k=max(1, int(per_query_k)),
            sbert_model=sbert_model,
            sbert_device=sbert_device,
            sbert_batch_size=sbert_batch_size,
        )

        top1 = ranked_part[0] if ranked_part else None
        traces.append(
            {
                "decomposed_query": part,
                "top1_candidate_id": top1.candidate.candidate_id if top1 else None,
                "top1_score": match_score(top1) if top1 is not None else None,
            }
        )

        for m in ranked_part:
            cid = str(m.candidate.candidate_id)
            sc = match_score(m)
            prev = best_by_candidate_id.get(cid)
            if prev is None or sc > prev[0]:
                best_by_candidate_id[cid] = (sc, m.candidate, part)

    merged = sorted(best_by_candidate_id.values(), key=lambda x: x[0], reverse=True)
    top = merged[: max(1, int(top_k))]

    ranked_out: List[Any] = []
    for i, (score, cand, _part) in enumerate(top, start=1):
        ranked_out.append(build_match_result(rank=i, score=score, candidate=cand))

    return ranked_out, traces


def build_synthesis_messages(
    *,
    question: str,
    decomposed_queries: Sequence[str],
    ranked: Sequence[Any],
    base_prompt: str,
    schema_text: str,
) -> List[Dict[str, str]]:
    base = render_base_prompt(base_prompt=base_prompt, schema_text=schema_text, question=question)

    decomp_block = "\n".join([f"- {q}" for q in decomposed_queries]) if decomposed_queries else "- (none)"

    candidate_blocks: List[str] = []
    for m in ranked:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate Rank {m.rank}",
                    f"SBERT Score: {match_score(m):.4f}",
                    f"Candidate Question: {m.candidate.question}",
                    "Candidate SQL:",
                    m.candidate.sql,
                ]
            )
        )

    user = (
        f"{base}\n\n"
        "Decomposed sub-questions used for retrieval:\n"
        f"{decomp_block}\n\n"
        "Top retrieved SBERT examples:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        "Synthesize one best final SQL query for the original question.\n"
        "Output only SQL (no markdown, no explanation).\n"
    )

    return [
        {"role": "system", "content": "You are an expert SQLite NL2SQL assistant."},
        {"role": "user", "content": user},
    ]


def serialize_match(m: Any) -> Dict[str, Any]:
    d = asdict(m)
    if hasattr(m, "candidate"):
        d["candidate"] = asdict(m.candidate)
    return d


class ModelBackend:
    def generate(self, messages: List[Dict[str, str]], stage: str) -> Dict[str, Any]:
        raise NotImplementedError


def is_decomposition_stage(stage: str) -> bool:
    return str(stage or "").strip().lower().startswith("decomp")


def stage_model_name(args: argparse.Namespace, stage: str) -> str:
    if is_decomposition_stage(stage):
        v = str(getattr(args, "decompose_model_name", "") or "").strip()
        if v:
            return v
    else:
        v = str(getattr(args, "synthesis_model_name", "") or "").strip()
        if v:
            return v
    return str(getattr(args, "model_name"))


def stage_param(args: argparse.Namespace, stage: str, decompose_attr: str, synth_attr: str, fallback_attr: str) -> Any:
    attr = decompose_attr if is_decomposition_stage(stage) else synth_attr
    v = getattr(args, attr, None)
    if v is None:
        return getattr(args, fallback_attr)
    if isinstance(v, str) and not v.strip():
        return getattr(args, fallback_attr)
    return v


class OpenAICompatBackend(ModelBackend):
    def __init__(self, args: argparse.Namespace) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for backend=openai_compat")
        self.args = args
        self.client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    def generate(self, messages: List[Dict[str, str]], stage: str) -> Dict[str, Any]:
        attempts = max(1, int(self.args.num_retries) + 1)
        last_err: Optional[Exception] = None

        model_name = stage_model_name(self.args, stage)
        max_tokens = int(stage_param(self.args, stage, "decompose_max_new_tokens", "synthesis_max_new_tokens", "max_new_tokens"))
        temperature = float(stage_param(self.args, stage, "decompose_temperature", "synthesis_temperature", "temperature"))
        top_p = float(stage_param(self.args, stage, "decompose_top_p", "synthesis_top_p", "top_p"))

        for attempt in range(1, attempts + 1):
            try:
                req: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "timeout": float(self.args.timeout),
                    "logprobs": bool(int(self.args.logprobs)),
                }
                completion = self.client.chat.completions.create(**req)
                raw_text = _parse_openai_text(completion)
                conf = _openai_confidence_from_logprobs(completion) if bool(int(self.args.logprobs)) else None
                return {
                    "raw_text": raw_text,
                    "confidence_overall": conf,
                    "model_meta": {
                        "backend": "openai_compat",
                        "provider_call_ok": True,
                        "stage": stage,
                        "model_name": model_name,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                    "error": None,
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= attempts or not _is_retryable_provider_error(e):
                    break
                backoff = min(30.0, (2 ** (attempt - 1)))
                time.sleep(backoff)

        return {
            "raw_text": "",
            "confidence_overall": None,
            "model_meta": {
                "backend": "openai_compat",
                "provider_call_ok": False,
                "stage": stage,
                "model_name": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "error": str(last_err) if last_err is not None else "UNKNOWN_OPENAI_ERROR",
        }


class VLLMLocalBackend(ModelBackend):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError("backend=vllm_local requires transformers + vllm") from exc

        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
        self.llm = LLM(
            model=args.model_path,
            trust_remote_code=bool(args.trust_remote_code),
            tensor_parallel_size=int(args.tensor_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            dtype=args.dtype,
            seed=int(args.seed),
        )

    def generate(self, messages: List[Dict[str, str]], stage: str) -> Dict[str, Any]:
        max_tokens = int(stage_param(self.args, stage, "decompose_max_new_tokens", "synthesis_max_new_tokens", "max_new_tokens"))
        temperature = float(stage_param(self.args, stage, "decompose_temperature", "synthesis_temperature", "temperature"))
        top_p = float(stage_param(self.args, stage, "decompose_top_p", "synthesis_top_p", "top_p"))

        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sampling = self.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            out = self.llm.generate([prompt], sampling)
            raw_text = ""
            if out and out[0].outputs:
                raw_text = (out[0].outputs[0].text or "").strip()
            return {
                "raw_text": raw_text,
                "confidence_overall": None,
                "model_meta": {
                    "backend": "vllm_local",
                    "provider_call_ok": True,
                    "stage": stage,
                    "model_path": self.args.model_path,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "raw_text": "",
                "confidence_overall": None,
                "model_meta": {
                    "backend": "vllm_local",
                    "provider_call_ok": False,
                    "stage": stage,
                    "model_path": self.args.model_path,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "error": str(e),
            }


def run_execution_preview(args: argparse.Namespace, final_sql: str) -> Dict[str, Any]:
    if int(args.skip_exec) == 1:
        return {"columns": [], "rows": [], "error": None}

    if not final_sql:
        return {"columns": [], "rows": [], "error": "EMPTY_SQL_FROM_MODEL"}

    cols, rows, err = rsq.execute_sql_preview(Path(args.db_path), final_sql, max_rows=int(args.preview_rows))
    return {"columns": cols, "rows": rows, "error": err}


def normalize_sql_for_key(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).lower()


def normalize_question_for_key(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip()).lower()


def build_error_result(task: Dict[str, Any], err: str) -> Dict[str, Any]:
    return {
        "task_index": task.get("task_index"),
        "global_index": task.get("global_index"),
        "item_id": task.get("item_id"),
        "input_question": task.get("question", ""),
        "error": err,
        "decomposition": {
            "raw_text": "",
            "decomposed_queries": [],
            "model_meta": {},
            "error": err,
        },
        "retrieval_trace": [],
        "ranked_results": [],
        "synthesized": {
            "raw_text": "",
            "final_sql": "",
            "confidence_overall": None,
            "model_meta": {},
        },
        "execution_preview": {
            "columns": [],
            "rows": [],
            "error": err,
        },
    }


def process_one_question(
    *,
    task: Dict[str, Any],
    candidates: Sequence[Any],
    backend: ModelBackend,
    base_prompt: str,
    schema_text: str,
    examples_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if task.get("error"):
        return build_error_result(task, str(task["error"]))

    question = str(task.get("question") or "").strip()
    if not question:
        return build_error_result(task, "EMPTY_QUESTION")

    try:
        decomp_messages = build_decomposition_messages(
            question=question,
            base_prompt=base_prompt,
            schema_text=schema_text,
            examples_text=examples_text,
            max_queries=max(1, int(args.max_decomposed_queries)),
        )
        decomp_gen = backend.generate(decomp_messages, stage="decomposition")

        decomp_queries: List[str] = []
        decomp_err: Optional[str] = None
        if decomp_gen.get("error"):
            decomp_err = str(decomp_gen["error"])
        else:
            decomp_queries = parse_decomposition(
                raw_text=str(decomp_gen.get("raw_text") or ""),
                max_items=max(1, int(args.max_decomposed_queries)),
            )
            if not decomp_queries:
                decomp_err = "FAILED_TO_PARSE_DECOMPOSITION"

        if not decomp_queries:
            decomp_queries = [question]

        ranked, trace = retrieve_from_decomposition(
            candidates=candidates,
            decomposed_queries=decomp_queries,
            top_k=max(1, int(args.top_k)),
            per_query_k=max(1, int(args.retrieval_per_decomp)),
            sbert_model=args.sbert_model,
            sbert_device=args.sbert_device,
            sbert_batch_size=int(args.sbert_batch_size),
        )

        if not ranked:
            return build_error_result(task, "EMPTY_RETRIEVAL_RESULTS")

        synth_messages = build_synthesis_messages(
            question=question,
            decomposed_queries=decomp_queries,
            ranked=ranked,
            base_prompt=base_prompt,
            schema_text=schema_text,
        )
        synth_gen = backend.generate(synth_messages, stage="synthesis")
        if synth_gen.get("error"):
            return build_error_result(task, str(synth_gen["error"]))

        raw_text = str(synth_gen.get("raw_text") or "")
        final_sql = extract_sql(raw_text)
        if not final_sql:
            return build_error_result(task, "FAILED_TO_EXTRACT_FINAL_SQL")

        exec_preview = run_execution_preview(args, final_sql)

        return {
            "task_index": task.get("task_index"),
            "global_index": task.get("global_index"),
            "item_id": task.get("item_id"),
            "input_question": question,
            "error": None,
            "decomposition": {
                "raw_text": decomp_gen.get("raw_text") or "",
                "decomposed_queries": decomp_queries,
                "model_meta": decomp_gen.get("model_meta") if isinstance(decomp_gen.get("model_meta"), dict) else {},
                "error": decomp_err,
            },
            "retrieval_trace": trace,
            "ranked_results": [serialize_match(m) for m in ranked],
            "synthesized": {
                "raw_text": raw_text,
                "final_sql": final_sql,
                "confidence_overall": synth_gen.get("confidence_overall"),
                "model_meta": synth_gen.get("model_meta") if isinstance(synth_gen.get("model_meta"), dict) else {},
            },
            "execution_preview": exec_preview,
        }
    except Exception as e:  # noqa: BLE001
        return build_error_result(task, f"TASK_PROCESS_ERROR:{e}")


def process_phase_tasks(
    *,
    tasks: Sequence[Dict[str, Any]],
    candidates: Sequence[Any],
    backend: ModelBackend,
    base_prompt: str,
    schema_text: str,
    examples_text: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    total = len(tasks)
    if total == 0:
        return []

    chunk_size = max(1, int(args.gen_batch_size))
    workers = 1 if args.backend == "vllm_local" else max(1, int(args.batch_concurrency))

    by_task_idx: Dict[int, Dict[str, Any]] = {}

    for b0 in range(0, total, chunk_size):
        chunk = list(tasks[b0 : b0 + chunk_size])

        if workers <= 1 or len(chunk) <= 1:
            for t in chunk:
                res = process_one_question(
                    task=t,
                    candidates=candidates,
                    backend=backend,
                    base_prompt=base_prompt,
                    schema_text=schema_text,
                    examples_text=examples_text,
                    args=args,
                )
                by_task_idx[int(t["task_index"])] = res
        else:
            max_workers = min(workers, len(chunk))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {
                    ex.submit(
                        process_one_question,
                        task=t,
                        candidates=candidates,
                        backend=backend,
                        base_prompt=base_prompt,
                        schema_text=schema_text,
                        examples_text=examples_text,
                        args=args,
                    ): int(t["task_index"])
                    for t in chunk
                }

                for fut in as_completed(fut_map):
                    idx = fut_map[fut]
                    try:
                        by_task_idx[idx] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        t = next((x for x in chunk if int(x["task_index"]) == idx), None)
                        if t is None:
                            t = {"task_index": idx, "question": "", "item_id": None, "global_index": None}
                        by_task_idx[idx] = build_error_result(t, f"TASK_PROCESS_ERROR:{e}")

        done = min(b0 + len(chunk), total)
        print(f"Processed {done}/{total}")

    return [by_task_idx[int(t["task_index"])] for t in tasks]


def build_eval_input_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "item_id": r.get("item_id"),
                "question": r.get("input_question"),
                "pred_sql": (r.get("synthesized", {}).get("final_sql") or "").strip(),
                "confidence_overall": r.get("synthesized", {}).get("confidence_overall"),
                "error": r.get("error"),
            }
        )
    return out


def run_eval_v2(
    *,
    eval_script: Path,
    pred_path: Path,
    gt_path: Path,
    db_path: Path,
    output_json: Path,
    compute_bertscore: int,
    ast_weight_select: float,
    ast_weight_where: float,
    ast_weight_from: float,
) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--pred_path",
        str(pred_path),
        "--gt_path",
        str(gt_path),
        "--db_path",
        str(db_path),
        "--output_json",
        str(output_json),
        "--pred_format",
        "json",
        "--gt_format",
        "json",
        "--pred_sql_key",
        "pred_sql",
        "--id_key",
        "item_id",
        "--question_key",
        "natural_question",
        "--compute_bertscore",
        str(int(compute_bertscore)),
        "--ast_weight_select",
        str(ast_weight_select),
        "--ast_weight_where",
        str(ast_weight_where),
        "--ast_weight_from",
        str(ast_weight_from),
    ]

    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(
            f"Eval script failed ({run.returncode})\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
        )


def parse_threshold_value(s: str) -> Any:
    t = (s or "").strip()
    tl = t.lower()
    if tl in {"true", "false"}:
        return tl == "true"
    try:
        return float(t)
    except Exception:  # noqa: BLE001
        return t


def parse_metric_rule_value(val: str) -> Tuple[str, Any]:
    t = (val or "").strip()
    m = re.match(r"^(>=|<=|==|!=|>|<)\s*(.+)$", t)
    if m:
        op = m.group(1)
        thr = parse_threshold_value(m.group(2))
        return op, thr
    # Default numeric comparator
    return ">=", parse_threshold_value(t)


def ensure_default_acceptance_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Seed acceptance rules (key=value)",
                "# Main condition: auto-accept when exact_exec_match == true",
                "main_metric=exact_exec_match",
                "main_operator===",
                "main_threshold=true",
                "",
                "# Otherwise require at least N non-main metric conditions",
                "required_non_main=2",
                "",
                "# Add/edit metrics here. Format: metric.<name>=<operator><threshold>",
                "# Example: metric.sql_ast_similarity=>=0.89",
                "metric.sql_ast_similarity=>=0.89",
                "metric.chrf=>=90",
                "metric.rouge_l_f1=>=90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def load_acceptance_rules(path: Path) -> Dict[str, Any]:
    ensure_default_acceptance_config(path)

    cfg: Dict[str, Any] = {
        "main_metric": "exact_exec_match",
        "main_operator": "==",
        "main_threshold": True,
        "required_non_main": 2,
        "metrics": [],
    }

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key == "main_metric":
            cfg["main_metric"] = val
        elif key == "main_operator":
            cfg["main_operator"] = val
        elif key == "main_threshold":
            cfg["main_threshold"] = parse_threshold_value(val)
        elif key == "required_non_main":
            try:
                cfg["required_non_main"] = int(val)
            except Exception:  # noqa: BLE001
                pass
        elif key.startswith("metric."):
            name = key[len("metric.") :].strip()
            if not name:
                continue
            op, thr = parse_metric_rule_value(val)
            cfg["metrics"].append({"name": name, "op": op, "threshold": thr})

    return cfg


def _to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "false"}:
            return t == "true"
    return None


def compare_value(actual: Any, op: str, threshold: Any) -> bool:
    if actual is None:
        return False

    # Boolean comparison path
    if isinstance(threshold, bool):
        ab = _to_bool(actual)
        if ab is None:
            return False
        if op == "==":
            return ab == threshold
        if op == "!=":
            return ab != threshold
        return False

    # Numeric comparison path
    if isinstance(threshold, (int, float)):
        try:
            av = float(actual)
            tv = float(threshold)
            # Convenience: if user uses percentage threshold for a [0,1] metric.
            if tv > 1.0 and 0.0 <= av <= 1.0:
                av *= 100.0
            if op == ">=":
                return av >= tv
            if op == "<=":
                return av <= tv
            if op == ">":
                return av > tv
            if op == "<":
                return av < tv
            if op == "==":
                return av == tv
            if op == "!=":
                return av != tv
            return False
        except Exception:  # noqa: BLE001
            return False

    # String fallback
    as_ = str(actual)
    ts = str(threshold)
    if op == "==":
        return as_ == ts
    if op == "!=":
        return as_ != ts
    return False


def acceptance_for_item(eval_item: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    main_metric = str(rules.get("main_metric") or "exact_exec_match")
    main_op = str(rules.get("main_operator") or "==")
    main_thr = rules.get("main_threshold", True)
    required_non_main = max(0, int(rules.get("required_non_main", 0)))
    metric_rules = list(rules.get("metrics") or [])

    main_val = eval_item.get(main_metric)
    main_pass = compare_value(main_val, main_op, main_thr)

    non_main_hits = 0
    checks: List[Dict[str, Any]] = []
    for r in metric_rules:
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        val = eval_item.get(name)
        op = str(r.get("op") or ">=")
        thr = r.get("threshold")
        ok = compare_value(val, op, thr)
        checks.append(
            {
                "metric": name,
                "value": val,
                "operator": op,
                "threshold": thr,
                "pass": ok,
            }
        )
        if ok:
            non_main_hits += 1

    accepted = bool(main_pass) or (non_main_hits >= required_non_main)
    reason = "main_metric" if main_pass else ("non_main_threshold" if accepted else "rejected")

    return {
        "accepted": accepted,
        "reason": reason,
        "main_metric": {
            "name": main_metric,
            "value": main_val,
            "operator": main_op,
            "threshold": main_thr,
            "pass": main_pass,
        },
        "non_main": {
            "required": required_non_main,
            "hits": non_main_hits,
            "checks": checks,
        },
    }


def build_seed_additions(
    *,
    phase_results: Sequence[Dict[str, Any]],
    eval_items: Sequence[Dict[str, Any]],
    rules: Dict[str, Any],
    phase_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_item_id = {str(r.get("item_id")): r for r in phase_results if r.get("item_id") is not None}

    additions: List[Dict[str, Any]] = []
    stats = {
        "evaluated": 0,
        "accepted": 0,
        "accepted_main_metric": 0,
        "accepted_non_main": 0,
        "rejected": 0,
    }

    for ev in eval_items:
        item_id = ev.get("item_id")
        if item_id is None:
            continue

        stats["evaluated"] += 1
        decision = acceptance_for_item(ev, rules)
        accepted = bool(decision.get("accepted"))
        if not accepted:
            stats["rejected"] += 1
            continue

        src = by_item_id.get(str(item_id))
        if src is None:
            stats["rejected"] += 1
            continue

        pred_sql = (src.get("synthesized", {}).get("final_sql") or "").strip()
        question = (src.get("input_question") or "").strip()
        if not pred_sql or not question:
            stats["rejected"] += 1
            continue

        if decision.get("reason") == "main_metric":
            stats["accepted_main_metric"] += 1
        else:
            stats["accepted_non_main"] += 1

        additions.append(
            {
                "original_question": question,
                "question": question,
                "gt_sql": pred_sql,
                "decomposition_questions": src.get("decomposition", {}).get("decomposed_queries") or [],
                "added_from_phase": phase_id,
                "item_id": item_id,
                "selection": {
                    "decision": decision,
                    "eval_metrics": {
                        "exact_exec_match": ev.get("exact_exec_match"),
                        "sql_ast_similarity": ev.get("sql_ast_similarity"),
                        "chrf": ev.get("chrf"),
                        "rouge_l_f1": ev.get("rouge_l_f1"),
                    },
                },
            }
        )
        stats["accepted"] += 1

    return additions, stats


def merge_seed_rows(base_rows: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    merged: List[Dict[str, Any]] = [dict(x) for x in base_rows]
    seen = set()

    def row_key(r: Dict[str, Any]) -> Tuple[str, str]:
        q = normalize_question_for_key(str(r.get("question") or r.get("original_question") or ""))
        sql = normalize_sql_for_key(str(r.get("sql") or r.get("gt_sql") or ""))
        return q, sql

    for r in merged:
        seen.add(row_key(r))

    added = 0
    for r in new_rows:
        k = row_key(r)
        if not k[0] or not k[1]:
            continue
        if k in seen:
            continue
        seen.add(k)
        merged.append(dict(r))
        added += 1

    return merged, added


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_questions = load_questions(args.question_json)
    start = max(0, int(args.start_index))
    total_limit = int(args.total_questions)
    if total_limit < 0:
        total_limit = len(all_questions) - start
    total_limit = min(total_limit, max(0, len(all_questions) - start))

    if total_limit <= 0:
        raise SystemExit("No questions selected. Check --start-index and --total-questions")

    phase_size = max(1, int(args.phase_size))
    num_phases = (total_limit + phase_size - 1) // phase_size

    base_prompt = load_template_text(args.prompt_file)
    schema_text = load_schema_text(args.schema_json)

    initial_seed_path = Path(args.candidate_json) if args.candidate_json else Path(args.seed_json)
    current_seed_rows = load_seed_rows(initial_seed_path)
    current_seed_path = out_dir / "seed_phase0_initial.json"
    save_json(current_seed_path, current_seed_rows)

    examples_text = load_decomposition_examples(args.decompose_examples_file, current_seed_rows)
    rules_path = Path(args.acceptance_config)
    rules = load_acceptance_rules(rules_path)

    if args.backend == "openai_compat":
        backend: ModelBackend = OpenAICompatBackend(args)
    else:
        backend = VLLMLocalBackend(args)

    pipeline_summary: Dict[str, Any] = {
        "meta": {
            "question_json": args.question_json,
            "seed_json": str(initial_seed_path),
            "output_dir": str(out_dir),
            "start_index": start,
            "total_questions": total_limit,
            "phase_size": phase_size,
            "num_phases": num_phases,
            "backend": args.backend,
            "model_name": args.model_name if args.backend == "openai_compat" else None,
            "decompose_model_name": (args.decompose_model_name or args.model_name) if args.backend == "openai_compat" else None,
            "synthesis_model_name": (args.synthesis_model_name or args.model_name) if args.backend == "openai_compat" else None,
            "model_path": args.model_path if args.backend == "vllm_local" else None,
            "call_mode": "two_call_decompose_then_synthesize",
            "generation": {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "decompose_max_new_tokens": args.decompose_max_new_tokens,
                "synthesis_max_new_tokens": args.synthesis_max_new_tokens,
                "decompose_temperature": args.decompose_temperature,
                "synthesis_temperature": args.synthesis_temperature,
                "decompose_top_p": args.decompose_top_p,
                "synthesis_top_p": args.synthesis_top_p,
            },
            "acceptance_config": str(rules_path),
        },
        "phases": [],
    }

    for phase_id in range(1, num_phases + 1):
        phase_start = start + (phase_id - 1) * phase_size
        phase_end = min(start + total_limit, phase_start + phase_size)
        phase_rows = all_questions[phase_start:phase_end]
        if not phase_rows:
            break

        print("=" * 90)
        print(f"PHASE {phase_id}/{num_phases} | rows {phase_start}..{phase_end - 1} | seed={current_seed_path.name}")
        print("=" * 90)
        print("Call mode: two-call (decomposition -> synthesis)")

        tasks = build_phase_tasks(
            rows=phase_rows,
            question_key=args.question_key,
            id_key=args.id_key,
            global_start_index=phase_start,
        )

        candidates = rsq.load_candidates_from_seed_json(current_seed_path)
        if not candidates:
            raise RuntimeError(f"No candidates loaded from seed file: {current_seed_path}")

        phase_results = process_phase_tasks(
            tasks=tasks,
            candidates=candidates,
            backend=backend,
            base_prompt=base_prompt,
            schema_text=schema_text,
            examples_text=examples_text,
            args=args,
        )

        phase_counts = {
            "total_tasks": len(phase_results),
            "error_tasks": sum(1 for r in phase_results if r.get("error")),
            "generated_nonempty_sql": sum(1 for r in phase_results if (r.get("synthesized", {}).get("final_sql") or "").strip()),
            "exec_ok": sum(1 for r in phase_results if not (r.get("execution_preview", {}).get("error"))),
        }

        phase_name = f"500_phase{phase_id}"
        phase_output_path = out_dir / f"{phase_name}.json"
        phase_payload = {
            "meta": {
                "phase_id": phase_id,
                "phase_start_index": phase_start,
                "phase_end_index": phase_end - 1,
                "phase_size": phase_end - phase_start,
                "seed_input_path": str(current_seed_path),
                "candidate_count": len(candidates),
                "top_k": int(args.top_k),
                "retrieval_per_decomp": int(args.retrieval_per_decomp),
                "max_decomposed_queries": int(args.max_decomposed_queries),
                "sbert_model": args.sbert_model,
            },
            "counts": phase_counts,
            "results": phase_results,
        }
        save_json(phase_output_path, phase_payload)

        eval_summary: Dict[str, Any] = {}
        additions_stats: Dict[str, Any] = {
            "evaluated": 0,
            "accepted": 0,
            "accepted_main_metric": 0,
            "accepted_non_main": 0,
            "rejected": 0,
            "added_unique": 0,
        }
        added_seed_rows_path = out_dir / f"{phase_name}.seed_additions.json"

        new_rows: List[Dict[str, Any]] = []
        if int(args.run_eval) == 1:
            eval_input_rows = build_eval_input_rows(phase_results)
            eval_input_path = out_dir / f"{phase_name}.eval_input.json"
            gt_subset_path = out_dir / f"{phase_name}.gt_subset.json"
            eval_output_path = out_dir / f"{phase_name}.eval.json"

            save_json(eval_input_path, eval_input_rows)
            save_json(gt_subset_path, phase_rows)

            run_eval_v2(
                eval_script=Path(args.eval_script),
                pred_path=eval_input_path,
                gt_path=gt_subset_path,
                db_path=Path(args.db_path),
                output_json=eval_output_path,
                compute_bertscore=int(args.compute_bertscore),
                ast_weight_select=float(args.ast_weight_select),
                ast_weight_where=float(args.ast_weight_where),
                ast_weight_from=float(args.ast_weight_from),
            )

            eval_obj = json.loads(eval_output_path.read_text(encoding="utf-8"))
            eval_summary = (eval_obj.get("summary") if isinstance(eval_obj, dict) else {}) or {}
            eval_items = (eval_obj.get("per_item") if isinstance(eval_obj, dict) else []) or []
            if not isinstance(eval_items, list):
                eval_items = []

            new_rows, additions_stats = build_seed_additions(
                phase_results=phase_results,
                eval_items=eval_items,
                rules=rules,
                phase_id=phase_id,
            )
            save_json(added_seed_rows_path, new_rows)
        else:
            save_json(added_seed_rows_path, new_rows)

        merged_seed_rows, added_unique = merge_seed_rows(current_seed_rows, new_rows)
        additions_stats["added_unique"] = int(added_unique)

        next_seed_path = out_dir / f"seed_phase{phase_id}.json"
        save_json(next_seed_path, merged_seed_rows)

        phase_summary = {
            "phase_id": phase_id,
            "phase_name": phase_name,
            "phase_output_json": str(phase_output_path),
            "eval_summary": eval_summary,
            "seed_additions_stats": additions_stats,
            "seed_input_path": str(current_seed_path),
            "seed_output_path": str(next_seed_path),
            "seed_size_before": len(current_seed_rows),
            "seed_size_after": len(merged_seed_rows),
        }
        pipeline_summary["phases"].append(phase_summary)

        current_seed_rows = merged_seed_rows
        current_seed_path = next_seed_path

    summary_path = out_dir / "pipeline_summary.json"
    save_json(summary_path, pipeline_summary)

    print("=" * 90)
    print("Pipeline completed")
    print(f"Summary: {summary_path}")
    for p in pipeline_summary.get("phases", []):
        print(
            f"Phase {p['phase_id']}: seed {p['seed_size_before']} -> {p['seed_size_after']} "
            f"(added_unique={p['seed_additions_stats'].get('added_unique', 0)})"
        )


if __name__ == "__main__":
    main()
