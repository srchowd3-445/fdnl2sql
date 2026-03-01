#!/usr/bin/env python3
"""Decompose question -> SBERT retrieve top-k -> synthesize final SQL.

Pipeline per question:
1) Build decomposition prompt starting from zero_shot_sql_expert.txt
2) Ask model to decompose the natural-language question
3) Run SBERT retrieval on decomposed query text against seed candidates
4) Feed top-k retrieved (question, sql) pairs back to model for final SQL synthesis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Decompose -> SBERT retrieve -> synthesize SQL.")

    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    source.add_argument("--candidate-json")
    source.add_argument("--candidate-sqlite")

    ap.add_argument("--candidate-table", default="query_library")
    ap.add_argument("--candidate-question-col", default="question")
    ap.add_argument("--candidate-sql-col", default="sql")
    ap.add_argument("--candidate-id-col", default="id")

    qsrc = ap.add_mutually_exclusive_group(required=False)
    qsrc.add_argument("--question", help="Ad-hoc natural-language question")
    qsrc.add_argument("--question-json", default=str(root / "data" / "natural_question_1500.json"))

    ap.add_argument("--question-index", type=int, default=0, help="Single-mode index when using --question-json")
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--id-key", default="item_id")

    ap.add_argument("--batch-mode", type=int, default=0, help="Set 1 to process range from --question-json")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1, help="Batch size; -1 for all remaining")

    ap.add_argument("--top-k", type=int, default=3, help="Final retrieved candidates passed to synthesis")
    ap.add_argument(
        "--retrieval-per-decomp",
        type=int,
        default=5,
        help="Retrieve this many per decomposed query before global top-k merge",
    )
    ap.add_argument("--max-decomposed-queries", type=int, default=5)

    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-device", default=None)
    ap.add_argument("--sbert-batch-size", type=int, default=64)

    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--prompt-file", default=str(root / "prompting" / "zero_shot_sql_expert.txt"))

    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")

    # OpenAI-compatible backend args
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)
    ap.add_argument("--gen-batch-size", type=int, default=32, help="Tasks dispatched per chunk")
    ap.add_argument("--batch-concurrency", type=int, default=8, help="Concurrent tasks for openai_compat backend")

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
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=8192)

    # Shared generation args
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)

    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=0)

    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))

    ap.add_argument(
        "--output-json",
        default=str(root / "results" / "retrieval_decompose_sbert_gemma_synth.json"),
    )
    ap.add_argument("--output-jsonl", default="", help="Optional sql_with_conf JSONL output")

    return ap.parse_args()


def import_retriever(root: Path):
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import retrieve_similar_queries as rsq  # type: ignore

    return rsq


def import_utils(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as u  # type: ignore

    return u


try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from openai.types import ResponseFormatJSONSchema
except Exception:
    ResponseFormatJSONSchema = None

try:
    from structured_logprobs import add_logprobs
except Exception:
    add_logprobs = None


if BaseModel is not None:

    class SynthResponse(BaseModel):
        sql: str


if BaseModel is not None:

    class DecomposeResponse(BaseModel):
        decomposed_queries: List[str]


def build_response_schema(model_cls: Any, enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    if BaseModel is None or ResponseFormatJSONSchema is None:
        raise RuntimeError("pydantic + openai.types.ResponseFormatJSONSchema are required for schema mode")

    json_schema = model_cls.model_json_schema()
    response_schema = ResponseFormatJSONSchema.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": json_schema,
            },
        }
    )
    return response_schema.model_dump(by_alias=True)


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


def load_template_text(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def load_schema_text(path: Optional[str]) -> str:
    if not path:
        return ""
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


def load_candidates(args: argparse.Namespace, rsq) -> List[Any]:
    if args.candidate_sqlite:
        return rsq.load_candidates_from_sqlite(
            db_path=Path(args.candidate_sqlite),
            table=args.candidate_table,
            question_col=args.candidate_question_col,
            sql_col=args.candidate_sql_col,
            id_col=args.candidate_id_col or None,
        )

    src = Path(args.candidate_json) if args.candidate_json else Path(args.seed_json)
    return rsq.load_candidates_from_seed_json(src)


def extract_question_from_row(row: Dict[str, Any], key: str) -> Tuple[str, Optional[str]]:
    q = row.get(key)
    if isinstance(q, str) and q.strip():
        return q.strip(), None

    for alt in ("natural_question", "question", "original_question"):
        alt_q = row.get(alt)
        if isinstance(alt_q, str) and alt_q.strip():
            return alt_q.strip(), None

    return "", f"MISSING_QUESTION_KEY:{key}"


def load_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.question:
        return [
            {
                "task_index": 0,
                "question_index": None,
                "item_id": None,
                "question": args.question.strip(),
                "error": None,
            }
        ]

    obj = json.loads(Path(args.question_json).read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list at {args.question_json}")

    tasks: List[Dict[str, Any]] = []

    if int(args.batch_mode) == 1:
        start = max(0, int(args.start_index))
        end = len(obj) if int(args.limit) == -1 else min(len(obj), start + max(0, int(args.limit)))
        indices = list(range(start, end))
    else:
        idx = int(args.question_index)
        if idx < 0 or idx >= len(obj):
            raise IndexError(f"question-index {idx} out of range for {len(obj)} rows")
        indices = [idx]

    for task_idx, q_idx in enumerate(indices):
        row = obj[q_idx]
        if not isinstance(row, dict):
            tasks.append(
                {
                    "task_index": task_idx,
                    "question_index": q_idx,
                    "item_id": None,
                    "question": "",
                    "error": "ROW_NOT_OBJECT",
                }
            )
            continue

        q_text, q_err = extract_question_from_row(row, args.question_key)
        tasks.append(
            {
                "task_index": task_idx,
                "question_index": q_idx,
                "item_id": row.get(args.id_key),
                "question": q_text,
                "error": q_err,
            }
        )

    return tasks


def build_decomposition_messages(
    question: str,
    base_prompt: str,
    schema_text: str,
    use_pydantic_schema: bool,
) -> List[Dict[str, str]]:
    base = render_base_prompt(base_prompt=base_prompt, schema_text=schema_text, question=question)

    out_instruction = (
        'Return ONLY JSON with key "decomposed_queries" and a list of concise strings.'
        if use_pydantic_schema
        else "Return only the decomposed sub-questions, one per line (no SQL)."
    )

    user = (
        f"{base}\n\n"
        "Before writing final SQL, decompose the question into atomic SQL-oriented sub-questions "
        "that preserve the entities/constraints from the original question.\n"
        "Use 1 to 5 sub-questions.\n"
        f"{out_instruction}"
    )

    system = (
        "You are an expert SQL planning assistant. "
        "Decompose questions into retrieval-friendly sub-questions."
    )

    return [
        {"role": "system", "content": system},
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


def build_match_result(rsq, rank: int, score: float, candidate: Any) -> Any:
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
    rsq,
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
        ranked_out.append(build_match_result(rsq, rank=i, score=score, candidate=cand))

    return ranked_out, traces


def build_synthesis_messages(
    question: str,
    decomposed_queries: Sequence[str],
    ranked: Sequence[Any],
    base_prompt: str,
    schema_text: str,
    use_pydantic_schema: bool,
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

    out_instruction = (
        'Return ONLY a JSON object with key "sql".'
        if use_pydantic_schema
        else "Output only SQL (no markdown, no explanation)."
    )

    user = (
        f"{base}\n\n"
        "Decomposed sub-questions used for retrieval:\n"
        f"{decomp_block}\n\n"
        "Top retrieved SBERT examples:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        "Synthesize one best final SQL query for the original question.\n"
        f"{out_instruction}"
    )

    system = (
        "You are an expert SQLite NL2SQL assistant. "
        "Follow instructions exactly and output only valid JSON when schema is provided."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def init_vllm_local(args: argparse.Namespace):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("vllm_local backend requires transformers + vllm") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))

    llm = LLM(
        model=args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype=args.dtype,
        seed=int(args.seed),
    )

    sampling = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    return tokenizer, llm, sampling


def generate_one_vllm_local(
    tokenizer: Any,
    llm: Any,
    sampling: Any,
    messages: Sequence[Dict[str, str]],
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = llm.generate([prompt], sampling)
    if not out:
        return ""
    choices = out[0].outputs if out[0] is not None else []
    return (choices[0].text or "").strip() if choices else ""


def _run_one_openai_call(
    client: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    response_schema: Optional[Dict[str, Any]],
    is_retryable_fn: Any,
    use_logprobs: bool,
) -> Any:
    attempts = max(1, int(args.num_retries) + 1)
    last_err: Any = None

    for attempt in range(1, attempts + 1):
        try:
            req: Dict[str, Any] = {
                "model": args.model_name,
                "messages": messages,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_tokens": int(args.max_new_tokens),
                "timeout": float(args.timeout),
                "logprobs": bool(use_logprobs),
            }
            if response_schema is not None:
                req["response_format"] = response_schema
            return client.chat.completions.create(**req)
        except Exception as e:  # noqa: BLE001
            last_err = e
            retryable = is_retryable_fn(e)
            if attempt >= attempts or not retryable:
                break
            backoff = min(30.0, (2 ** (attempt - 1)))
            print(f"Retry call attempt={attempt}/{attempts} sleep={backoff:.1f}s err={str(e).splitlines()[0]}")
            time.sleep(backoff)

    return last_err


def run_generation(
    *,
    args: argparse.Namespace,
    backend_state: Dict[str, Any],
    utils_mod: Any,
    messages: List[Dict[str, str]],
    response_schema: Optional[Dict[str, Any]],
    use_logprobs: bool,
) -> Dict[str, Any]:
    if args.backend == "vllm_local":
        tokenizer = backend_state["tokenizer"]
        llm = backend_state["llm"]
        sampling = backend_state["sampling"]
        try:
            raw_text = generate_one_vllm_local(tokenizer, llm, sampling, messages)
            return {
                "raw_text": raw_text,
                "completion": None,
                "model_meta": {"provider_call_ok": True, "backend": "vllm_local"},
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "raw_text": "",
                "completion": None,
                "model_meta": {"provider_call_ok": False, "backend": "vllm_local"},
                "error": str(e),
            }

    client = backend_state["client"]
    out_obj = _run_one_openai_call(
        client=client,
        messages=messages,
        args=args,
        response_schema=response_schema,
        is_retryable_fn=utils_mod.is_retryable_provider_error,
        use_logprobs=use_logprobs,
    )

    if isinstance(out_obj, Exception):
        return {
            "raw_text": "",
            "completion": None,
            "model_meta": {"provider_call_ok": False, "backend": "openai_compat"},
            "error": str(out_obj),
        }

    raw_text = utils_mod.parse_openai_text(out_obj)
    model_meta = utils_mod.openai_completion_to_meta(out_obj)
    model_meta["backend"] = "openai_compat"

    return {
        "raw_text": raw_text,
        "completion": out_obj,
        "model_meta": model_meta,
        "error": None,
    }


def run_execution_preview(rsq, args: argparse.Namespace, final_sql: str) -> Dict[str, Any]:
    if int(args.skip_exec) == 1:
        return {"columns": [], "rows": [], "error": None}

    if not final_sql:
        return {"columns": [], "rows": [], "error": "EMPTY_SQL_FROM_MODEL"}

    cols, rows, err = rsq.execute_sql_preview(Path(args.db_path), final_sql, max_rows=int(args.preview_rows))
    return {
        "columns": cols,
        "rows": rows,
        "error": err,
    }


def load_gt_maps(gt_json: str, id_key: str, question_key: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    p = Path(gt_json)
    if not p.exists():
        return {}, {}

    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return {}, {}

    by_id: Dict[str, Dict[str, Any]] = {}
    by_q: Dict[str, Dict[str, Any]] = {}
    for r in obj:
        if not isinstance(r, dict):
            continue
        item_id = r.get(id_key)
        q = r.get(question_key) or r.get("natural_question") or r.get("question")

        if isinstance(item_id, str) and item_id:
            by_id[item_id] = r
        if isinstance(q, str) and q.strip():
            by_q[q.strip()] = r

    return by_id, by_q


def find_gt_row(
    task: Dict[str, Any],
    gt_by_id: Dict[str, Dict[str, Any]],
    gt_by_q: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    item_id = task.get("item_id")
    q = (task.get("question") or "").strip()

    if isinstance(item_id, str) and item_id in gt_by_id:
        return gt_by_id[item_id]
    if q and q in gt_by_q:
        return gt_by_q[q]
    return None


def to_sql_with_conf_row(
    task: Dict[str, Any],
    gt_row: Optional[Dict[str, Any]],
    synthesized: Dict[str, Any],
) -> Dict[str, Any]:
    question = (task.get("question") or "").strip()
    gt_sql = (gt_row.get("gt_sql") if isinstance(gt_row, dict) else "") or ""

    return {
        "question": question,
        "gt_sql": gt_sql,
        "gt_filters": gt_row.get("gt_filters") if isinstance(gt_row, dict) else {},
        "gt_columns": gt_row.get("gt_columns") if isinstance(gt_row, dict) else [],
        "pred_sql": (synthesized.get("final_sql") or "").strip(),
        "pred_filters": {},
        "pred_columns": [],
        "raw_model_output": synthesized.get("raw_text") or "",
        "field_logprobs": synthesized.get("field_logprobs") if isinstance(synthesized.get("field_logprobs"), dict) else {},
        "field_confidence": synthesized.get("field_confidence") if isinstance(synthesized.get("field_confidence"), dict) else {},
        "confidence_overall": synthesized.get("confidence_overall"),
        "model_meta": synthesized.get("model_meta") if isinstance(synthesized.get("model_meta"), dict) else {},
    }


def build_error_result_row(
    task: Dict[str, Any],
    error_msg: str,
    gt_by_id: Dict[str, Dict[str, Any]],
    gt_by_q: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    task_idx = int(task.get("task_index", -1))
    question = str(task.get("question") or "").strip()
    gt_row = find_gt_row(task, gt_by_id, gt_by_q)
    synthesized = {
        "raw_text": "",
        "final_sql": "",
        "field_logprobs": {},
        "field_confidence": {},
        "confidence_overall": None,
        "model_meta": {},
    }
    return {
        "task_index": task_idx,
        "question_index": task.get("question_index"),
        "item_id": task.get("item_id"),
        "input_question": question,
        "error": str(error_msg),
        "decomposition": {
            "raw_text": "",
            "decomposed_queries": [],
            "model_meta": {},
            "error": str(error_msg),
        },
        "retrieval_trace": [],
        "ranked_results": [],
        "synthesized": synthesized,
        "execution_preview": {"columns": [], "rows": [], "error": str(error_msg)},
        "sql_with_conf_row": to_sql_with_conf_row(task, gt_row, synthesized),
    }


def process_task(
    *,
    task: Dict[str, Any],
    args: argparse.Namespace,
    rsq: Any,
    utils_mod: Any,
    base_prompt: str,
    schema_text: str,
    candidates: Sequence[Any],
    backend_state: Dict[str, Any],
    decomp_schema: Optional[Dict[str, Any]],
    synth_schema: Optional[Dict[str, Any]],
    gt_by_id: Dict[str, Dict[str, Any]],
    gt_by_q: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    task_idx = int(task["task_index"])
    question = str(task.get("question") or "").strip()
    err = task.get("error")

    if err:
        return build_error_result_row(task, str(err), gt_by_id, gt_by_q)

    try:
        # 1) Decompose with LLM
        decomp_messages = build_decomposition_messages(
            question=question,
            base_prompt=base_prompt,
            schema_text=schema_text,
            use_pydantic_schema=bool(args.use_pydantic_schema),
        )
        decomp_gen = run_generation(
            args=args,
            backend_state=backend_state,
            utils_mod=utils_mod,
            messages=decomp_messages,
            response_schema=decomp_schema,
            use_logprobs=False,
        )

        decomposed_queries: List[str] = []
        decomp_parse_error: Optional[str] = None
        if decomp_gen.get("error"):
            decomp_parse_error = str(decomp_gen["error"])
        else:
            decomposed_queries = parse_decomposition(
                raw_text=str(decomp_gen.get("raw_text") or ""),
                max_items=max(1, int(args.max_decomposed_queries)),
            )
            if not decomposed_queries:
                decomp_parse_error = "FAILED_TO_PARSE_DECOMPOSITION"

        if not decomposed_queries:
            decomposed_queries = [question]

        # 2) SBERT retrieval over decomposed queries
        ranked: List[Any] = []
        retrieval_trace: List[Dict[str, Any]] = []
        retrieval_error: Optional[str] = None
        try:
            ranked, retrieval_trace = retrieve_from_decomposition(
                rsq=rsq,
                candidates=candidates,
                decomposed_queries=decomposed_queries,
                top_k=max(1, int(args.top_k)),
                per_query_k=max(1, int(args.retrieval_per_decomp)),
                sbert_model=args.sbert_model,
                sbert_device=args.sbert_device,
                sbert_batch_size=int(args.sbert_batch_size),
            )
            if not ranked:
                retrieval_error = "EMPTY_RETRIEVAL_RESULTS"
        except Exception as e:  # noqa: BLE001
            retrieval_error = str(e)

        # 3) Final synthesis
        synth_gen: Dict[str, Any] = {
            "raw_text": "",
            "completion": None,
            "model_meta": {},
            "error": None,
        }
        final_sql = ""
        field_lp: Dict[str, Any] = {}
        field_conf: Dict[str, Any] = {}
        conf_overall: Optional[float] = None

        if retrieval_error is None:
            synth_messages = build_synthesis_messages(
                question=question,
                decomposed_queries=decomposed_queries,
                ranked=ranked,
                base_prompt=base_prompt,
                schema_text=schema_text,
                use_pydantic_schema=bool(args.use_pydantic_schema),
            )
            synth_gen = run_generation(
                args=args,
                backend_state=backend_state,
                utils_mod=utils_mod,
                messages=synth_messages,
                response_schema=synth_schema,
                use_logprobs=(args.logprob_mode == "structured"),
            )

            if not synth_gen.get("error"):
                raw_text = str(synth_gen.get("raw_text") or "")
                if bool(args.use_pydantic_schema):
                    try:
                        parsed = json.loads(raw_text)
                        final_sql = str(parsed.get("sql") or "").strip()
                    except Exception as e:  # noqa: BLE001
                        synth_gen.setdefault("model_meta", {})["response_parse_error"] = str(e)

                if not final_sql:
                    final_sql = extract_sql(raw_text)

                if args.backend == "openai_compat" and args.logprob_mode == "structured":
                    try:
                        field_lp, field_conf, conf_overall = utils_mod.structured_logprob_payload(
                            synth_gen.get("completion"), add_logprobs
                        )
                    except Exception as lp_err:  # noqa: BLE001
                        synth_gen.setdefault("model_meta", {})["structured_logprob_error"] = str(lp_err).splitlines()[0]
                        field_lp, field_conf, conf_overall = utils_mod.openai_token_logprob_payload(
                            synth_gen.get("completion")
                        )
                        synth_gen.setdefault("model_meta", {})["logprob_fallback"] = "openai_token_logprobs"
            else:
                retrieval_error = str(synth_gen.get("error"))

        execution = run_execution_preview(rsq, args, final_sql)

        combined_error = retrieval_error
        if combined_error is None and not final_sql:
            combined_error = "FAILED_TO_EXTRACT_FINAL_SQL"

        synthesized = {
            "raw_text": synth_gen.get("raw_text") or "",
            "final_sql": final_sql,
            "field_logprobs": field_lp,
            "field_confidence": field_conf,
            "confidence_overall": conf_overall,
            "model_meta": synth_gen.get("model_meta") if isinstance(synth_gen.get("model_meta"), dict) else {},
        }

        decomp_payload = {
            "raw_text": decomp_gen.get("raw_text") or "",
            "decomposed_queries": decomposed_queries,
            "model_meta": decomp_gen.get("model_meta") if isinstance(decomp_gen.get("model_meta"), dict) else {},
            "error": decomp_parse_error,
        }

        result_row = {
            "task_index": task_idx,
            "question_index": task.get("question_index"),
            "item_id": task.get("item_id"),
            "input_question": question,
            "error": combined_error,
            "decomposition": decomp_payload,
            "retrieval_trace": retrieval_trace,
            "ranked_results": [
                {
                    **asdict(m),
                    "candidate": asdict(m.candidate),
                }
                for m in ranked
            ],
            "synthesized": synthesized,
            "execution_preview": execution,
        }

        result_row["sql_with_conf_row"] = to_sql_with_conf_row(
            task=task,
            gt_row=find_gt_row(task, gt_by_id, gt_by_q),
            synthesized=synthesized,
        )
        return result_row
    except Exception as e:  # noqa: BLE001
        return build_error_result_row(task, f"TASK_PROCESS_ERROR:{e}", gt_by_id, gt_by_q)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.backend == "openai_compat":
        if OpenAI is None:
            raise RuntimeError("openai package is required for backend=openai_compat")
        if bool(args.use_pydantic_schema):
            if BaseModel is None or ResponseFormatJSONSchema is None:
                raise RuntimeError("pydantic + openai.types.ResponseFormatJSONSchema are required when --use-pydantic-schema=1")
        if args.logprob_mode == "structured" and add_logprobs is None:
            raise RuntimeError("structured_logprobs is required for --logprob-mode structured")

    rsq = import_retriever(root)
    utils_mod = import_utils(root)

    base_prompt = load_template_text(args.prompt_file)
    schema_text = load_schema_text(args.schema_json)

    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No tasks loaded.")

    candidates = load_candidates(args, rsq)
    if not candidates:
        raise SystemExit("No candidates loaded. Check candidate source format/path.")

    gt_by_id, gt_by_q = load_gt_maps(args.gt_json, args.id_key, args.question_key)

    backend_state: Dict[str, Any] = {}
    if args.backend == "vllm_local":
        tokenizer, llm, sampling = init_vllm_local(args)
        backend_state = {"tokenizer": tokenizer, "llm": llm, "sampling": sampling}
    else:
        backend_state = {"client": OpenAI(base_url=args.api_base, api_key=args.api_key)}

    decomp_schema = build_response_schema(DecomposeResponse, bool(args.use_pydantic_schema)) if BaseModel is not None else None
    synth_schema = build_response_schema(SynthResponse, bool(args.use_pydantic_schema)) if BaseModel is not None else None

    # Warm up SBERT candidate embeddings once to avoid repeated first-hit latency/races.
    first_question = next((str(t.get("question") or "").strip() for t in tasks if not t.get("error")), "")
    if first_question:
        try:
            rsq.rank_candidates(
                question=first_question,
                candidates=candidates,
                top_k=1,
                sbert_model=args.sbert_model,
                sbert_device=args.sbert_device,
                sbert_batch_size=int(args.sbert_batch_size),
            )
        except Exception:
            pass

    total_tasks = len(tasks)
    results_by_task_index: Dict[int, Dict[str, Any]] = {}
    chunk_size = max(1, int(args.gen_batch_size))
    workers = 1 if args.backend == "vllm_local" else max(1, int(args.batch_concurrency))

    for b0 in range(0, total_tasks, chunk_size):
        chunk = tasks[b0 : b0 + chunk_size]

        if workers <= 1 or len(chunk) == 1:
            for task in chunk:
                row = process_task(
                    task=task,
                    args=args,
                    rsq=rsq,
                    utils_mod=utils_mod,
                    base_prompt=base_prompt,
                    schema_text=schema_text,
                    candidates=candidates,
                    backend_state=backend_state,
                    decomp_schema=decomp_schema,
                    synth_schema=synth_schema,
                    gt_by_id=gt_by_id,
                    gt_by_q=gt_by_q,
                )
                results_by_task_index[int(row.get("task_index", task.get("task_index", -1)))] = row
        else:
            max_workers = min(workers, len(chunk))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_task_idx: Dict[Any, int] = {}
                for task in chunk:
                    fut = ex.submit(
                        process_task,
                        task=task,
                        args=args,
                        rsq=rsq,
                        utils_mod=utils_mod,
                        base_prompt=base_prompt,
                        schema_text=schema_text,
                        candidates=candidates,
                        backend_state=backend_state,
                        decomp_schema=decomp_schema,
                        synth_schema=synth_schema,
                        gt_by_id=gt_by_id,
                        gt_by_q=gt_by_q,
                    )
                    fut_to_task_idx[fut] = int(task["task_index"])

                for fut in as_completed(fut_to_task_idx):
                    task_idx = fut_to_task_idx[fut]
                    try:
                        row = fut.result()
                    except Exception as e:  # noqa: BLE001
                        task_lookup = next((t for t in chunk if int(t["task_index"]) == task_idx), None)
                        if task_lookup is None:
                            task_lookup = {"task_index": task_idx, "question": "", "question_index": None, "item_id": None}
                        row = build_error_result_row(task_lookup, f"TASK_PROCESS_ERROR:{e}", gt_by_id, gt_by_q)
                    results_by_task_index[int(row.get("task_index", task_idx))] = row

        done = min(b0 + len(chunk), total_tasks)
        print(f"Processed {done}/{total_tasks}")

    results = [results_by_task_index[int(t["task_index"])] for t in tasks]

    counts = {
        "total_tasks": len(results),
        "error_tasks": sum(1 for r in results if r.get("error")),
        "generated_nonempty_sql": sum(1 for r in results if (r.get("synthesized", {}).get("final_sql") or "").strip()),
        "exec_ok": sum(1 for r in results if not (r.get("execution_preview", {}).get("error"))),
        "with_confidence": sum(1 for r in results if isinstance(r.get("synthesized", {}).get("confidence_overall"), (int, float))),
    }

    payload: Dict[str, Any] = {
        "meta": {
            "backend": args.backend,
            "question": args.question,
            "question_json": args.question_json,
            "question_index": int(args.question_index),
            "batch_mode": int(args.batch_mode),
            "start_index": int(args.start_index),
            "limit": int(args.limit),
            "prompt_file": args.prompt_file,
            "schema_json": args.schema_json,
            "seed_json": args.seed_json,
            "candidate_json": args.candidate_json,
            "candidate_sqlite": args.candidate_sqlite,
            "candidate_table": args.candidate_table,
            "top_k": int(args.top_k),
            "retrieval_per_decomp": int(args.retrieval_per_decomp),
            "max_decomposed_queries": int(args.max_decomposed_queries),
            "sbert": {
                "model": args.sbert_model,
                "device": args.sbert_device,
                "batch_size": int(args.sbert_batch_size),
            },
            "api_base": args.api_base if args.backend == "openai_compat" else None,
            "model_name": args.model_name if args.backend == "openai_compat" else None,
            "model_path": args.model_path if args.backend == "vllm_local" else None,
            "db_path": args.db_path,
            "skip_exec": int(args.skip_exec),
            "logprob_mode": args.logprob_mode,
            "use_pydantic_schema": bool(args.use_pydantic_schema),
            "gen_batch_size": int(args.gen_batch_size),
            "batch_concurrency": int(args.batch_concurrency),
            "output_jsonl": args.output_jsonl or None,
        },
        "counts": counts,
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if (args.output_jsonl or "").strip():
        out_jsonl = Path(args.output_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                row = r.get("sql_with_conf_row")
                if isinstance(row, dict):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Saved JSONL:", out_jsonl)

    print("Saved JSON:", out_path)
    print("Tasks:", counts["total_tasks"])
    print("Errors:", counts["error_tasks"])
    print("Generated SQL:", counts["generated_nonempty_sql"])
    if int(args.skip_exec) != 1:
        print("Exec ok:", counts["exec_ok"])
    print("Rows with confidence_overall:", counts["with_confidence"])


if __name__ == "__main__":
    main()
