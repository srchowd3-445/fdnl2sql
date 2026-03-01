#!/usr/bin/env python3
"""Decompose questions, score decomposition with SBERT retrieval, and save results.

This script performs only the first half of the pipeline:
1) Decompose each natural-language question with an LLM.
2) Retrieve top-k seed examples via SBERT using decomposed sub-questions.
3) Save decomposition + retrieval artifacts for a later synthesis step.
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


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _choose_token_param_name(model_name: str) -> str:
    m = (model_name or "").strip().lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Decompose questions and retrieve top-k SBERT candidates.")

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

    ap.add_argument("--top-k", type=int, default=3, help="Global top-k after merge across decomposed sub-queries")
    ap.add_argument(
        "--retrieval-per-decomp",
        type=int,
        default=3,
        help="Retrieve this many per decomposed query before global top-k merge",
    )
    ap.add_argument("--max-decomposed-queries", type=int, default=5)

    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-device", default=None)
    ap.add_argument("--sbert-batch-size", type=int, default=64)

    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--prompt-file", default=str(root / "method" / "prompt" / "decompose_prompt.txt"))
    ap.add_argument(
        "--decompose-examples-file",
        default=str(root / "method" / "prompt" / "decompose_examples.txt"),
        help="Optional good/bad decomposition examples text file",
    )

    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")

    # OpenAI-compatible backend args
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="gemma-3-27b-local")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)

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

    # Decomposition generation args
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)

    ap.add_argument("--gen-batch-size", type=int, default=32, help="Tasks dispatched per chunk")
    ap.add_argument("--batch-concurrency", type=int, default=8, help="Concurrent tasks for openai_compat backend")

    ap.add_argument(
        "--output-json",
        default=str(root / "results" / "decompose_retrieve_top3_gemma.json"),
    )
    ap.add_argument("--output-jsonl", default="", help="Optional compact JSONL output")

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


def default_good_bad_examples() -> str:
    return (
        "Good decomposition example:\n"
        "Original question: Which melanoma phase 3 trials used Pembrolizumab?\n"
        "Decomposed queries:\n"
        "- Which trials have Cancer type = Melanoma?\n"
        "- Which trials have Trial phase = Phase 3?\n"
        "- Which trials have Name of ICI = Pembrolizumab?\n\n"
        "Bad decomposition example:\n"
        "Original question: Which colorectal trials had 3 or more arms and used multikinase inhibitor as control?\n"
        "Bad decomposed queries (too vague / not retrieval-friendly):\n"
        "- Find trials from the table.\n"
        "- Write SQL for this question.\n"
        "- Show me the answer.\n"
    )


def load_decomposition_examples(path: str) -> str:
    p = Path(path)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return default_good_bad_examples()


def load_candidates(args: argparse.Namespace, rsq: Any) -> List[Any]:
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
    *,
    question: str,
    base_prompt: str,
    schema_text: str,
    examples_text: str,
    max_queries: int,
) -> List[Dict[str, str]]:
    base = render_base_prompt(base_prompt=base_prompt, schema_text=schema_text, question=question)

    # If examples are already embedded in the prompt file, avoid appending duplicates.
    base_l = base.lower()
    has_embedded_examples = "good decomposition example" in base_l and "bad decomposition example" in base_l
    examples_block = "" if has_embedded_examples else f"{examples_text}\n\n"

    user = (
        f"{base}\n\n"
        "You are now performing query decomposition, not final SQL generation.\n"
        "Decompose the original question into short, retrieval-friendly atomic sub-questions.\n"
        "Each sub-question should map to one WHERE-style predicate (column + operator + value).\n"
        "Prefer schema-aligned phrasing that can map directly to SQL WHERE conditions.\n"
        f"Return between 1 and {max(1, int(max_queries))} sub-questions.\n\n"
        f"{examples_block}"
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


def _parse_decomposition_obj(obj: Any, max_items: int) -> List[str]:
    if isinstance(obj, dict):
        dq = obj.get("decomposed_queries")
        if isinstance(dq, list):
            cleaned = _clean_decomposed_items(dq, max_items=max_items)
            if cleaned:
                return cleaned

        d1 = obj.get("decomposed_query")
        if isinstance(d1, str) and d1.strip():
            return _clean_decomposed_items([d1], max_items=max_items)

    if isinstance(obj, list):
        cleaned = _clean_decomposed_items(obj, max_items=max_items)
        if cleaned:
            return cleaned

    return []


def _extract_json_object_substring(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _candidate_json_payloads(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []

    payloads: List[str] = [t]

    # Common model format: fenced JSON block.
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, flags=re.IGNORECASE)
    if m:
        inner = (m.group(1) or "").strip()
        if inner:
            payloads.append(inner)

    obj_sub = _extract_json_object_substring(t)
    if obj_sub:
        payloads.append(obj_sub)

    out: List[str] = []
    seen = set()
    for p in payloads:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def parse_decomposition(raw_text: str, max_items: int) -> List[str]:
    t = (raw_text or "").strip()
    if not t:
        return []

    for payload in _candidate_json_payloads(t):
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001
            continue

        parsed = _parse_decomposition_obj(obj, max_items=max_items)
        if parsed:
            return parsed

    lines: List[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("```"):
            continue

        s = re.sub(r"^[-*]\s+", "", s)
        s = re.sub(r"^\d+[\.)]\s+", "", s)
        s = s.strip()
        if not s:
            continue

        # If fallback lines still include inline JSON, try parsing that line.
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = json.loads(s)
            except Exception:  # noqa: BLE001
                pass
            else:
                parsed = _parse_decomposition_obj(obj, max_items=max_items)
                if parsed:
                    return parsed
            continue

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


def build_match_result(rsq: Any, rank: int, score: float, candidate: Any) -> Any:
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


def serialize_match_obj(m: Any) -> Dict[str, Any]:
    d = asdict(m)
    if hasattr(m, "candidate"):
        d["candidate"] = asdict(m.candidate)
    return d


def retrieve_from_decomposition(
    *,
    rsq: Any,
    candidates: Sequence[Any],
    decomposed_queries: Sequence[str],
    top_k: int,
    per_query_k: int,
    sbert_model: str,
    sbert_device: Optional[str],
    sbert_batch_size: int,
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    traces: List[Dict[str, Any]] = []
    best_by_candidate_id: Dict[str, Tuple[float, Any, str]] = {}
    per_decomp_ranked: List[Dict[str, Any]] = []

    parts = [q for q in decomposed_queries if (q or "").strip()]
    if not parts:
        return [], traces, per_decomp_ranked

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

        per_decomp_ranked.append(
            {
                "decomposed_query": part,
                "ranked_results": [serialize_match_obj(m) for m in ranked_part],
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

    return ranked_out, traces, per_decomp_ranked


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
    *,
    client: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    utils_mod: Any,
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
                "timeout": float(args.timeout),
            }
            token_param = _choose_token_param_name(str(args.model_name))
            if int(args.max_new_tokens) > 0:
                req[token_param] = int(args.max_new_tokens)
            return client.chat.completions.create(**req)
        except Exception as e:  # noqa: BLE001
            last_err = e
            retryable = utils_mod.is_retryable_provider_error(e)
            if attempt >= attempts or not retryable:
                break
            backoff = min(30.0, (2 ** (attempt - 1)))
            print(f"Retry decompose attempt={attempt}/{attempts} sleep={backoff:.1f}s err={str(e).splitlines()[0]}")
            time.sleep(backoff)

    return last_err


def run_decompose_generation(
    *,
    args: argparse.Namespace,
    backend_state: Dict[str, Any],
    utils_mod: Any,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    if args.backend == "vllm_local":
        tokenizer = backend_state["tokenizer"]
        llm = backend_state["llm"]
        sampling = backend_state["sampling"]
        try:
            raw_text = generate_one_vllm_local(tokenizer, llm, sampling, messages)
            return {
                "raw_text": raw_text,
                "model_meta": {"provider_call_ok": True, "backend": "vllm_local"},
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "raw_text": "",
                "model_meta": {"provider_call_ok": False, "backend": "vllm_local"},
                "error": str(e),
            }

    client = backend_state["client"]
    out_obj = _run_one_openai_call(client=client, messages=messages, args=args, utils_mod=utils_mod)

    if isinstance(out_obj, Exception):
        return {
            "raw_text": "",
            "model_meta": {"provider_call_ok": False, "backend": "openai_compat"},
            "error": str(out_obj),
        }

    raw_text = utils_mod.parse_openai_text(out_obj)
    model_meta = utils_mod.openai_completion_to_meta(out_obj)
    model_meta["backend"] = "openai_compat"

    return {
        "raw_text": raw_text,
        "model_meta": model_meta,
        "error": None,
    }


def process_task(
    *,
    task: Dict[str, Any],
    args: argparse.Namespace,
    rsq: Any,
    utils_mod: Any,
    base_prompt: str,
    schema_text: str,
    examples_text: str,
    candidates: Sequence[Any],
    backend_state: Dict[str, Any],
) -> Dict[str, Any]:
    task_idx = int(task.get("task_index", -1))
    question = str(task.get("question") or "").strip()
    err = task.get("error")

    if err:
        return {
            "task_index": task_idx,
            "question_index": task.get("question_index"),
            "item_id": task.get("item_id"),
            "input_question": question,
            "error": str(err),
            "decomposition": {
                "raw_text": "",
                "decomposed_queries": [],
                "model_meta": {},
                "error": str(err),
            },
            "retrieval_trace": [],
            "retrieval_by_decomposition": [],
            "ranked_results": [],
        }

    try:
        decomp_messages = build_decomposition_messages(
            question=question,
            base_prompt=base_prompt,
            schema_text=schema_text,
            examples_text=examples_text,
            max_queries=max(1, int(args.max_decomposed_queries)),
        )
        decomp_gen = run_decompose_generation(
            args=args,
            backend_state=backend_state,
            utils_mod=utils_mod,
            messages=decomp_messages,
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

        retrieval_error: Optional[str] = None
        ranked: List[Any] = []
        retrieval_trace: List[Dict[str, Any]] = []
        retrieval_by_decomposition: List[Dict[str, Any]] = []

        try:
            ranked, retrieval_trace, retrieval_by_decomposition = retrieve_from_decomposition(
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
            retrieval_error = f"RETRIEVAL_ERROR:{e}"

        return {
            "task_index": task_idx,
            "question_index": task.get("question_index"),
            "item_id": task.get("item_id"),
            "input_question": question,
            "error": retrieval_error,
            "decomposition": {
                "raw_text": decomp_gen.get("raw_text") or "",
                "decomposed_queries": decomposed_queries,
                "model_meta": decomp_gen.get("model_meta") if isinstance(decomp_gen.get("model_meta"), dict) else {},
                "error": decomp_parse_error,
            },
            "retrieval_trace": retrieval_trace,
            "retrieval_by_decomposition": retrieval_by_decomposition,
            "ranked_results": [
                serialize_match_obj(m)
                for m in ranked
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "task_index": task_idx,
            "question_index": task.get("question_index"),
            "item_id": task.get("item_id"),
            "input_question": question,
            "error": f"TASK_PROCESS_ERROR:{e}",
            "decomposition": {
                "raw_text": "",
                "decomposed_queries": [],
                "model_meta": {},
                "error": f"TASK_PROCESS_ERROR:{e}",
            },
            "retrieval_trace": [],
            "retrieval_by_decomposition": [],
            "ranked_results": [],
        }


def to_jsonl_row(r: Dict[str, Any]) -> Dict[str, Any]:
    ranked = r.get("ranked_results") if isinstance(r.get("ranked_results"), list) else []
    top = ranked[0] if ranked else {}
    top_c = top.get("candidate") if isinstance(top, dict) else {}

    return {
        "task_index": r.get("task_index"),
        "question_index": r.get("question_index"),
        "item_id": r.get("item_id"),
        "input_question": r.get("input_question"),
        "error": r.get("error"),
        "decomposed_queries": (r.get("decomposition") or {}).get("decomposed_queries", []),
        "top_candidate_id": top_c.get("candidate_id") if isinstance(top_c, dict) else None,
        "top_candidate_question": top_c.get("question") if isinstance(top_c, dict) else None,
        "top_candidate_sql": top_c.get("sql") if isinstance(top_c, dict) else None,
        "top_score": top.get("sbert_score") if isinstance(top, dict) and "sbert_score" in top else (top.get("total_score") if isinstance(top, dict) else None),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.backend == "openai_compat" and OpenAI is None:
        raise RuntimeError("openai package is required for backend=openai_compat")

    rsq = import_retriever(root)
    utils_mod = import_utils(root)

    base_prompt = load_template_text(args.prompt_file)
    schema_text = load_schema_text(args.schema_json)
    examples_text = load_decomposition_examples(args.decompose_examples_file)

    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No tasks loaded.")

    candidates = load_candidates(args, rsq)
    if not candidates:
        raise SystemExit("No candidates loaded. Check candidate source format/path.")

    backend_state: Dict[str, Any] = {}
    if args.backend == "vllm_local":
        tokenizer, llm, sampling = init_vllm_local(args)
        backend_state = {"tokenizer": tokenizer, "llm": llm, "sampling": sampling}
    else:
        backend_state = {"client": OpenAI(base_url=args.api_base, api_key=args.api_key)}

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
                    examples_text=examples_text,
                    candidates=candidates,
                    backend_state=backend_state,
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
                        examples_text=examples_text,
                        candidates=candidates,
                        backend_state=backend_state,
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
                        row = {
                            "task_index": task_idx,
                            "question_index": task_lookup.get("question_index"),
                            "item_id": task_lookup.get("item_id"),
                            "input_question": task_lookup.get("question"),
                            "error": f"TASK_PROCESS_ERROR:{e}",
                            "decomposition": {"raw_text": "", "decomposed_queries": [], "model_meta": {}, "error": f"TASK_PROCESS_ERROR:{e}"},
                            "retrieval_trace": [],
                            "retrieval_by_decomposition": [],
                            "ranked_results": [],
                        }
                    results_by_task_index[int(row.get("task_index", task_idx))] = row

        done = min(b0 + len(chunk), total_tasks)
        print(f"Processed {done}/{total_tasks}")

    results = [results_by_task_index[int(t["task_index"])] for t in tasks]

    counts = {
        "total_tasks": len(results),
        "error_tasks": sum(1 for r in results if r.get("error")),
        "with_decomposition": sum(1 for r in results if (r.get("decomposition", {}).get("decomposed_queries") or [])),
        "with_retrieval": sum(1 for r in results if (r.get("ranked_results") or [])),
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
            "decompose_examples_file": args.decompose_examples_file,
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
            "gen_batch_size": int(args.gen_batch_size),
            "batch_concurrency": int(args.batch_concurrency),
            "pipeline_stage": "decompose_then_retrieve",
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
                f.write(json.dumps(to_jsonl_row(r), ensure_ascii=False) + "\n")
        print("Saved JSONL:", out_jsonl)

    print("Saved JSON:", out_path)
    print("Tasks:", counts["total_tasks"])
    print("Errors:", counts["error_tasks"])
    print("With decomposition:", counts["with_decomposition"])
    print("With retrieval:", counts["with_retrieval"])


if __name__ == "__main__":
    main()
