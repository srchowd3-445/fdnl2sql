#!/usr/bin/env python3
"""Synthesize final SQL from precomputed retrieval top-k candidates.

Primary mode for confidence extraction:
- openai_compat backend (vLLM serve / OpenAI-compatible API)
- Pydantic response schema + response_format JSON schema
- structured field logprobs and confidence (same pattern as vllm_api_inference.py)

Optional fallback mode:
- vllm_local backend (local vLLM Python API, no structured confidence)
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

    ap = argparse.ArgumentParser(description="Synthesize SQL from precomputed retrieval top-k candidates.")

    ap.add_argument(
        "--retrieval-json",
        default=str(root / "method" / "retrieval_top3_all_questions_seq.json"),
        help="Path to retrieval output JSON containing results[].top_k",
    )
    ap.add_argument("--start-index", type=int, default=0, help="Start row index in retrieval results")
    ap.add_argument("--limit", type=int, default=-1, help="How many rows to process; -1 means all remaining")
    ap.add_argument("--top-k", type=int, default=3, help="Use up to top-k candidates from retrieval row")

    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--prompt-file", default=str(root / "prompting" / "zero_shot_sql_expert.txt"))

    # Backend control
    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")

    # OpenAI-compatible backend args (for confidence extraction)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)
    ap.add_argument("--batch-concurrency", type=int, default=4)

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
    ap.add_argument("--gen-batch-size", type=int, default=1)

    # Optional GT mapping for sql_with_conf JSONL format
    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--id-key", default="item_id")
    ap.add_argument("--question-key", default="natural_question")

    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=0, help="1 to skip SQL execution preview")

    ap.add_argument(
        "--output-json",
        default=str(root / "results" / "retrieval_top3_gemma_synth_from_json.json"),
        help="Structured synthesis output JSON",
    )
    ap.add_argument(
        "--output-jsonl",
        default="",
        help="Optional JSONL output matching sql_with_conf format",
    )

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
        filters: Dict[str, Any] = {}
        columns: List[str] = []


def build_response_schema(enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    if BaseModel is None or ResponseFormatJSONSchema is None:
        raise RuntimeError("pydantic + openai.types.ResponseFormatJSONSchema are required for schema mode")

    json_schema = SynthResponse.model_json_schema()
    response_schema = ResponseFormatJSONSchema.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "SynthResponse",
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


def candidate_from_dict(rsq, payload: Dict[str, Any]) -> Any:
    c = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    return rsq.Candidate(
        candidate_id=str(c.get("candidate_id", payload.get("candidate_id", ""))),
        question=str(c.get("question", payload.get("candidate_question", ""))),
        sql=str(c.get("sql", payload.get("candidate_sql", ""))),
        parent_question=c.get("parent_question"),
        source=str(c.get("source", "retrieval_json")),
    )


def match_from_dict(rsq, payload: Dict[str, Any], fallback_rank: int) -> Any:
    return rsq.MatchResult(
        rank=int(payload.get("rank", fallback_rank)),
        total_score=float(payload.get("total_score", 0.0)),
        lexical_score=float(payload.get("lexical_score", 0.0)),
        char_score=float(payload.get("char_score", 0.0)),
        literal_score=float(payload.get("literal_score", 0.0)),
        operator_score=float(payload.get("operator_score", 0.0)),
        column_score=float(payload.get("column_score", 0.0)),
        candidate=candidate_from_dict(rsq, payload),
    )


def load_tasks_from_retrieval(args: argparse.Namespace, rsq) -> List[Dict[str, Any]]:
    obj = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    rows = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Expected retrieval JSON with top-level results[] in {args.retrieval_json}")

    start = max(0, int(args.start_index))
    end = len(rows)
    if int(args.limit) > -1:
        end = min(end, start + int(args.limit))

    tasks: List[Dict[str, Any]] = []
    top_k = max(1, int(args.top_k))

    for task_idx, idx in enumerate(range(start, end)):
        row = rows[idx]
        if not isinstance(row, dict):
            tasks.append(
                {
                    "task_index": task_idx,
                    "question_index": None,
                    "item_id": None,
                    "question": "",
                    "error": "ROW_NOT_OBJECT",
                    "ranked": [],
                    "retrieval_row_index": idx,
                }
            )
            continue

        question = row.get("question") or row.get("input_question") or ""
        question = question.strip() if isinstance(question, str) else ""

        ranked_payload = row.get("top_k")
        if not isinstance(ranked_payload, list):
            ranked_payload = row.get("ranked_results") if isinstance(row.get("ranked_results"), list) else []

        ranked: List[Any] = []
        for j, one in enumerate(ranked_payload[:top_k], start=1):
            if not isinstance(one, dict):
                continue
            ranked.append(match_from_dict(rsq, one, fallback_rank=j))

        err = row.get("error")
        if err is None:
            if not question:
                err = "EMPTY_QUESTION"
            elif not ranked:
                err = "EMPTY_RETRIEVAL_TOPK"

        tasks.append(
            {
                "task_index": task_idx,
                "question_index": row.get("question_index"),
                "item_id": row.get("item_id"),
                "question": question,
                "error": err,
                "ranked": ranked,
                "retrieval_row_index": idx,
            }
        )

    return tasks


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


def build_messages(
    question: str,
    ranked: Sequence[Any],
    base_prompt: str,
    schema_text: str,
) -> List[Dict[str, str]]:
    candidate_blocks: List[str] = []
    for m in ranked:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate Rank {m.rank}",
                    f"Score: {m.total_score:.4f}",
                    f"Candidate Question: {m.candidate.question}",
                    "Candidate SQL:",
                    m.candidate.sql,
                ]
            )
        )

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
        prompt = prompt.replace("{QUESTION}", question)
    else:
        prompt = f"{prompt}\n\nQuestion: {question}\nGenerate SQL:"

    user = (
        f"{prompt}\n\n"
        "Retrieved candidate SQL examples:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        "Synthesize one best final SQL query for the question.\n"
        "Output only SQL (no markdown, no explanation)."
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


def generate_vllm_local(
    tokenizer: Any,
    llm: Any,
    sampling: Any,
    messages_list: Sequence[List[Dict[str, str]]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    n = len(messages_list)
    outputs: List[Dict[str, Any]] = [
        {
            "raw_text": "",
            "pred_sql": "",
            "pred_filters": {},
            "pred_columns": [],
            "field_logprobs": {},
            "field_confidence": {},
            "confidence_overall": None,
            "error": None,
        }
        for _ in range(n)
    ]

    bs = max(1, int(batch_size))
    for b0 in range(0, n, bs):
        b1 = min(n, b0 + bs)
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list[b0:b1]
        ]

        outs = llm.generate(prompts, sampling)
        for j, out in enumerate(outs):
            idx = b0 + j
            text = (out.outputs[0].text or "").strip() if out.outputs else ""
            outputs[idx]["raw_text"] = text
            outputs[idx]["pred_sql"] = extract_sql(text)

        print(f"Generated {b1}/{n}")

    return outputs


def _run_one_openai_call(
    client: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    response_schema: Optional[Dict[str, Any]],
    is_retryable_fn: Any,
) -> Any:
    attempts = max(1, int(args.num_retries) + 1)
    last_err: Any = None

    for attempt in range(1, attempts + 1):
        try:
            req: Dict[str, Any] = {
                "model": args.model_name,
                "messages": messages,
                "logprobs": (args.logprob_mode == "structured"),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_tokens": int(args.max_new_tokens),
                "timeout": float(args.timeout),
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
            print(f"Retry openai call attempt={attempt}/{attempts} sleep={backoff:.1f}s err={str(e).splitlines()[0]}")
            time.sleep(backoff)

    return last_err


def generate_openai_compat(
    client: Any,
    messages_list: Sequence[List[Dict[str, str]]],
    args: argparse.Namespace,
    response_schema: Optional[Dict[str, Any]],
    utils_mod: Any,
) -> List[Dict[str, Any]]:
    n = len(messages_list)
    outputs: List[Dict[str, Any]] = [
        {
            "raw_text": "",
            "pred_sql": "",
            "pred_filters": {},
            "pred_columns": [],
            "field_logprobs": {},
            "field_confidence": {},
            "confidence_overall": None,
            "error": None,
        }
        for _ in range(n)
    ]

    bs = max(1, int(args.gen_batch_size))
    for b0 in range(0, n, bs):
        b1 = min(n, b0 + bs)
        chunk = messages_list[b0:b1]

        max_workers = max(1, min(int(args.batch_concurrency), len(chunk)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {
                ex.submit(
                    _run_one_openai_call,
                    client,
                    messages,
                    args,
                    response_schema,
                    utils_mod.is_retryable_provider_error,
                ): j
                for j, messages in enumerate(chunk)
            }

            for fut in as_completed(fut_to_idx):
                rel_j = fut_to_idx[fut]
                idx = b0 + rel_j
                try:
                    out_obj = fut.result()
                except Exception as e:  # noqa: BLE001
                    out_obj = e

                if isinstance(out_obj, Exception):
                    outputs[idx]["error"] = str(out_obj)
                    continue

                raw_text = utils_mod.parse_openai_text(out_obj)
                pred_sql = ""
                pred_filters: Dict[str, Any] = {}
                pred_columns: List[Any] = []
                parse_err: Optional[str] = None

                if bool(args.use_pydantic_schema):
                    try:
                        parsed = json.loads(raw_text)
                        pred_sql = (parsed.get("sql") or "").strip()
                        f = parsed.get("filters")
                        c = parsed.get("columns")
                        pred_filters = f if isinstance(f, dict) else {}
                        pred_columns = c if isinstance(c, list) else []
                    except Exception as e:  # noqa: BLE001
                        parse_err = f"JSON_PARSE_ERROR:{e}"

                if not pred_sql:
                    pred_sql = extract_sql(raw_text)

                field_lp: Dict[str, Any] = {}
                field_conf: Dict[str, Any] = {}
                conf_overall: Optional[float] = None

                if args.logprob_mode == "structured":
                    try:
                        field_lp, field_conf, conf_overall = utils_mod.structured_logprob_payload(out_obj, add_logprobs)
                    except Exception:
                        field_lp, field_conf, conf_overall = {}, {}, None

                    if not field_lp:
                        try:
                            field_lp, field_conf, conf_overall = utils_mod.openai_token_logprob_payload(out_obj)
                        except Exception:
                            field_lp, field_conf, conf_overall = {}, {}, None

                outputs[idx].update(
                    {
                        "raw_text": raw_text,
                        "pred_sql": pred_sql,
                        "pred_filters": pred_filters,
                        "pred_columns": pred_columns,
                        "field_logprobs": field_lp,
                        "field_confidence": field_conf,
                        "confidence_overall": conf_overall,
                        "error": parse_err,
                    }
                )

        print(f"Generated {b1}/{n}")

    return outputs


def run_execution_preview(rsq, args: argparse.Namespace, final_sql: str) -> Dict[str, Any]:
    if int(args.skip_exec) == 1:
        return {
            "columns": [],
            "rows": [],
            "error": None,
        }

    if not final_sql:
        return {
            "columns": [],
            "rows": [],
            "error": "EMPTY_SQL_FROM_MODEL",
        }

    cols, rows, err = rsq.execute_sql_preview(Path(args.db_path), final_sql, max_rows=int(args.preview_rows))
    return {
        "columns": cols,
        "rows": rows,
        "error": err,
    }


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
    gen: Dict[str, Any],
) -> Dict[str, Any]:
    question = (task.get("question") or "").strip()
    gt_sql = (gt_row.get("gt_sql") if isinstance(gt_row, dict) else "") or ""
    gt_filters = gt_row.get("gt_filters") if isinstance(gt_row, dict) else {}
    gt_columns = gt_row.get("gt_columns") if isinstance(gt_row, dict) else []

    return {
        "question": question,
        "gt_sql": gt_sql,
        "gt_filters": gt_filters,
        "gt_columns": gt_columns,
        "pred_sql": (gen.get("pred_sql") or "").strip(),
        "pred_filters": gen.get("pred_filters") if isinstance(gen.get("pred_filters"), dict) else {},
        "pred_columns": gen.get("pred_columns") if isinstance(gen.get("pred_columns"), list) else [],
        "raw_model_output": gen.get("raw_text") or "",
        "field_logprobs": gen.get("field_logprobs") if isinstance(gen.get("field_logprobs"), dict) else {},
        "field_confidence": gen.get("field_confidence") if isinstance(gen.get("field_confidence"), dict) else {},
        "confidence_overall": gen.get("confidence_overall"),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    rsq = import_retriever(root)
    utils_mod = import_utils(root)

    base_prompt = load_template_text(args.prompt_file)
    schema_text = load_schema_text(args.schema_json)

    tasks = load_tasks_from_retrieval(args, rsq)
    if not tasks:
        raise SystemExit("No tasks loaded from retrieval JSON.")

    gt_by_id, gt_by_q = load_gt_maps(args.gt_json, args.id_key, args.question_key)

    prepared: List[Dict[str, Any]] = []
    results_by_task_index: Dict[int, Dict[str, Any]] = {}

    for task in tasks:
        task_idx = int(task["task_index"])
        err = task.get("error")
        if err:
            results_by_task_index[task_idx] = {
                "task_index": task_idx,
                "retrieval_row_index": task.get("retrieval_row_index"),
                "question_index": task.get("question_index"),
                "item_id": task.get("item_id"),
                "input_question": task.get("question", ""),
                "error": str(err),
                "ranked_results": [],
                "synthesized": {
                    "raw_text": "",
                    "final_sql": "",
                    "pred_filters": {},
                    "pred_columns": [],
                    "field_logprobs": {},
                    "field_confidence": {},
                    "confidence_overall": None,
                },
                "execution_preview": {"columns": [], "rows": [], "error": str(err)},
            }
            continue

        question = str(task.get("question") or "").strip()
        ranked = task.get("ranked") or []

        messages = build_messages(
            question=question,
            ranked=ranked,
            base_prompt=base_prompt,
            schema_text=schema_text,
        )
        prepared.append(
            {
                "task": task,
                "ranked": ranked,
                "messages": messages,
                "gt_row": find_gt_row(task, gt_by_id, gt_by_q),
            }
        )

    generated: List[Dict[str, Any]] = []
    if prepared:
        messages_list = [x["messages"] for x in prepared]

        if args.backend == "vllm_local":
            tokenizer, llm, sampling = init_vllm_local(args)
            generated = generate_vllm_local(
                tokenizer=tokenizer,
                llm=llm,
                sampling=sampling,
                messages_list=messages_list,
                batch_size=int(args.gen_batch_size),
            )
        else:
            if OpenAI is None:
                raise RuntimeError("openai package is required for backend=openai_compat")
            response_schema = build_response_schema(bool(args.use_pydantic_schema))
            client = OpenAI(base_url=args.api_base, api_key=args.api_key)
            generated = generate_openai_compat(
                client=client,
                messages_list=messages_list,
                args=args,
                response_schema=response_schema,
                utils_mod=utils_mod,
            )

        for packed, gen in zip(prepared, generated):
            task = packed["task"]
            ranked = packed["ranked"]
            task_idx = int(task["task_index"])

            final_sql = (gen.get("pred_sql") or "").strip()
            execution = run_execution_preview(rsq, args, final_sql)

            results_by_task_index[task_idx] = {
                "task_index": task_idx,
                "retrieval_row_index": task.get("retrieval_row_index"),
                "question_index": task.get("question_index"),
                "item_id": task.get("item_id"),
                "input_question": task.get("question"),
                "error": gen.get("error"),
                "ranked_results": [
                    {
                        **asdict(m),
                        "candidate": asdict(m.candidate),
                    }
                    for m in ranked
                ],
                "synthesized": {
                    "raw_text": gen.get("raw_text") or "",
                    "final_sql": final_sql,
                    "pred_filters": gen.get("pred_filters") if isinstance(gen.get("pred_filters"), dict) else {},
                    "pred_columns": gen.get("pred_columns") if isinstance(gen.get("pred_columns"), list) else [],
                    "field_logprobs": gen.get("field_logprobs") if isinstance(gen.get("field_logprobs"), dict) else {},
                    "field_confidence": gen.get("field_confidence") if isinstance(gen.get("field_confidence"), dict) else {},
                    "confidence_overall": gen.get("confidence_overall"),
                },
                "execution_preview": execution,
                "sql_with_conf_row": to_sql_with_conf_row(task, packed.get("gt_row"), gen),
            }

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
            "retrieval_json": args.retrieval_json,
            "start_index": int(args.start_index),
            "limit": int(args.limit),
            "top_k": int(args.top_k),
            "prompt_file": args.prompt_file,
            "schema_json": args.schema_json,
            "api_base": args.api_base if args.backend == "openai_compat" else None,
            "model_name": args.model_name if args.backend == "openai_compat" else None,
            "model_path": args.model_path if args.backend == "vllm_local" else None,
            "gen_batch_size": int(args.gen_batch_size),
            "db_path": args.db_path,
            "skip_exec": int(args.skip_exec),
            "logprob_mode": args.logprob_mode,
            "use_pydantic_schema": bool(args.use_pydantic_schema),
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
