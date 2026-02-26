#!/usr/bin/env python3
"""Read decomposition+retrieval JSON and run synthesis only.

This script is the second half of the split pipeline:
1) Read decomposition/retrieval artifacts saved from decompose_retrieve_top3_gemma_sql.py
2) Build synthesis prompt per question using decomposed sub-queries and retrieved seed examples
3) Generate final SQL and save outputs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None

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


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Synthesize SQL from decomposition+retrieval JSON.")

    ap.add_argument("--retrieval-json", required=True, help="Output JSON from decompose_retrieve_top3_gemma_sql.py")

    ap.add_argument("--prompt-file", default=str(root / "method" / "prompt" / "synthesis_prompt.txt"))
    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))

    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")

    # OpenAI-compatible backend args
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="gemma-3-27b-local")
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)
    ap.add_argument("--logprobs", type=int, default=0, help="Fallback OpenAI token-level logprobs when logprob-mode=none")

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

    # Synthesis generation args
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)
    ap.add_argument(
        "--synth-evidence-per-decomp",
        type=int,
        default=2,
        help="How many retrieved examples to include per decomposed query in synthesis prompt",
    )
    ap.add_argument(
        "--synth-include-global-merged",
        type=int,
        default=0,
        help="Include globally merged ranked_results block in synthesis prompt",
    )
    ap.add_argument(
        "--synth-where-only-evidence",
        type=int,
        default=1,
        help="When 1, include concise WHERE-clause pattern alongside full SQL to reduce prompt noise",
    )

    ap.add_argument("--gen-batch-size", type=int, default=32, help="Tasks dispatched per chunk")
    ap.add_argument("--batch-concurrency", type=int, default=8, help="Concurrent tasks for openai_compat backend")

    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=1)

    # Eval integration
    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--run-eval", type=int, default=1)
    ap.add_argument("--eval-script", default=str(root / "eval_run_baselines_v2.py"))
    ap.add_argument("--eval-ready-json", default="", help="Optional eval-ready JSON prediction file path")
    ap.add_argument("--eval-ready-jsonl", default="", help="Optional eval-ready JSONL prediction file path")
    ap.add_argument("--eval-output-json", default="", help="Optional eval output JSON path")
    ap.add_argument("--compute-bertscore", type=int, default=0)
    ap.add_argument("--ast-weight-select", type=float, default=0.5)
    ap.add_argument("--ast-weight-where", type=float, default=0.4)
    ap.add_argument("--ast-weight-from", type=float, default=0.1)

    # Acceptance integration
    ap.add_argument(
        "--acceptance-config",
        default=str(root / "method" / "decompose_method" / "seed_acceptance_rules.config"),
        help="Per-question acceptance rules config (.config)",
    )
    ap.add_argument("--acceptance-output-json", default="", help="Optional acceptance per-item output path")

    ap.add_argument(
        "--output-json",
        default=str(root / "results" / "synth_from_decompose_retrieve_top3_gemma.json"),
    )
    ap.add_argument("--output-jsonl", default="", help="Optional compact SQL JSONL output")

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


def match_score_obj(obj: Dict[str, Any]) -> float:
    if "sbert_score" in obj:
        try:
            return float(obj.get("sbert_score") or 0.0)
        except Exception:  # noqa: BLE001
            pass
    try:
        return float(obj.get("total_score") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def extract_where_clause(sql: str) -> str:
    s = str(sql or "").strip().rstrip(";")
    if not s:
        return ""
    m = re.search(
        r"\bwhere\b\s+(.+?)(?:\border\s+by\b|\bgroup\s+by\b|\blimit\b|\boffset\b|$)",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    where_body = re.sub(r"\s+", " ", m.group(1)).strip()
    return where_body


def build_synthesis_messages(
    *,
    question: str,
    decomposed_queries: Sequence[str],
    retrieval_by_decomposition: Sequence[Dict[str, Any]],
    ranked_results: Sequence[Dict[str, Any]],
    base_prompt: str,
    schema_text: str,
    use_pydantic_schema: bool,
    evidence_per_decomp: int,
    include_global_merged: bool,
    where_only_evidence: bool,
) -> List[Dict[str, str]]:
    base = render_base_prompt(base_prompt=base_prompt, schema_text=schema_text, question=question)

    decomp_block = "\n".join([f"- {q}" for q in decomposed_queries]) if decomposed_queries else "- (none)"

    top_n = max(1, int(evidence_per_decomp))
    per_decomp_blocks: List[str] = []
    for group in retrieval_by_decomposition:
        if not isinstance(group, dict):
            continue

        dq = str(group.get("decomposed_query") or "").strip()
        ranked_for_dq = group.get("ranked_results") if isinstance(group.get("ranked_results"), list) else []
        ranked_for_dq = ranked_for_dq[:top_n]

        lines: List[str] = [f"Decomposed Query: {dq or '(empty)'}"]
        for j, one in enumerate(ranked_for_dq, start=1):
            if not isinstance(one, dict):
                continue
            c = one.get("candidate") if isinstance(one.get("candidate"), dict) else {}
            score = match_score_obj(one)
            c_sql = str(c.get("sql") or "")
            where_hint = extract_where_clause(c_sql) if where_only_evidence else ""
            lines.extend(
                [
                    f"  Top {j} | SBERT Score: {score:.4f}",
                    f"  Seed Question: {str(c.get('question') or '')}",
                    f"  Seed WHERE pattern: {where_hint}" if where_hint else "  Seed WHERE pattern: (none)",
                    "  Seed SQL:",
                    c_sql,
                ]
            )

        per_decomp_blocks.append("\n".join(lines))

    per_decomp_text = "\n\n".join(per_decomp_blocks) if per_decomp_blocks else "- (none)"

    candidate_blocks: List[str] = []
    if include_global_merged:
        for i, m in enumerate(ranked_results, start=1):
            if not isinstance(m, dict):
                continue
            candidate = m.get("candidate") if isinstance(m.get("candidate"), dict) else {}
            rank = int(m.get("rank") or i)
            score = match_score_obj(m)
            c_question = str(candidate.get("question") or "")
            c_sql = str(candidate.get("sql") or "")

            candidate_blocks.append(
                "\n".join(
                    [
                        f"Candidate Rank {rank}",
                        f"SBERT Score: {score:.4f}",
                        f"Candidate Question: {c_question}",
                        "Candidate SQL:",
                        c_sql,
                    ]
                )
            )

    out_instruction = (
        'Return ONLY a JSON object with key "sql".'
        if use_pydantic_schema
        else "Output only SQL (no markdown, no explanation)."
    )

    global_block = (
        "Globally merged top retrieved SBERT examples:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        if include_global_merged
        else ""
    )

    user = (
        f"{base}\n\n"
        "Parent question:\n"
        f"{question}\n\n"
        "Decomposed sub-questions used for retrieval:\n"
        f"{decomp_block}\n\n"
        f"Per-decomposed-query top matched seed question+SQL (top {top_n} per decomposition):\n"
        f"{per_decomp_text}\n\n"
        f"{global_block}"
        "Synthesize one best final SQL query for the original question.\n"
        "Use retrieved examples as structural hints; do not copy wrong literals from unrelated seeds.\n"
        f"{out_instruction}\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert SQLite NL2SQL assistant. "
                "Follow instructions exactly and output only valid JSON when schema is provided."
            ),
        },
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
    *,
    client: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    utils_mod: Any,
    response_schema: Optional[Dict[str, Any]],
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
            retryable = utils_mod.is_retryable_provider_error(e)
            if attempt >= attempts or not retryable:
                break
            backoff = min(30.0, (2 ** (attempt - 1)))
            print(f"Retry synthesis attempt={attempt}/{attempts} sleep={backoff:.1f}s err={str(e).splitlines()[0]}")
            time.sleep(backoff)

    return last_err


def run_synthesis_generation(
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
        utils_mod=utils_mod,
        response_schema=response_schema,
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


def run_execution_preview(rsq: Any, args: argparse.Namespace, final_sql: str) -> Dict[str, Any]:
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


def load_retrieval_rows(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        rows = obj.get("results")
        if not isinstance(rows, list):
            raise ValueError(f"Expected key 'results' list in {path}")
        return [r for r in rows if isinstance(r, dict)], obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)], {}

    raise ValueError(f"Unsupported retrieval JSON format: {path}")


def build_error_row(src: Dict[str, Any], err: str) -> Dict[str, Any]:
    return {
        "task_index": src.get("task_index"),
        "question_index": src.get("question_index"),
        "item_id": src.get("item_id"),
        "input_question": src.get("input_question") or src.get("question") or "",
        "error": err,
        "decomposition": src.get("decomposition") if isinstance(src.get("decomposition"), dict) else {},
        "retrieval_trace": src.get("retrieval_trace") if isinstance(src.get("retrieval_trace"), list) else [],
        "retrieval_by_decomposition": src.get("retrieval_by_decomposition") if isinstance(src.get("retrieval_by_decomposition"), list) else [],
        "ranked_results": src.get("ranked_results") if isinstance(src.get("ranked_results"), list) else [],
        "synthesized": {
            "raw_text": "",
            "final_sql": "",
            "field_logprobs": {},
            "field_confidence": {},
            "confidence_overall": None,
            "model_meta": {},
        },
        "execution_preview": {
            "columns": [],
            "rows": [],
            "error": err,
        },
    }


def to_sql_jsonl_row(r: Dict[str, Any]) -> Dict[str, Any]:
    synth = r.get("synthesized") if isinstance(r.get("synthesized"), dict) else {}
    q = r.get("input_question")
    return {
        "task_index": r.get("task_index"),
        "question_index": r.get("question_index"),
        "item_id": r.get("item_id"),
        "question": q,
        "natural_question": q,
        "pred_sql": (synth.get("final_sql") or "").strip(),
        "confidence_overall": synth.get("confidence_overall"),
        "error": r.get("error"),
    }


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
    return ">=", parse_threshold_value(t)


def ensure_default_acceptance_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Seed acceptance rules config",
                "main_metric=exact_exec_match",
                "main_operator===",
                "main_threshold=true",
                "required_non_main=2",
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
        if not line or line.startswith("#") or "=" not in line:
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

    if isinstance(threshold, bool):
        ab = _to_bool(actual)
        if ab is None:
            return False
        if op == "==":
            return ab == threshold
        if op == "!=":
            return ab != threshold
        return False

    if isinstance(threshold, (int, float)):
        try:
            av = float(actual)
            tv = float(threshold)
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
        checks.append({"metric": name, "value": val, "operator": op, "threshold": thr, "pass": ok})
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


def evaluate_acceptance(eval_items: Sequence[Dict[str, Any]], rules: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    summary = {
        "evaluated": 0,
        "accepted": 0,
        "accepted_main_metric": 0,
        "accepted_non_main": 0,
        "rejected": 0,
    }

    for ev in eval_items:
        if not isinstance(ev, dict):
            continue
        d = acceptance_for_item(ev, rules)
        decisions.append(
            {
                "item_id": ev.get("item_id"),
                "question": ev.get("question"),
                "exact_exec_match": ev.get("exact_exec_match"),
                "sql_ast_similarity": ev.get("sql_ast_similarity"),
                "chrf": ev.get("chrf"),
                "rouge_l_f1": ev.get("rouge_l_f1"),
                "decision": d,
            }
        )

        summary["evaluated"] += 1
        if d.get("accepted"):
            summary["accepted"] += 1
            if d.get("reason") == "main_metric":
                summary["accepted_main_metric"] += 1
            else:
                summary["accepted_non_main"] += 1
        else:
            summary["rejected"] += 1

    return decisions, summary


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


def derived_path(base_out: Path, suffix: str) -> Path:
    name = base_out.name
    stem = name[:-5] if name.lower().endswith(".json") else name
    return base_out.with_name(f"{stem}{suffix}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_row(
    *,
    src: Dict[str, Any],
    args: argparse.Namespace,
    rsq: Any,
    utils_mod: Any,
    base_prompt: str,
    schema_text: str,
    backend_state: Dict[str, Any],
    synth_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    question = str(src.get("input_question") or src.get("question") or "").strip()
    ranked_results = src.get("ranked_results") if isinstance(src.get("ranked_results"), list) else []
    retrieval_by_decomposition = src.get("retrieval_by_decomposition") if isinstance(src.get("retrieval_by_decomposition"), list) else []

    if not question:
        return build_error_row(src, "EMPTY_QUESTION")
    if not ranked_results:
        return build_error_row(src, "EMPTY_RETRIEVAL_RESULTS")

    try:
        decomp = src.get("decomposition") if isinstance(src.get("decomposition"), dict) else {}
        decomposed_queries = decomp.get("decomposed_queries") if isinstance(decomp.get("decomposed_queries"), list) else []

        messages = build_synthesis_messages(
            question=question,
            decomposed_queries=[str(x) for x in decomposed_queries if str(x).strip()],
            retrieval_by_decomposition=retrieval_by_decomposition,
            ranked_results=ranked_results,
            base_prompt=base_prompt,
            schema_text=schema_text,
            use_pydantic_schema=bool(int(args.use_pydantic_schema)),
            evidence_per_decomp=int(args.synth_evidence_per_decomp),
            include_global_merged=bool(int(args.synth_include_global_merged)),
            where_only_evidence=bool(int(args.synth_where_only_evidence)),
        )

        use_structured = args.backend == "openai_compat" and args.logprob_mode == "structured"
        use_token_logprobs = bool(int(args.logprobs))
        gen = run_synthesis_generation(
            args=args,
            backend_state=backend_state,
            utils_mod=utils_mod,
            messages=messages,
            response_schema=synth_schema,
            use_logprobs=(use_structured or use_token_logprobs),
        )
        if gen.get("error"):
            return build_error_row(src, str(gen.get("error")))

        raw_text = str(gen.get("raw_text") or "")
        final_sql = ""
        if bool(int(args.use_pydantic_schema)):
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    final_sql = str(parsed.get("sql") or "").strip()
            except Exception as e:  # noqa: BLE001
                gen.setdefault("model_meta", {})["response_parse_error"] = str(e).splitlines()[0]

        if not final_sql:
            final_sql = extract_sql(raw_text)
        if not final_sql:
            return build_error_row(src, "FAILED_TO_EXTRACT_FINAL_SQL")

        field_lp: Dict[str, Any] = {}
        field_conf: Dict[str, Any] = {}
        conf_overall: Optional[float] = None
        if args.backend == "openai_compat":
            if args.logprob_mode == "structured":
                try:
                    field_lp, field_conf, conf_overall = utils_mod.structured_logprob_payload(
                        gen.get("completion"), add_logprobs
                    )
                except Exception as lp_err:  # noqa: BLE001
                    gen.setdefault("model_meta", {})["structured_logprob_error"] = str(lp_err).splitlines()[0]
                    try:
                        field_lp, field_conf, conf_overall = utils_mod.openai_token_logprob_payload(gen.get("completion"))
                        gen.setdefault("model_meta", {})["logprob_fallback"] = "openai_token_logprobs"
                    except Exception:
                        conf_overall = None
            elif bool(int(args.logprobs)):
                try:
                    field_lp, field_conf, conf_overall = utils_mod.openai_token_logprob_payload(gen.get("completion"))
                except Exception:
                    conf_overall = None

        execution = run_execution_preview(rsq, args, final_sql)

        return {
            "task_index": src.get("task_index"),
            "question_index": src.get("question_index"),
            "item_id": src.get("item_id"),
            "input_question": question,
            "error": None,
            "decomposition": src.get("decomposition") if isinstance(src.get("decomposition"), dict) else {},
            "retrieval_trace": src.get("retrieval_trace") if isinstance(src.get("retrieval_trace"), list) else [],
            "retrieval_by_decomposition": retrieval_by_decomposition,
            "ranked_results": ranked_results,
            "synthesized": {
                "raw_text": raw_text,
                "final_sql": final_sql,
                "field_logprobs": field_lp,
                "field_confidence": field_conf,
                "confidence_overall": conf_overall,
                "model_meta": gen.get("model_meta") if isinstance(gen.get("model_meta"), dict) else {},
            },
            "execution_preview": execution,
        }
    except Exception as e:  # noqa: BLE001
        return build_error_row(src, f"TASK_PROCESS_ERROR:{e}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.backend == "openai_compat":
        if OpenAI is None:
            raise RuntimeError("openai package is required for backend=openai_compat")
        if bool(int(args.use_pydantic_schema)):
            if BaseModel is None or ResponseFormatJSONSchema is None:
                raise RuntimeError("pydantic + openai.types.ResponseFormatJSONSchema are required when --use-pydantic-schema=1")
        if args.logprob_mode == "structured" and add_logprobs is None:
            raise RuntimeError("structured_logprobs is required for --logprob-mode structured (pip install structured-logprobs)")

    rsq = import_retriever(root)
    utils_mod = import_utils(root)

    base_prompt = load_template_text(args.prompt_file)
    schema_text = load_schema_text(args.schema_json)

    source_rows, source_meta = load_retrieval_rows(args.retrieval_json)
    if not source_rows:
        raise SystemExit("No rows found in retrieval JSON.")

    backend_state: Dict[str, Any] = {}
    if args.backend == "vllm_local":
        tokenizer, llm, sampling = init_vllm_local(args)
        backend_state = {"tokenizer": tokenizer, "llm": llm, "sampling": sampling}
    else:
        backend_state = {"client": OpenAI(base_url=args.api_base, api_key=args.api_key)}

    synth_schema = build_response_schema(SynthResponse, bool(int(args.use_pydantic_schema))) if BaseModel is not None else None

    total_tasks = len(source_rows)
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    chunk_size = max(1, int(args.gen_batch_size))
    workers = 1 if args.backend == "vllm_local" else max(1, int(args.batch_concurrency))

    for b0 in range(0, total_tasks, chunk_size):
        chunk = source_rows[b0 : b0 + chunk_size]

        if workers <= 1 or len(chunk) == 1:
            for i, src in enumerate(chunk):
                row = process_row(
                    src=src,
                    args=args,
                    rsq=rsq,
                    utils_mod=utils_mod,
                    base_prompt=base_prompt,
                    schema_text=schema_text,
                    backend_state=backend_state,
                    synth_schema=synth_schema,
                )
                key = b0 + i
                results_by_idx[key] = row
        else:
            max_workers = min(workers, len(chunk))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_idx: Dict[Any, int] = {}
                for i, src in enumerate(chunk):
                    fut = ex.submit(
                        process_row,
                        src=src,
                        args=args,
                        rsq=rsq,
                        utils_mod=utils_mod,
                        base_prompt=base_prompt,
                        schema_text=schema_text,
                        backend_state=backend_state,
                        synth_schema=synth_schema,
                    )
                    fut_to_idx[fut] = b0 + i

                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    src = chunk[idx - b0]
                    try:
                        row = fut.result()
                    except Exception as e:  # noqa: BLE001
                        row = build_error_row(src, f"TASK_PROCESS_ERROR:{e}")
                    results_by_idx[idx] = row

        done = min(b0 + len(chunk), total_tasks)
        print(f"Processed {done}/{total_tasks}")

    results = [results_by_idx[i] for i in range(total_tasks)]

    counts = {
        "total_tasks": len(results),
        "error_tasks": sum(1 for r in results if r.get("error")),
        "generated_nonempty_sql": sum(1 for r in results if (r.get("synthesized", {}).get("final_sql") or "").strip()),
        "exec_ok": sum(1 for r in results if not (r.get("execution_preview", {}).get("error"))),
        "with_confidence": sum(1 for r in results if isinstance(r.get("synthesized", {}).get("confidence_overall"), (int, float))),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_ready_rows = [to_sql_jsonl_row(r) for r in results]
    eval_ready_json_path = Path(args.eval_ready_json) if (args.eval_ready_json or "").strip() else derived_path(out_path, ".eval_ready.json")
    eval_ready_jsonl_path = (
        Path(args.eval_ready_jsonl)
        if (args.eval_ready_jsonl or "").strip()
        else (Path(args.output_jsonl) if (args.output_jsonl or "").strip() else derived_path(out_path, ".eval_ready.jsonl"))
    )

    write_json(eval_ready_json_path, eval_ready_rows)
    write_jsonl(eval_ready_jsonl_path, eval_ready_rows)

    eval_ran = False
    eval_output_json_path: Optional[Path] = None
    eval_summary: Dict[str, Any] = {}
    eval_items: List[Dict[str, Any]] = []

    if int(args.run_eval) == 1:
        eval_output_json_path = Path(args.eval_output_json) if (args.eval_output_json or "").strip() else derived_path(out_path, ".eval.json")
        run_eval_v2(
            eval_script=Path(args.eval_script),
            pred_path=eval_ready_json_path,
            gt_path=Path(args.gt_json),
            db_path=Path(args.db_path),
            output_json=eval_output_json_path,
            compute_bertscore=int(args.compute_bertscore),
            ast_weight_select=float(args.ast_weight_select),
            ast_weight_where=float(args.ast_weight_where),
            ast_weight_from=float(args.ast_weight_from),
        )
        eval_ran = True

        ev_obj = json.loads(eval_output_json_path.read_text(encoding="utf-8"))
        if isinstance(ev_obj, dict):
            eval_summary = ev_obj.get("summary") if isinstance(ev_obj.get("summary"), dict) else {}
            raw_items = ev_obj.get("per_item")
            if isinstance(raw_items, list):
                eval_items = [x for x in raw_items if isinstance(x, dict)]

    rules_path = Path(args.acceptance_config)
    rules = load_acceptance_rules(rules_path)

    acceptance_decisions: List[Dict[str, Any]] = []
    acceptance_summary: Dict[str, Any] = {
        "evaluated": 0,
        "accepted": 0,
        "accepted_main_metric": 0,
        "accepted_non_main": 0,
        "rejected": 0,
    }
    acceptance_output_path: Optional[Path] = None

    if eval_items:
        acceptance_decisions, acceptance_summary = evaluate_acceptance(eval_items, rules)
        acceptance_output_path = (
            Path(args.acceptance_output_json)
            if (args.acceptance_output_json or "").strip()
            else derived_path(out_path, ".acceptance.json")
        )

        by_id: Dict[str, Dict[str, Any]] = {}
        for d in acceptance_decisions:
            k = d.get("item_id")
            if k is not None:
                by_id[str(k)] = d

        for r in results:
            rid = r.get("item_id")
            if rid is None:
                continue
            dec = by_id.get(str(rid))
            if dec is not None:
                r["acceptance"] = dec

        write_json(
            acceptance_output_path,
            {
                "config_path": str(rules_path),
                "rules": rules,
                "summary": acceptance_summary,
                "per_item": acceptance_decisions,
            },
        )

    payload: Dict[str, Any] = {
        "meta": {
            "pipeline_stage": "synthesis_from_decomposition_retrieval",
            "retrieval_json": args.retrieval_json,
            "retrieval_meta": source_meta,
            "retrieval_contract": "expects per-decomposed top-k in retrieval_by_decomposition when available",
            "backend": args.backend,
            "prompt_file": args.prompt_file,
            "schema_json": args.schema_json,
            "api_base": args.api_base if args.backend == "openai_compat" else None,
            "model_name": args.model_name if args.backend == "openai_compat" else None,
            "model_path": args.model_path if args.backend == "vllm_local" else None,
            "db_path": args.db_path,
            "skip_exec": int(args.skip_exec),
            "logprobs": int(args.logprobs),
            "logprob_mode": args.logprob_mode,
            "use_pydantic_schema": bool(int(args.use_pydantic_schema)),
            "synth_evidence_per_decomp": int(args.synth_evidence_per_decomp),
            "synth_include_global_merged": bool(int(args.synth_include_global_merged)),
            "synth_where_only_evidence": bool(int(args.synth_where_only_evidence)),
            "gen_batch_size": int(args.gen_batch_size),
            "batch_concurrency": int(args.batch_concurrency),
            "eval": {
                "run_eval": int(args.run_eval),
                "gt_json": args.gt_json,
                "eval_script": args.eval_script,
                "eval_ready_json": str(eval_ready_json_path),
                "eval_ready_jsonl": str(eval_ready_jsonl_path),
                "eval_output_json": str(eval_output_json_path) if eval_output_json_path else None,
                "eval_ran": eval_ran,
            },
            "acceptance": {
                "config_path": str(rules_path),
                "config_format": "seed_acceptance_rules.config",
                "acceptance_output_json": str(acceptance_output_path) if acceptance_output_path else None,
            },
        },
        "counts": counts,
        "eval_summary": eval_summary,
        "acceptance_summary": acceptance_summary,
        "results": results,
    }

    write_json(out_path, payload)

    print("Saved JSON:", out_path)
    print("Saved eval-ready JSON:", eval_ready_json_path)
    print("Saved eval-ready JSONL:", eval_ready_jsonl_path)
    print("Tasks:", counts["total_tasks"])
    print("Errors:", counts["error_tasks"])
    print("Generated SQL:", counts["generated_nonempty_sql"])
    if int(args.skip_exec) != 1:
        print("Exec ok:", counts["exec_ok"])
    print("Rows with confidence_overall:", counts["with_confidence"])

    if eval_ran:
        print("Eval output:", eval_output_json_path)
        print("Acceptance config:", rules_path)
        if acceptance_output_path is not None:
            print("Acceptance output:", acceptance_output_path)


if __name__ == "__main__":
    main()
