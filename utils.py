#!/usr/bin/env python3
"""Utility helpers for baseline runners."""

import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InputPayload:
    records: List[Dict[str, Any]]
    format_used: str


def setup_logger(log_dir: str, output_path: str, logger_name: str = "run_baselines") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(output_path))[0] or logger_name
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}_{ts}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Log file: %s", log_path)
    return logger


def is_retryable_provider_error(exc: Exception) -> bool:
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


def extract_sql(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\bSELECT\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():].strip()
    m2 = re.search(r"\n\s*SELECT\b", t, flags=re.IGNORECASE)
    if m2:
        t = t[:m2.start()].strip()
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


def quote_ident(name: str) -> str:
    return '"' + (name or "").replace('"', '""') + '"'


def sql_literal_for_prompt(v: Any, max_len: int = 80) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len] + "..."
    return "'" + s.replace("'", "''") + "'"


def fetch_schema_columns_from_json(schema_file: str) -> List[str]:
    with open(schema_file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        out = [str(x).strip() for x in obj if str(x).strip()]
        if out:
            return out
    raise ValueError(f"Unsupported schema JSON in {schema_file}; expected JSON list of column names.")


def fetch_schema_columns_from_db(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}");')
    return [r[1] for r in cur.fetchall()]


def fetch_schema_value_hints(
    conn: sqlite3.Connection,
    table: str,
    schema_cols: List[str],
    max_values_per_col: int = 12,
    max_distinct_full: int = 40,
    max_value_chars: int = 80,
) -> Dict[str, Dict[str, Any]]:
    hints: Dict[str, Dict[str, Any]] = {}
    table_q = quote_ident(table)

    for col in schema_cols:
        col_q = quote_ident(col)
        try:
            total_row = conn.execute(
                f"SELECT COUNT(DISTINCT {col_q}) FROM {table_q} WHERE {col_q} IS NOT NULL;"
            ).fetchone()
            total_distinct = int((total_row or [0])[0] or 0)
            if total_distinct <= 0:
                continue

            if total_distinct <= max_distinct_full:
                cur = conn.execute(
                    f"SELECT DISTINCT {col_q} FROM {table_q} "
                    f"WHERE {col_q} IS NOT NULL ORDER BY {col_q};"
                )
                values = [r[0] for r in cur.fetchall()]
            else:
                cur = conn.execute(
                    f"SELECT {col_q}, COUNT(*) AS _cnt FROM {table_q} "
                    f"WHERE {col_q} IS NOT NULL GROUP BY {col_q} "
                    f"ORDER BY _cnt DESC, {col_q} LIMIT ?;",
                    (int(max(1, max_values_per_col)),),
                )
                values = [r[0] for r in cur.fetchall()]

            rendered = [sql_literal_for_prompt(v, max_len=max_value_chars) for v in values]
            hints[col] = {
                "values": rendered,
                "shown": len(rendered),
                "total_distinct": total_distinct,
                "truncated": total_distinct > len(rendered),
            }
        except Exception:
            continue
    return hints


def parse_question_keys(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def detect_input_format(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            return "json" if t.startswith("[") else "jsonl"
    return "jsonl"


def load_input(path: str, input_format: str) -> InputPayload:
    fmt = detect_input_format(path) if input_format == "auto" else input_format

    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("JSON input must be a list of objects")
        return InputPayload(records=[x for x in obj if isinstance(x, dict)], format_used="json")

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return InputPayload(records=rows, format_used="jsonl")


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"completed": {}, "meta": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("completed"), dict):
            return obj
    except Exception:
        pass
    return {"completed": {}, "meta": {}}


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compose_output_rows(
    rows: List[Dict[str, Any]],
    completed: Dict[str, Dict[str, Any]],
    id_key: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        item_id = str(row.get(id_key) or f"idx_{i}")
        merged = dict(row)
        if item_id in completed:
            merged.update(completed[item_id])
        out.append(merged)
    return out


def persist_outputs(
    *,
    output_path: str,
    output_format: str,
    input_rows: List[Dict[str, Any]],
    completed: Dict[str, Dict[str, Any]],
    id_key: str,
    checkpoint_path: str,
    checkpoint_meta: Dict[str, Any],
) -> None:
    merged = compose_output_rows(input_rows, completed, id_key)
    if output_format == "json":
        write_json(output_path, merged)
    else:
        write_jsonl(output_path, merged)
    write_json(checkpoint_path, {"completed": completed, "meta": checkpoint_meta})


def exp_structure(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: exp_structure(v) for k, v in x.items()}
    if isinstance(x, list):
        return [exp_structure(v) for v in x]
    if isinstance(x, (int, float)):
        try:
            return float(math.exp(x))
        except Exception:
            return None
    return x


def flatten_numbers(x: Any) -> List[float]:
    if isinstance(x, dict):
        vals: List[float] = []
        for v in x.values():
            vals.extend(flatten_numbers(v))
        return vals
    if isinstance(x, list):
        vals = []
        for v in x:
            vals.extend(flatten_numbers(v))
        return vals
    if isinstance(x, (int, float)):
        return [float(x)]
    return []


def prompt_schema_inline(schema_cols: List[str]) -> str:
    return ", ".join([f'"{c}"' for c in schema_cols])


def render_schema_hints(schema_cols: List[str], hints: Dict[str, Dict[str, Any]]) -> str:
    if not hints:
        return ""
    lines = ["Observed unique row values for schema:"]
    for col in schema_cols:
        hint = hints.get(col)
        if not hint:
            continue
        values = hint.get("values") or []
        if not values:
            continue
        values_inline = ", ".join([str(v) for v in values])
        shown = int(hint.get("shown", len(values)))
        total = int(hint.get("total_distinct", shown))
        if bool(hint.get("truncated", False)):
            lines.append(f'- "{col}": {values_inline} (showing {shown}/{total})')
        else:
            lines.append(f'- "{col}": {values_inline}')
    return "\n".join(lines).strip()


def build_prompt(
    *,
    question: str,
    prompt_style: str,
    zs_template: str,
    few_shot_text: str,
    schema_cols: List[str],
    schema_hints_text: str,
    cot_suffix: str,
) -> str:
    q = (question or "").strip()
    schema_inline = prompt_schema_inline(schema_cols)
    # Base prompt for all styles is always the zero-shot template.
    prompt = (
        zs_template.replace("{SCHEMA}", schema_inline)
        .replace("{QUESTION}", q)
        .replace("{Question}", q)
    )

    if schema_hints_text and "Observed unique row values for schema:" not in prompt:
        prompt = prompt.rstrip() + "\n\n" + schema_hints_text + "\n\nQuestion: {QUESTION}\nGenerate SQL:"
        prompt = prompt.replace("{QUESTION}", q).replace("{Question}", q)

    if prompt_style == "few_shot":
        prompt = prompt.rstrip() + "\n\nFew-shot examples:\n" + few_shot_text.strip()
    elif prompt_style == "cot":
        prompt = prompt.rstrip() + "\n\n" + cot_suffix.strip()
    return prompt


def chunked_indices(n: int, batch_size: int) -> List[Tuple[int, int]]:
    bsz = max(1, int(batch_size))
    out: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(n, i + bsz)
        out.append((i, j))
        i = j
    return out


def pick_question(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def openai_completion_to_meta(completion: Any) -> Dict[str, Any]:
    choices = getattr(completion, "choices", []) or []
    c0 = choices[0] if choices else None
    usage = getattr(completion, "usage", None)
    return {
        "provider_call_ok": True,
        "finish_reason": getattr(c0, "finish_reason", None) if c0 is not None else None,
        "stop_reason": None,
        "token_ids_len": getattr(usage, "completion_tokens", None) if usage is not None else None,
    }


def parse_openai_text(completion: Any) -> str:
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


def structured_logprob_payload(
    completion: Any,
    add_logprobs_fn: Optional[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[float]]:
    if add_logprobs_fn is None:
        return {}, {}, None
    wrapped = add_logprobs_fn(completion)
    log_probs = getattr(wrapped, "log_probs", None)
    field_logprobs = log_probs[0] if isinstance(log_probs, list) and log_probs else {}
    field_confidence = exp_structure(field_logprobs)
    nums = flatten_numbers(field_logprobs)
    conf_overall = float(math.exp(sum(nums) / len(nums))) if nums else None
    return field_logprobs, field_confidence, conf_overall


def openai_token_logprob_payload(completion: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[float]]:
    """Fallback logprob extraction from OpenAI-compatible token logprobs."""
    choices = getattr(completion, "choices", []) or []
    c0 = choices[0] if choices else None
    if c0 is None:
        return {}, {}, None

    c0_logprobs = getattr(c0, "logprobs", None)
    content = getattr(c0_logprobs, "content", None) if c0_logprobs is not None else None
    if not isinstance(content, list):
        return {}, {}, None

    tokens: List[str] = []
    token_logprobs: List[Optional[float]] = []
    for tok in content:
        token_text = getattr(tok, "token", None)
        lp = getattr(tok, "logprob", None)
        tokens.append(str(token_text) if token_text is not None else "")
        token_logprobs.append(float(lp) if isinstance(lp, (int, float)) else None)

    finite_lps = [x for x in token_logprobs if isinstance(x, (int, float))]
    conf_overall = float(math.exp(sum(finite_lps) / len(finite_lps))) if finite_lps else None
    field_logprobs = {
        "token_logprobs": token_logprobs,
        "token_count": len(tokens),
        "tokens": tokens,
    }
    field_confidence = {"token_confidence": exp_structure(token_logprobs)}
    return field_logprobs, field_confidence, conf_overall


def build_checkpoint_meta(args: Any, total_rows: int, input_format: str) -> Dict[str, Any]:
    return {
        "backend": "openai_compat",
        "model": args.model,
        "prompt_style": args.prompt_style,
        "logprob_mode": args.logprob_mode,
        "total_selected": total_rows,
        "input_format_used": input_format,
    }


def make_row_result(
    *,
    item_id: str,
    question: str,
    prompt_style: str,
    backend: str,
    model: str,
    raw_model_output: str,
    pred_sql: str,
    logprob_mode: str,
    field_logprobs: Dict[str, Any],
    field_confidence: Dict[str, Any],
    confidence_overall: Optional[float],
    model_meta: Dict[str, Any],
    error: Optional[str],
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "question_used": question,
        "prompt_style": prompt_style,
        "backend": backend,
        "model": model,
        "raw_model_output": raw_model_output,
        "pred_sql": pred_sql,
        "pred_sql_extracted_ok": bool(pred_sql),
        "logprob_mode": logprob_mode,
        "field_logprobs": field_logprobs or {},
        "field_confidence": field_confidence or {},
        "confidence_overall": confidence_overall,
        "model_meta": model_meta or {},
        "error": error,
    }
