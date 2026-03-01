#!/usr/bin/env python3
"""
model_baseline.py

Baseline runner:
- Read question from input JSON (default: generated_question)
- Generate predicted SQL with vLLM or GCP (Vertex via LiteLLM)
- Execute predicted SQL to get answer rows
- Evaluate predicted SQL against EXISTING gold SQL in input JSON (default key: new_sql)
  using Hungarian matching + ROUGE-1 F1 (cell-level)
- Save resumable checkpoints

Important: This script does NOT generate new_sql. It only uses gold SQL from input.
"""

import argparse
import json
import os
import re
import sqlite3
import time
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

try:
    from litellm import completion as litellm_completion
except Exception:
    litellm_completion = None


# -------------------------
# SQL/Text helpers
# -------------------------
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def quote_ident(name: str) -> str:
    return '"' + (name or "").replace('"', '""') + '"'


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


def canonical_value(v: Any) -> Any:
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)


def sql_literal_for_prompt(v: Any, max_len: int = 80) -> str:
    v = canonical_value(v)
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len] + "..."
    return "'" + s.replace("'", "''") + "'"


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def tokenize_text(s: str) -> List[str]:
    s = normalize_text(s)
    return re.findall(r"[a-z0-9]+(?:[+/_-][a-z0-9]+)*", s)


def rouge1_f1(a: str, b: str) -> float:
    ta = tokenize_text(a)
    tb = tokenize_text(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum((ca & cb).values())
    p = overlap / len(ta)
    r = overlap / len(tb)
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def cell_similarity(a: Any, b: Any, text_metric: str = "rouge1_f1") -> float:
    a = canonical_value(a)
    b = canonical_value(b)

    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 1.0 if float(a) == float(b) else 0.0

    if isinstance(a, str) or isinstance(b, str):
        sa = str(a)
        sb = str(b)
        if text_metric == "exact":
            return 1.0 if normalize_text(sa) == normalize_text(sb) else 0.0
        return rouge1_f1(sa, sb)

    return 1.0 if a == b else 0.0


def row_similarity(row_a: Tuple[Any, ...], row_b: Tuple[Any, ...], text_metric: str = "rouge1_f1") -> float:
    if len(row_a) != len(row_b):
        return 0.0
    if not row_a:
        return 1.0
    s = 0.0
    for va, vb in zip(row_a, row_b):
        s += cell_similarity(va, vb, text_metric=text_metric)
    return s / len(row_a)


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def rows_to_dicts(cols: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d: Dict[str, Any] = {}
        for c, v in zip(cols, r):
            d[c] = canonical_value(v)
        out.append(d)
    return out


# -------------------------
# DB helpers
# -------------------------
def fetch_schema(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}");')
    rows = cur.fetchall()
    return [r[1] for r in rows]


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


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


# -------------------------
# Hungarian eval
# -------------------------
def hungarian_min_cost_square(cost: List[List[float]]) -> Tuple[List[int], float]:
    n = len(cost)
    if n == 0:
        return [], 0.0
    for row in cost:
        if len(row) != n:
            raise ValueError("cost matrix must be square")

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1

    total = 0.0
    for i in range(n):
        j = assignment[i]
        if j >= 0:
            total += cost[i][j]
    return assignment, total


def multiset_metrics(pred_proj: List[Tuple[Any, ...]], gold_proj: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    cp = Counter(pred_proj)
    cg = Counter(gold_proj)

    overlap = sum((cp & cg).values())
    union = sum((cp | cg).values())

    p = safe_div(overlap, len(pred_proj)) if pred_proj else (1.0 if not gold_proj else 0.0)
    r = safe_div(overlap, len(gold_proj)) if gold_proj else (1.0 if not pred_proj else 0.0)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    jacc = safe_div(overlap, union) if union > 0 else 1.0

    return {
        "hard_overlap_rows": int(overlap),
        "soft_overlap_score": float(overlap),
        "precision": p,
        "recall": r,
        "f1": f1,
        "row_jaccard": jacc,
        "relaxed_em_unordered": (cp == cg),
    }


def hungarian_metrics(
    pred_proj: List[Tuple[Any, ...]],
    gold_proj: List[Tuple[Any, ...]],
    text_metric: str,
) -> Dict[str, Any]:
    m = len(pred_proj)
    n = len(gold_proj)
    k = max(m, n)

    if k == 0:
        return {
            "hard_overlap_rows": 0,
            "soft_overlap_score": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "row_jaccard": 1.0,
            "relaxed_em_unordered": True,
        }

    sim = [[0.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = row_similarity(pred_proj[i], gold_proj[j], text_metric=text_metric)

    cost = [[1.0 - sim[i][j] for j in range(k)] for i in range(k)]
    assignment, _ = hungarian_min_cost_square(cost)

    soft_overlap = 0.0
    hard_overlap = 0
    for i in range(m):
        j = assignment[i]
        if 0 <= j < n:
            s = sim[i][j]
            soft_overlap += s
            if s >= 1.0 - 1e-12:
                hard_overlap += 1

    p = safe_div(soft_overlap, m) if m > 0 else (1.0 if n == 0 else 0.0)
    r = safe_div(soft_overlap, n) if n > 0 else (1.0 if m == 0 else 0.0)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    denom = (m + n - soft_overlap)
    jacc = safe_div(soft_overlap, denom) if denom > 0 else 1.0

    relaxed_em = (m == n and hard_overlap == m)
    return {
        "hard_overlap_rows": int(hard_overlap),
        "soft_overlap_score": float(soft_overlap),
        "precision": p,
        "recall": r,
        "f1": f1,
        "row_jaccard": jacc,
        "relaxed_em_unordered": relaxed_em,
    }


def evaluate_pair(
    conn: sqlite3.Connection,
    pred_sql: str,
    gold_sql: str,
    max_rows: int,
    match_mode: str,
    hungarian_max_rows: int,
    cell_text_metric: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred_exec_ok": False,
        "gold_exec_ok": False,
        "eval_ok": False,
        "error": None,
        "pred_row_count": 0,
        "gold_row_count": 0,
        "pred_col_count": 0,
        "gold_col_count": 0,
        "common_cols_count": 0,
        "common_columns": [],
        "hard_overlap_rows": 0,
        "soft_overlap_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "row_jaccard": 0.0,
        "relaxed_em_unordered": False,
        "strict_projection_match": False,
        "match_mode_requested": match_mode,
        "match_mode_used": match_mode,
        "match_mode_fallback_reason": None,
        "cell_text_metric_requested": cell_text_metric,
        "cell_text_metric_used": cell_text_metric if match_mode == "hungarian" else None,
    }

    try:
        pred_cols, pred_rows = execute_sql_fetch(conn, pred_sql, max_rows=max_rows)
        out["pred_exec_ok"] = True
    except Exception as e:
        out["error"] = f"PRED_EXEC_ERROR: {e}"
        return out

    try:
        gold_cols, gold_rows = execute_sql_fetch(conn, gold_sql, max_rows=max_rows)
        out["gold_exec_ok"] = True
    except Exception as e:
        out["error"] = f"GOLD_EXEC_ERROR: {e}"
        return out

    out["pred_col_count"] = len(pred_cols)
    out["gold_col_count"] = len(gold_cols)
    out["pred_row_count"] = len(pred_rows)
    out["gold_row_count"] = len(gold_rows)
    out["strict_projection_match"] = pred_cols == gold_cols

    common_cols = [c for c in pred_cols if c in gold_cols]
    out["common_cols_count"] = len(common_cols)
    out["common_columns"] = common_cols

    if not common_cols:
        out["error"] = "NO_COMMON_COLS"
        return out

    pred_idx = [pred_cols.index(c) for c in common_cols]
    gold_idx = [gold_cols.index(c) for c in common_cols]

    pred_proj = [tuple(canonical_value(r[i]) for i in pred_idx) for r in pred_rows]
    gold_proj = [tuple(canonical_value(r[i]) for i in gold_idx) for r in gold_rows]

    mode_used = match_mode
    if match_mode == "hungarian" and max(len(pred_proj), len(gold_proj)) > max(1, hungarian_max_rows):
        mode_used = "multiset"
        out["match_mode_used"] = mode_used
        out["match_mode_fallback_reason"] = f"result rows exceed --hungarian_max_rows={hungarian_max_rows}"

    if mode_used == "hungarian":
        out["cell_text_metric_used"] = cell_text_metric
        m = hungarian_metrics(pred_proj, gold_proj, text_metric=cell_text_metric)
    else:
        out["cell_text_metric_used"] = None
        m = multiset_metrics(pred_proj, gold_proj)

    out["match_mode_used"] = mode_used
    out["hard_overlap_rows"] = int(m["hard_overlap_rows"])
    out["soft_overlap_score"] = float(m["soft_overlap_score"])
    out["precision"] = float(m["precision"])
    out["recall"] = float(m["recall"])
    out["f1"] = float(m["f1"])
    out["row_jaccard"] = float(m["row_jaccard"])
    out["relaxed_em_unordered"] = bool(m["relaxed_em_unordered"])
    out["eval_ok"] = True
    return out


# -------------------------
# Prompt/model helpers
# -------------------------
def build_prompt(
    question: str,
    table_name: str,
    schema_cols: List[str],
    prompt_style: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    few_shot_text: str = "",
    schema_value_hints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    q = (question or "").strip()
    schema_inline = ", ".join([f'"{c}"' for c in schema_cols])

    base = [
        f'You are a SQL generator. Write one SQLite SELECT query over "{table_name}".',
        "Rules:",
        "- Output ONLY the SQL query (no prose, no markdown, no code fences).",
        "- The SQL MUST start with SELECT.",
        f"- Use only columns from this schema: {schema_inline}",
        "- Put double quotes around EVERY column name exactly as in the schema.",
        "- Use single quotes for string literals.",
        "- Include all constraints from the question in WHERE.",
        "- Do not invent values not grounded in the question.",
    ]

    if prompt_style == "cot":
        base.append("- Think step-by-step internally, but output only final SQL.")

    prompt = "\n".join(base) + "\n\n"

    if schema_value_hints:
        prompt += "Observed values by column (for WHERE value grounding):\n"
        for col in schema_cols:
            hint = schema_value_hints.get(col)
            if not hint:
                continue
            vals = hint.get("values") or []
            if not vals:
                continue
            vals_inline = ", ".join([str(v) for v in vals])
            shown = int(hint.get("shown", len(vals)))
            total_distinct = int(hint.get("total_distinct", shown))
            if bool(hint.get("truncated", False)):
                prompt += f'- "{col}": {vals_inline} (showing {shown}/{total_distinct})\n'
            else:
                prompt += f'- "{col}": {vals_inline}\n'
        prompt += "\n"

    if prompt_style == "few_shot":
        if (few_shot_text or "").strip():
            prompt += "Few-shot examples:\n"
            prompt += few_shot_text.strip() + "\n\n"
            prompt += f"Question: {q}\nSQL:"
            return prompt

        examples = few_shot_examples or []
        if not examples:
            examples = [
                {
                    "question": "List trial IDs and authors for colorectal phase 3 studies.",
                    "sql": 'SELECT "NCT", "Author", "Year", "Cancer type", "Trial phase" FROM clinical_trials WHERE "Cancer type" = \'Colorectal\' AND "Trial phase" = \'Phase 3\';',
                },
                {
                    "question": "Show renal cell nivolumab combination studies.",
                    "sql": 'SELECT "NCT", "Author", "Year", "Name of ICI", "Monotherapy/combination", "Cancer type" FROM clinical_trials WHERE "Cancer type" = \'Renal cell\' AND "Name of ICI" = \'Nivolumab\' AND "Monotherapy/combination" = \'Combination\';',
                },
            ]

        prompt += "Few-shot examples:\n"
        for ex in examples:
            prompt += f"Question: {ex.get('question', '').strip()}\n"
            prompt += f"SQL: {ex.get('sql', '').strip()}\n\n"

    prompt += f"Question: {q}\nSQL:"
    return prompt


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _litellm_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if t is None:
                    t = item.get("content")
                if t is None:
                    t = item.get("value")
                if t is not None:
                    parts.append(str(t))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


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
    ]
    return any(m in s for m in markers)


def generate_sql_with_backend(
    backend: str,
    prompt: str,
    llm: Any,
    sampling: Any,
    *,
    gcp_model: str,
    gcp_project: str,
    vertex_location: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    num_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    if backend == "vllm":
        outs = llm.generate([prompt], sampling)
        out = outs[0] if outs else None
        text = (out.outputs[0].text or "").strip() if out and out.outputs else ""
        meta = {
            "backend": "vllm",
            "provider_call_ok": True,
            "finish_reason": getattr(out.outputs[0], "finish_reason", None) if out and out.outputs else None,
            "stop_reason": getattr(out.outputs[0], "stop_reason", None) if out and out.outputs else None,
            "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []) if out and out.outputs else None,
            "raw_output_chars": len(text),
        }
        return text, meta

    if litellm_completion is None:
        raise RuntimeError("litellm is not installed. Install with: pip install litellm")

    req: Dict[str, Any] = {
        "model": gcp_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "num_retries": num_retries,
    }
    if timeout and timeout > 0:
        req["timeout"] = timeout
    if gcp_project:
        req["vertex_project"] = gcp_project
    if vertex_location:
        req["vertex_location"] = vertex_location

    resp = litellm_completion(**req)
    choices = _obj_get(resp, "choices", []) or []
    choice0 = choices[0] if choices else {}
    message = _obj_get(choice0, "message", {}) or {}
    content = _obj_get(message, "content", "")
    text = _litellm_content_to_text(content).strip()

    usage = _obj_get(resp, "usage", {}) or {}
    completion_tokens = _obj_get(usage, "completion_tokens", None)
    meta = {
        "backend": "gcp",
        "provider_call_ok": True,
        "model": gcp_model,
        "choices_count": len(choices),
        "finish_reason": _obj_get(choice0, "finish_reason", None),
        "stop_reason": _obj_get(choice0, "stop_reason", None),
        "token_ids_len": completion_tokens,
        "raw_output_chars": len(text),
    }
    return text, meta


# -------------------------
# Checkpoint/output helpers
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"completed": {}, "meta": {}}
    try:
        obj = load_json(path)
        if isinstance(obj, dict) and isinstance(obj.get("completed"), dict):
            return obj
    except Exception:
        pass
    return {"completed": {}, "meta": {}}


def compose_output_rows(rows: List[Dict[str, Any]], completed: Dict[str, Dict[str, Any]], item_id_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        item_id = str(row.get(item_id_key) or f"idx_{i}")
        merged = dict(row)
        if item_id in completed:
            merged.update(completed[item_id])
        out.append(merged)
    return out


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--table_name", default="clinical_trials")

    ap.add_argument("--output_json", required=True)
    ap.add_argument("--checkpoint_json", default="")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=10)

    ap.add_argument("--item_id_key", default="item_id")
    ap.add_argument("--question_key", default="generated_question")
    ap.add_argument("--fallback_question_key", default="original_question")
    ap.add_argument("--gold_sql_key", default="new_sql")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_questions", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=0, help="Optional number of questions per batch. <=0 disables batching.")
    ap.add_argument("--batch_index", type=int, default=-1, help="0-based batch index when --batch_size > 0. -1 means process all selected rows.")

    ap.add_argument("--backend", choices=["vllm", "gcp"], default="vllm")

    # vLLM args
    ap.add_argument("--model_path", default="")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    # GCP/LiteLLM args
    ap.add_argument("--gcp_project", default="praxis-flight-482822-q2")
    ap.add_argument("--vertex_location", default="us-central1")
    ap.add_argument("--gcp_model", default="vertex_ai/gemini-2.5-flash")
    ap.add_argument("--litellm_timeout", type=float, default=120.0)
    ap.add_argument("--litellm_num_retries", type=int, default=2)
    ap.add_argument("--rate_limit_retry_attempts", type=int, default=4, help="Outer retry attempts on rate-limit/transient provider errors")
    ap.add_argument("--retry_backoff_seconds", type=float, default=2.0, help="Initial backoff (seconds) between retries")
    ap.add_argument("--retry_backoff_multiplier", type=float, default=2.0, help="Exponential multiplier for retry backoff")
    ap.add_argument("--retry_max_backoff_seconds", type=float, default=30.0, help="Maximum backoff (seconds) between retries")

    # prompt/gen settings
    ap.add_argument("--prompt_style", choices=["zero_shot", "cot", "few_shot"], default="zero_shot")
    ap.add_argument("--few_shot_json", default="", help="Optional JSON list [{question, sql}, ...]")
    ap.add_argument("--few_shot_txt", default="", help="Optional TXT few-shot prompt block to prepend when --prompt_style few_shot")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)

    # answer/eval settings
    ap.add_argument("--answer_max_rows", type=int, default=50)
    ap.add_argument("--eval_max_rows", type=int, default=10000)
    ap.add_argument("--match_mode", choices=["hungarian", "multiset"], default="hungarian")
    ap.add_argument("--hungarian_max_rows", type=int, default=400)
    ap.add_argument("--cell_text_metric", choices=["rouge1_f1", "exact"], default="rouge1_f1")

    # misc
    ap.add_argument("--save_prompts", type=int, default=0)
    ap.add_argument("--print_per_item", type=int, default=1, help="1=print per-item eval metrics during run")
    ap.add_argument("--print_model_check", type=int, default=1, help="1=print model-call diagnostics per item")
    args = ap.parse_args()

    if not args.checkpoint_json:
        args.checkpoint_json = args.output_json + ".checkpoint.json"

    if args.backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    if not schema_cols:
        conn.close()
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")
    schema_value_hints = fetch_schema_value_hints(
        conn=conn,
        table=args.table_name,
        schema_cols=schema_cols,
    )

    rows = load_json(args.input_json)
    if not isinstance(rows, list):
        conn.close()
        raise ValueError("input_json must contain a JSON list")

    rows = rows[args.start:]
    if args.num_questions is not None and args.num_questions > -1:
        rows = rows[:args.num_questions]

    rows_before_batching = len(rows)
    batch_meta: Dict[str, Any] = {
        "enabled": False,
        "batch_size": 0,
        "batch_index": -1,
        "total_batches": 1,
        "batch_row_start": 0,
        "batch_row_end": rows_before_batching - 1,
    }

    if int(args.batch_size) > 0:
        bsz = int(args.batch_size)
        total_batches = (rows_before_batching + bsz - 1) // bsz if rows_before_batching > 0 else 0
        batch_meta["enabled"] = True
        batch_meta["batch_size"] = bsz
        batch_meta["total_batches"] = total_batches

        if int(args.batch_index) >= 0:
            bidx = int(args.batch_index)
            if total_batches == 0:
                conn.close()
                raise ValueError("--batch_index was set, but there are no rows after --start/--num_questions filtering")
            if bidx >= total_batches:
                conn.close()
                raise ValueError(f"--batch_index {bidx} out of range; total batches available: {total_batches}")

            lo = bidx * bsz
            hi = min(rows_before_batching, lo + bsz)
            rows = rows[lo:hi]
            batch_meta["batch_index"] = bidx
            batch_meta["batch_row_start"] = lo
            batch_meta["batch_row_end"] = hi - 1
        else:
            batch_meta["batch_index"] = -1
            batch_meta["batch_row_start"] = 0
            batch_meta["batch_row_end"] = rows_before_batching - 1

    if not rows:
        conn.close()
        dump_json(args.output_json, [])
        dump_json(args.checkpoint_json, {
            "completed": {},
            "meta": {
                "total_selected": 0,
                "batch_size": int(args.batch_size),
                "batch_index": int(args.batch_index),
                "total_batches": int(batch_meta.get("total_batches", 1)),
                "batch_row_start": int(batch_meta.get("batch_row_start", 0)),
                "batch_row_end": int(batch_meta.get("batch_row_end", -1)),
            },
        })
        print("\n================ BASELINE RUN SUMMARY ================")
        print(f"Output JSON:                   {args.output_json}")
        print(f"Checkpoint JSON:               {args.checkpoint_json}")
        print(f"Rows after start/num_questions:{rows_before_batching}")
        print(f"Batching enabled:              {bool(batch_meta.get('enabled'))}")
        if bool(batch_meta.get("enabled")):
            print(f"Batch size:                    {batch_meta.get('batch_size')}")
            print(f"Batch index:                   {batch_meta.get('batch_index')}")
            print(f"Total batches:                 {batch_meta.get('total_batches')}")
        print("Rows selected:                 0")
        print("Nothing to process.")
        print("======================================================\n")
        return

    few_shot_examples = None
    few_shot_text = ""
    if args.prompt_style == "few_shot" and args.few_shot_json:
        few_obj = load_json(args.few_shot_json)
        if isinstance(few_obj, list):
            few_shot_examples = few_obj
    if args.prompt_style == "few_shot" and args.few_shot_txt:
        with open(args.few_shot_txt, "r", encoding="utf-8") as f:
            few_shot_text = f.read()

    llm = None
    sampling = None
    if args.backend == "vllm":
        if LLM is None or SamplingParams is None:
            conn.close()
            raise RuntimeError("vllm is not installed/importable. Install vllm or use --backend gcp")
        if not args.model_path:
            conn.close()
            raise RuntimeError("--model_path is required for --backend vllm")

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        if litellm_completion is None:
            conn.close()
            raise RuntimeError("litellm is not installed/importable. Install with: pip install litellm")

    ckpt = load_checkpoint(args.checkpoint_json) if bool(args.resume) else {"completed": {}, "meta": {}}
    completed: Dict[str, Dict[str, Any]] = ckpt.get("completed", {}) if isinstance(ckpt, dict) else {}
    selected_ids = {str(r.get(args.item_id_key) or f"idx_{i}") for i, r in enumerate(rows)}
    completed = {k: v for k, v in completed.items() if k in selected_ids}

    total = len(rows)
    print("\n================ BASELINE RUN START ================")
    print(f"Input JSON:                    {args.input_json}")
    print(f"Output JSON:                   {args.output_json}")
    print(f"Checkpoint JSON:               {args.checkpoint_json}")
    print(f"Rows after start/num_questions:{rows_before_batching}")
    print(f"Batching enabled:              {bool(batch_meta.get('enabled'))}")
    if bool(batch_meta.get("enabled")):
        print(f"Batch size:                    {batch_meta.get('batch_size')}")
        print(f"Batch index:                   {batch_meta.get('batch_index')}")
        print(f"Total batches:                 {batch_meta.get('total_batches')}")
        print(f"Batch row range:               {batch_meta.get('batch_row_start')}..{batch_meta.get('batch_row_end')}")
    print(f"Rows selected:                 {total}")
    print(f"Backend:                       {args.backend}")
    print(f"Prompt style:                  {args.prompt_style}")
    print(f"Columns with value hints:      {len(schema_value_hints)}")
    if args.prompt_style == "few_shot":
        print(f"Few-shot TXT:                 {args.few_shot_txt or 'n/a'}")
        print(f"Few-shot JSON:                {args.few_shot_json or 'n/a'}")
    print(f"Match mode:                    {args.match_mode}")
    print(f"Cell text metric:              {args.cell_text_metric if args.match_mode == 'hungarian' else 'n/a'}")
    print(f"Rate-limit retry attempts:     {max(1, int(args.rate_limit_retry_attempts))}")
    print(f"Print per-item:                {bool(args.print_per_item)}")
    print(f"Print model check:             {bool(args.print_model_check)}")
    print(f"Resuming:                      {bool(args.resume)}")
    print(f"Already completed in ckpt:     {len(completed)}")
    print("====================================================\n")

    processed_since_save = 0
    for i, row in enumerate(rows):
        item_id = str(row.get(args.item_id_key) or f"idx_{i}")
        if item_id in completed:
            continue

        question = (row.get(args.question_key) or "").strip()
        if not question:
            question = (row.get(args.fallback_question_key) or "").strip()
        if not question:
            question = "Which clinical trials match the criteria?"

        gold_sql = (row.get(args.gold_sql_key) or "").strip()

        prompt = build_prompt(
            question=question,
            table_name=args.table_name,
            schema_cols=schema_cols,
            prompt_style=args.prompt_style,
            few_shot_examples=few_shot_examples,
            few_shot_text=few_shot_text,
            schema_value_hints=schema_value_hints,
        )

        raw_text = ""
        pred_sql = ""
        model_meta: Dict[str, Any] = {}
        model_check: Dict[str, Any] = {}
        answer_rows: List[Dict[str, Any]] = []
        answer_cols: List[str] = []
        answer_error: Optional[str] = None
        exec_eval: Optional[Dict[str, Any]] = None

        attempts = max(1, int(args.rate_limit_retry_attempts))
        for attempt in range(1, attempts + 1):
            try:
                raw_text, model_meta = generate_sql_with_backend(
                    backend=args.backend,
                    prompt=prompt,
                    llm=llm,
                    sampling=sampling,
                    gcp_model=args.gcp_model,
                    gcp_project=args.gcp_project,
                    vertex_location=args.vertex_location,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout=args.litellm_timeout,
                    num_retries=args.litellm_num_retries,
                )
                model_meta["call_attempt"] = attempt
                model_meta["call_attempts_configured"] = attempts
                pred_sql = extract_sql(raw_text)
                break
            except Exception as e:
                retryable = is_retryable_provider_error(e)
                if attempt >= attempts or not retryable:
                    model_meta = {
                        "backend": args.backend,
                        "provider_call_ok": False,
                        "model": args.gcp_model if args.backend == "gcp" else args.model_path,
                        "finish_reason": "error",
                        "stop_reason": None,
                        "token_ids_len": None,
                        "error": str(e),
                        "retryable_error": retryable,
                        "call_attempt": attempt,
                        "call_attempts_configured": attempts,
                    }
                    break

                sleep_s = min(
                    max(0.0, float(args.retry_backoff_seconds))
                    * (max(1.0, float(args.retry_backoff_multiplier)) ** (attempt - 1)),
                    max(0.0, float(args.retry_max_backoff_seconds)),
                )
                if bool(args.print_model_check):
                    one_line_error = str(e).splitlines()[0] if str(e) else "unknown error"
                    print(
                        f"[retry {attempt}/{attempts}] item_id={item_id} "
                        f"sleep={sleep_s:.1f}s err={one_line_error}"
                    )
                time.sleep(sleep_s)

        raw_nonempty = bool((raw_text or "").strip())
        pred_nonempty = bool((pred_sql or "").strip())
        raw_has_select = bool(re.search(r"\bSELECT\b", raw_text or "", flags=re.IGNORECASE))
        model_check = {
            "backend_requested": args.backend,
            "model_requested": args.gcp_model if args.backend == "gcp" else args.model_path,
            "provider_call_ok": bool(model_meta.get("provider_call_ok", False)),
            "raw_output_nonempty": raw_nonempty,
            "raw_contains_select": raw_has_select,
            "pred_sql_nonempty": pred_nonempty,
            "pred_sql_extracted": pred_nonempty,
            "call_attempt": model_meta.get("call_attempt"),
            "call_attempts_configured": model_meta.get("call_attempts_configured"),
            "retryable_error": model_meta.get("retryable_error"),
        }
        if not raw_nonempty:
            model_check["issue"] = "EMPTY_MODEL_OUTPUT"
        elif raw_nonempty and not pred_nonempty:
            model_check["issue"] = "NO_SELECT_EXTRACTED_FROM_MODEL_OUTPUT"
        else:
            model_check["issue"] = None

        if pred_sql:
            try:
                cols, rows_sql = execute_sql_fetch(conn, pred_sql, max_rows=args.answer_max_rows)
                answer_cols = cols
                answer_rows = rows_to_dicts(cols, rows_sql)
            except Exception as e:
                answer_error = f"EXEC_ERROR: {e}"
        else:
            answer_error = "EMPTY_PRED_SQL"

        if pred_sql and gold_sql:
            exec_eval = evaluate_pair(
                conn=conn,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                max_rows=args.eval_max_rows,
                match_mode=args.match_mode,
                hungarian_max_rows=args.hungarian_max_rows,
                cell_text_metric=args.cell_text_metric,
            )
        else:
            exec_eval = {
                "eval_ok": False,
                "error": "MISSING_PRED_OR_GOLD_SQL",
                "match_mode_requested": args.match_mode,
                "cell_text_metric_requested": args.cell_text_metric,
            }

        result_payload: Dict[str, Any] = {
            "baseline_question_used": question,
            "baseline_pred_sql": pred_sql,
            "baseline_model_raw_sql_output": raw_text,
            "baseline_model_meta": model_meta,
            "baseline_model_check": model_check,
            "baseline_backend": args.backend,
            "baseline_prompt_style": args.prompt_style,
            "baseline_answer_columns": answer_cols,
            "baseline_answer_rows": answer_rows,
            "baseline_answer_row_count": len(answer_rows),
            "baseline_answer_error": answer_error,
            "baseline_exec_eval": exec_eval,
        }
        if bool(args.save_prompts):
            result_payload["baseline_prompt"] = prompt

        completed[item_id] = result_payload
        processed_since_save += 1

        if bool(args.print_per_item):
            ee = exec_eval if isinstance(exec_eval, dict) else {}
            p = float(ee.get("precision", 0.0) or 0.0)
            r = float(ee.get("recall", 0.0) or 0.0)
            f1 = float(ee.get("f1", 0.0) or 0.0)
            j = float(ee.get("row_jaccard", 0.0) or 0.0)
            s = float(ee.get("soft_overlap_score", 0.0) or 0.0)

            print(f"[{i + 1}/{total}] item_id={item_id}")
            if bool(args.print_model_check):
                print(
                    "ModelCheck: "
                    f"call_ok={bool(model_check.get('provider_call_ok'))}, "
                    f"attempt={model_check.get('call_attempt')}/{model_check.get('call_attempts_configured')}, "
                    f"raw_nonempty={bool(model_check.get('raw_output_nonempty'))}, "
                    f"raw_has_select={bool(model_check.get('raw_contains_select'))}, "
                    f"pred_sql_nonempty={bool(model_check.get('pred_sql_nonempty'))}, "
                    f"issue={model_check.get('issue')}"
                )
                if model_meta.get("error"):
                    print(f"Model error: {model_meta.get('error')}")

            print(
                "Eval: "
                f"ok={bool(ee.get('eval_ok'))}, "
                f"precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, "
                f"row_jaccard={j:.4f}, soft_overlap={s:.4f}, "
                f"relaxed_em={bool(ee.get('relaxed_em_unordered'))}"
            )
            if ee.get("error"):
                print(f"Eval error: {ee.get('error')}")
            print("----------------------------------------------------")

        if processed_since_save >= max(1, int(args.save_every)):
            out_rows = compose_output_rows(rows, completed, args.item_id_key)
            dump_json(args.output_json, out_rows)
            dump_json(args.checkpoint_json, {
                "completed": completed,
                "meta": {
                    "backend": args.backend,
                    "prompt_style": args.prompt_style,
                    "match_mode": args.match_mode,
                    "cell_text_metric": args.cell_text_metric,
                    "total_selected": total,
                    "batch_size": int(args.batch_size),
                    "batch_index": int(args.batch_index),
                    "total_batches": int(batch_meta.get("total_batches", 1)),
                    "batch_row_start": int(batch_meta.get("batch_row_start", 0)),
                    "batch_row_end": int(batch_meta.get("batch_row_end", total - 1)),
                },
            })
            processed_since_save = 0

    out_rows = compose_output_rows(rows, completed, args.item_id_key)
    dump_json(args.output_json, out_rows)
    dump_json(args.checkpoint_json, {
        "completed": completed,
        "meta": {
            "backend": args.backend,
            "prompt_style": args.prompt_style,
            "match_mode": args.match_mode,
            "cell_text_metric": args.cell_text_metric,
            "total_selected": total,
            "batch_size": int(args.batch_size),
            "batch_index": int(args.batch_index),
            "total_batches": int(batch_meta.get("total_batches", 1)),
            "batch_row_start": int(batch_meta.get("batch_row_start", 0)),
            "batch_row_end": int(batch_meta.get("batch_row_end", total - 1)),
        },
    })

    conn.close()

    vals = list(completed.values())
    non_empty_sql = sum(1 for v in vals if (v.get("baseline_pred_sql") or "").strip())
    answer_ok = sum(1 for v in vals if not v.get("baseline_answer_error"))
    answer_err = sum(1 for v in vals if v.get("baseline_answer_error"))

    eval_ok_vals = [v for v in vals if isinstance(v.get("baseline_exec_eval"), dict) and v["baseline_exec_eval"].get("eval_ok")]
    eval_ok = len(eval_ok_vals)
    eval_err = len(vals) - eval_ok

    def _avg(key: str) -> float:
        arr = [float(v["baseline_exec_eval"].get(key, 0.0)) for v in eval_ok_vals]
        return mean(arr) if arr else 0.0

    rem_hits = sum(1 for v in eval_ok_vals if bool(v["baseline_exec_eval"].get("relaxed_em_unordered")))

    print("\n================ BASELINE RUN SUMMARY ================")
    print(f"Output JSON:                   {args.output_json}")
    print(f"Checkpoint JSON:               {args.checkpoint_json}")
    print(f"Rows selected:                 {total}")
    print(f"Rows completed:                {len(vals)}")
    print(f"Non-empty predicted SQL:       {non_empty_sql}")
    print(f"Answer execution OK:           {answer_ok}")
    print(f"Answer execution errors:       {answer_err}")
    print(f"Eval OK:                       {eval_ok}")
    print(f"Eval errors:                   {eval_err}")
    if eval_ok:
        print(f"Avg precision:                 {_avg('precision'):.4f}")
        print(f"Avg recall:                    {_avg('recall'):.4f}")
        print(f"Avg F1:                        {_avg('f1'):.4f}")
        print(f"Avg row_jaccard:               {_avg('row_jaccard'):.4f}")
        print(f"Avg soft_overlap_score:        {_avg('soft_overlap_score'):.4f}")
        print(f"Relaxed EM hits:               {rem_hits}/{eval_ok}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
