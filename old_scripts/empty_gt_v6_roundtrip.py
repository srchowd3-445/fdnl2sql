#!/usr/bin/env python3
"""
empty_gt_v6_roundtrip.py

End-to-end "question rewrite -> SQL generation -> evaluation" pipeline.

Input:
- A JSON file that contains a LIST of records (not JSONL).
  Expected fields (common in your data):
    - question (original question)
    - empty_gt_sql (original SQL)
    - new_gt_sql (new "ground truth" SQL that returns results)

Process:
1) Schema sanity check (prints table columns, rowcount, sample row preview).
2) For each record with new_gt_sql:
   a) Execute new_gt_sql and store preview rows
   b) Use Gemma (vLLM) to rewrite question so it matches new_gt_sql
   c) Use Gemma (vLLM) to generate SQL from rewritten question using schema prompt
   d) Evaluate pred_sql vs new_gt_sql:
      - normalized_exact_match
      - ast_match (sqlglot)
      - execution_match_strict (cols + rows)
      - execution_match_loose (common cols only)

Output:
- Writes a JSON LIST to --output_json (default: empty_gt_fixed_v6.json)
  Each record includes:
    - original_question, original_sql, new_sql, rewritten_question
    - new_sql_results_preview
    - pred_sql, model_raw_sql_output
    - metrics + errors + vllm metadata
    - filter diffs (added/removed WHERE clauses)

Notes:
- This script is NOT append-safe (writes full JSON at end). If it crashes, rerun.
  You can use --limit/--start for chunking.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

# pip install sqlglot
import sqlglot


# -------------------------
# Helpers: SQL normalization / schema / execution
# -------------------------
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    s = strip_trailing_semicolon(sql)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*=\s*", "=", s)
    s = re.sub(r"\s*>=\s*", ">=", s)
    s = re.sub(r"\s*<=\s*", "<=", s)
    s = re.sub(r"\s*>\s*", ">", s)
    s = re.sub(r"\s*<\s*", "<", s)
    return s


def fetch_schema(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}");')
    rows = cur.fetchall()
    return [r[1] for r in rows]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def get_table_rowcount(conn: sqlite3.Connection, table: str) -> Optional[int]:
    try:
        cur = conn.execute(f'SELECT COUNT(*) FROM "{table}";')
        return int(cur.fetchone()[0])
    except Exception:
        return None


def fetch_sample_rows(conn: sqlite3.Connection, table: str, limit: int = 3) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    try:
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)};')
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return cols, rows
    except Exception:
        return [], []


def print_schema_sanity(conn: sqlite3.Connection, db_path: str, table: str, schema_cols: List[str], sample_rows: int = 3, show_cols: int = 40) -> None:
    print("\n================ SCHEMA SANITY CHECK ================")
    print(f"DB:    {db_path}")
    print(f"Table: {table}")
    print(f"Table exists: {table_exists(conn, table)}")
    print(f"Num columns: {len(schema_cols)}")
    if schema_cols:
        head = schema_cols[: min(show_cols, len(schema_cols))]
        print(f"First {len(head)} columns:")
        for c in head:
            print(f"  - {c}")
        if len(schema_cols) > show_cols:
            print(f"... ({len(schema_cols) - show_cols} more columns not shown)")
    rc = get_table_rowcount(conn, table)
    if rc is not None:
        print(f"Row count: {rc}")
    cols, rows = fetch_sample_rows(conn, table, limit=sample_rows)
    if cols and rows:
        max_show = min(10, len(cols))
        print(f"Sample rows: {len(rows)} (showing {min(sample_rows, len(rows))})")
        print(f"Sample row keys (first {max_show} cols): {cols[:max_show]}")
        r0 = rows[0]
        print("Sample row[0] preview:")
        for i in range(max_show):
            v = r0[i]
            s = str(v)
            if len(s) > 120:
                s = s[:120] + "..."
            print(f"  {cols[i]} = {s}")
    else:
        print("Could not fetch sample rows (table empty or query failed).")
    print("=====================================================\n")


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int = 200) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


def execute_sql_preview(conn: sqlite3.Connection, sql: str, max_rows: int = 3) -> List[Dict[str, Any]]:
    cols, rows = execute_sql_fetch(conn, sql, max_rows=max_rows)
    return [dict(zip(cols, r)) for r in rows]


def canonicalize_result(cols: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d: Dict[str, Any] = {}
        for c, v in zip(cols, r):
            if isinstance(v, (int, float, str)) or v is None:
                d[c] = v
            else:
                d[c] = str(v)
        out.append(d)
    return out


def results_match_strict(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    if cols_a != cols_b:
        return False, "COL_MISMATCH"

    can_a = canonicalize_result(cols_a, rows_a)
    can_b = canonicalize_result(cols_b, rows_b)
    return can_a == can_b, None


def results_match_loose(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    common = [c for c in cols_a if c in cols_b]
    if not common:
        return False, "NO_COMMON_COLS"

    idx_a = [cols_a.index(c) for c in common]
    idx_b = [cols_b.index(c) for c in common]

    proj_a = [tuple(r[i] for i in idx_a) for r in rows_a]
    proj_b = [tuple(r[i] for i in idx_b) for r in rows_b]
    return proj_a == proj_b, None


# -------------------------
# AST eval
# -------------------------
def canonicalize_sql_ast(sql: str, dialect: str = "sqlite") -> Optional[dict]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    return tree.dump()


def ast_match_sql(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        a = canonicalize_sql_ast(sql_a, dialect=dialect)
        b = canonicalize_sql_ast(sql_b, dialect=dialect)
        if a is None or b is None:
            return False, "EMPTY_SQL"
        return a == b, None
    except Exception as e:
        return False, f"AST_ERROR: {e}"


# -------------------------
# WHERE diff (for debugging)
# -------------------------
def extract_where_conditions(sql: str) -> List[str]:
    if not sql:
        return []
    sql0 = strip_trailing_semicolon(sql)
    m = re.search(r"\bWHERE\b", sql0, flags=re.IGNORECASE)
    if not m:
        return []
    after = sql0[m.end():].strip()
    m2 = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b", after, flags=re.IGNORECASE)
    if m2:
        after = after[:m2.start()].strip()
    parts = [p.strip() for p in re.split(r"\s+AND\s+", after, flags=re.IGNORECASE) if p.strip()]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return parts


def diff_filters(original_sql: str, new_sql: str) -> Dict[str, List[str]]:
    orig = set(extract_where_conditions(original_sql))
    new = set(extract_where_conditions(new_sql))
    return {
        "filters_added": sorted(list(new - orig)),
        "filters_removed": sorted(list(orig - new)),
    }


# -------------------------
# Prompt builders (keep structure: rules + context + question/schema)
# -------------------------
def build_rewrite_prompt(original_question: str, new_sql: str) -> str:
    return f"""You are helping rewrite a natural-language question so it matches a SQL query that actually returns results.

Rules:
- Keep it concise and natural.
- Do NOT mention SQL, databases, tables, or column names explicitly.
- Keep the clinical-trials context.
- Preserve the intent style (asking for trials), but update constraints to match the new SQL exactly.
- If the new SQL changes numeric thresholds or categories, reflect those in the new question.

Original question:
{original_question}

New SQL (ground truth that should be matched):
{new_sql}

Return ONLY the rewritten question as a single sentence (no quotes, no bullet points).
"""


def build_sqlgen_prompt(schema_cols: List[str], question: str, table: str) -> str:
    schema_text = "\n".join([f'- "{c}"' for c in schema_cols])
    return f"""You are a SQL generator. Write a single SQLite SELECT query for the table "{table}".

Rules:
- Output ONLY one SQLite SELECT statement. No explanations. No markdown. No backticks.
- The query MUST start with: SELECT
- Use ONLY these column names (case and spelling must match exactly). Do NOT invent columns.
- Use double quotes for column names exactly as listed.
- Use single quotes for string literals.
- Use the table name exactly: {table}


Schema (columns):
{schema_text}

Question:
{question}
"""


def extract_first_line(text: str) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0].strip()


def extract_sql(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```sql\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


# -------------------------
# ID
# -------------------------
def compute_item_id(record: Dict[str, Any]) -> str:
    if "line_number" in record and record["line_number"] is not None:
        return f"line_{record['line_number']}"
    q = (record.get("question") or "").strip()
    s = (record.get("new_gt_sql") or "").strip()
    h = hashlib.sha1((q + "\n" + s).encode("utf-8")).hexdigest()[:12]
    return f"hash_{h}"


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to JSON list input")
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--output_json", default="empty_gt_fixed_v6.json")

    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
        help="Local HF snapshot directory for Gemma-3-27b-it",
    )

    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all")

    # rewrite generation params
    ap.add_argument("--rewrite_max_tokens", type=int, default=8192)
    ap.add_argument("--rewrite_temperature", type=float, default=0.2)
    ap.add_argument("--rewrite_top_p", type=float, default=0.9)

    # sql generation params
    ap.add_argument("--sql_max_tokens", type=int, default=8192)
    ap.add_argument("--sql_temperature", type=float, default=0.0)
    ap.add_argument("--sql_top_p", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_rows_compare", type=int, default=200)
    ap.add_argument("--preview_rows", type=int, default=3)

    ap.add_argument("--schema_sanity_only", action="store_true")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # DB + schema sanity
    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    print_schema_sanity(conn, args.db_path, args.table_name, schema_cols, sample_rows=3, show_cols=40)
    if args.schema_sanity_only:
        conn.close()
        return
    if not schema_cols:
        conn.close()
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")

    # Load input JSON list
    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        conn.close()
        raise ValueError("Input must be a JSON LIST of records.")

    # Filter + slice
    eligible = [r for r in records if r.get("new_gt_sql")]
    if args.limit is not None and args.limit > -1:
        eligible = eligible[args.start : args.start + args.limit]
    else:
        eligible = eligible[args.start :]

    if not eligible:
        conn.close()
        print("No eligible items (missing new_gt_sql).")
        return

    # Prepare LLM (single instance used for both stages)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    rewrite_sampling = SamplingParams(
        max_tokens=args.rewrite_max_tokens,
        temperature=args.rewrite_temperature,
        top_p=args.rewrite_top_p,
    )

    sql_sampling = SamplingParams(
        max_tokens=args.sql_max_tokens,
        temperature=args.sql_temperature,
        top_p=args.sql_top_p,
    )

    # Build rewrite prompts
    rewrite_prompts: List[str] = []
    items: List[Dict[str, Any]] = []
    for r in eligible:
        item_id = compute_item_id(r)
        original_question = r.get("question", "")
        original_sql = r.get("empty_gt_sql", "") or r.get("original_sql", "")
        new_sql = r.get("new_gt_sql", "")
        rewrite_prompts.append(build_rewrite_prompt(original_question, new_sql))
        items.append({
            "item_id": item_id,
            "original_question": original_question,
            "original_sql": original_sql,
            "new_sql": new_sql,
            "_raw_record": r,
        })

    # Stage 1: rewrite questions (batched)
    rewritten_questions: List[str] = [""] * len(items)
    rewrite_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    for b0 in range(0, len(items), args.batch_size):
        batch_prompts = rewrite_prompts[b0 : b0 + args.batch_size]
        outs = llm.generate(batch_prompts, rewrite_sampling)
        for i, out in enumerate(outs):
            idx = b0 + i
            gen_text = (out.outputs[0].text or "").strip() if out.outputs else ""
            rewritten_questions[idx] = extract_first_line(gen_text)
            try:
                rewrite_meta[idx] = {
                    "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                    "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                    "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                } if out.outputs else None
            except Exception:
                rewrite_meta[idx] = None

    # Stage 2: build SQL-gen prompts from rewritten question
    sql_prompts: List[str] = []
    for i, it in enumerate(items):
        q2 = rewritten_questions[i] or it["original_question"]
        sql_prompts.append(build_sqlgen_prompt(schema_cols, q2, args.table_name))

    # Stage 2: generate SQL (batched)
    pred_sqls: List[str] = [""] * len(items)
    raw_sql_outputs: List[str] = [""] * len(items)
    sql_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    for b0 in range(0, len(items), args.batch_size):
        batch_prompts = sql_prompts[b0 : b0 + args.batch_size]
        outs = llm.generate(batch_prompts, sql_sampling)
        for i, out in enumerate(outs):
            idx = b0 + i
            gen_text = (out.outputs[0].text or "").strip() if out.outputs else ""
            raw_sql_outputs[idx] = gen_text
            pred_sqls[idx] = extract_sql(gen_text)
            try:
                sql_meta[idx] = {
                    "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                    "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                    "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                } if out.outputs else None
            except Exception:
                sql_meta[idx] = None

    # Evaluate + assemble output records
    out_rows: List[Dict[str, Any]] = []

    total = 0
    exact_yes = 0
    ast_yes = 0
    exec_strict_yes = 0
    exec_loose_yes = 0
    empty_pred_cnt = 0
    exec_err_cnt = 0
    ast_err_cnt = 0

    for i, it in enumerate(items):
        total += 1
        new_sql = it["new_sql"]
        original_sql = it["original_sql"]
        rewritten_q = rewritten_questions[i]
        pred_sql = pred_sqls[i]

        # Preview new_sql results
        try:
            preview = execute_sql_preview(conn, new_sql, max_rows=args.preview_rows)
        except Exception as e:
            preview = [{"SQL_ERROR": str(e)}]

        # Filter diffs
        filter_diff = diff_filters(original_sql, new_sql)

        # Metrics
        exact = False
        ast_ok = False
        ast_err = None

        exec_ok_strict = False
        exec_err_strict = None
        exec_ok_loose = False
        exec_err_loose = None

        if not pred_sql:
            empty_pred_cnt += 1

        if pred_sql and new_sql:
            exact = normalize_sql(pred_sql) == normalize_sql(new_sql)

            ast_ok, ast_err = ast_match_sql(pred_sql, new_sql, dialect="sqlite")
            if ast_err:
                ast_err_cnt += 1

            exec_ok_strict, exec_err_strict = results_match_strict(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)
            if exec_err_strict and exec_err_strict.startswith("EXEC_ERROR"):
                exec_err_cnt += 1

            exec_ok_loose, exec_err_loose = results_match_loose(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)

        if exact:
            exact_yes += 1
        if ast_ok:
            ast_yes += 1
        if exec_ok_strict:
            exec_strict_yes += 1
        if exec_ok_loose:
            exec_loose_yes += 1

        out_rows.append({
            "item_id": it["item_id"],
            "original_question": it["original_question"],
            "original_sql": original_sql,
            "new_sql": new_sql,
            "new_sql_results_preview": preview,
            "filters_added": filter_diff["filters_added"],
            "filters_removed": filter_diff["filters_removed"],

            "rewritten_question": rewritten_q,
            "rewrite_vllm_meta": rewrite_meta[i],

            "pred_sql": pred_sql,
            "model_raw_sql_output": raw_sql_outputs[i],
            "sqlgen_vllm_meta": sql_meta[i],

            "normalized_exact_match": bool(exact),
            "ast_match": bool(ast_ok),
            "ast_error": ast_err,

            "execution_match_strict": bool(exec_ok_strict),
            "execution_error_strict": exec_err_strict,
            "execution_match_loose": bool(exec_ok_loose),
            "execution_error_loose": exec_err_loose,
        })

    conn.close()

    # Write output JSON list
    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n================ FINAL SUMMARY ================")
    print(f"Output JSON: {out_path}")
    print(f"Processed: {total}")
    print(f"normalized_exact_match:   {exact_yes}/{total} ({pct(exact_yes, total):.2f}%)")
    print(f"ast_match:                {ast_yes}/{total} ({pct(ast_yes, total):.2f}%)")
    print(f"execution_match_strict:   {exec_strict_yes}/{total} ({pct(exec_strict_yes, total):.2f}%)")
    print(f"execution_match_loose:    {exec_loose_yes}/{total} ({pct(exec_loose_yes, total):.2f}%)")
    print(f"Empty pred_sql:           {empty_pred_cnt}")
    print(f"AST errors:               {ast_err_cnt}")
    print(f"Execution errors (real):  {exec_err_cnt}")
    print("==============================================\n")


if __name__ == "__main__":
    main()