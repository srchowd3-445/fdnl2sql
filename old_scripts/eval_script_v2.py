#!/usr/bin/env python3
"""
sql_roundtrip_eval_vllm.py

Goal:
- Take natural-language questions from new_gt_questions_vllm.jsonl (or JSON)
- Ask Gemma-3-27b-it (via vLLM) to generate SQL for the clinical_trials table
- Evaluate whether it matches the existing "new_sql" in the input

Evaluation:
- normalized_exact_match: string match after aggressive normalization
- ast_match: compare canonicalized SQL ASTs (sqlglot, SQLite dialect)
- execution_match_strict: compare query outputs (columns + rows) exactly
- execution_match_loose: compare query outputs on intersection of columns (ignores column mismatch)
- also record parse/SQL errors and vLLM metadata (finish_reason, stop_reason, token count)

Writes JSONL output per record (append-safe, resumable):
  item_id, question_used, target_sql, pred_sql,
  normalized_exact_match, ast_match,
  execution_match_strict, execution_match_loose,
  errors, vllm_meta, etc.

Additionally prints a final aggregate summary at the end.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

# AST parsing / comparison
# pip install sqlglot
import sqlglot
from sqlglot import exp


# -------------------------
# SQL + result helpers
# -------------------------
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    """
    Normalization for exact match comparison:
    - strip trailing semicolon
    - collapse whitespace
    - remove spaces around commas/operators
    """
    s = strip_trailing_semicolon(sql)
    s = re.sub(r"\s+", " ", s).strip()
    # normalize spacing around punctuation/operators
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


def execute_sql_fetch(
    conn: sqlite3.Connection, sql: str, max_rows: int = 200
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """
    Execute SQL and fetch up to max_rows.
    Returns (columns, rows). Raises on SQL errors.
    """
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


def canonicalize_result(cols: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    """
    Canonicalize rows to list of dicts with stable key order.
    Convert all values to JSON-safe (str for weird types).
    """
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


def results_match_strict(
    conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200
) -> Tuple[bool, Optional[str]]:
    """
    Execute and compare outputs (columns + rows) up to max_rows.
    Strict: columns must match exactly (including order).
    Returns (match, error_msg_if_any).
    """
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


def results_match_loose(
    conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200
) -> Tuple[bool, Optional[str]]:
    """
    Execute and compare outputs on the intersection of columns.
    Loose: ignores extra/missing projected columns, compares only common columns (in cols_a order).
    Returns (match, error_msg_if_any).
    """
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
# AST helpers (sqlglot)
# -------------------------
def canonicalize_sql_ast(sql: str, dialect: str = "sqlite") -> Optional[dict]:
    """
    Parse SQL into an AST and return a canonical JSON-like dict representation.
    Structural equivalence only.
    """
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
# Prompting
# -------------------------
def build_prompt(schema_cols: List[str], question: str, table: str) -> str:
    """
    Same high-level structure (Rules, Schema, Question) but with stronger constraints
    and mapping hints to reduce wrong-column / wrong-field errors.
    """
    schema_text = "\n".join([f'- "{c}"' for c in schema_cols])
    return f"""You are a SQL generator. Write a single SQLite SELECT query for the table "{table}".

Rules:
- Output ONLY one SQLite SELECT statement. No explanations. No markdown. No backticks.
- The query MUST start with: SELECT
- Use ONLY these column names (case and spelling must match exactly). Do NOT invent columns.
- Use double quotes for column names exactly as listed.
- Use single quotes for string literals.
- Use the table name exactly: {table}
- Do NOT use JOINs (only one table exists).
- Prefer exact-match filters using '=' for categorical columns unless the question clearly requires partial match.
- If you use LIKE, use it only on text fields and include '%' wildcards explicitly.
- IMPORTANT mapping hints:
  - If the question mentions CTLA-4 / PD-1 / PD-L1 / LAG-3 etc., treat it as "Class of ICI" (not "Name of ICI") unless a specific drug name is given.
  - If the question mentions "combination therapy", use the "Monotherapy/combination" column (match the dataset's capitalization exactly).
- Before writing the final query, silently check that every selected/filtered column appears in the schema list below.

Schema (columns):
{schema_text}

Question:
{question}
"""


def extract_sql(text: str) -> str:
    """
    Extract SQL from model output. We instruct "only SQL", but be defensive.
    """
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```sql\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # If multiple statements, keep the first up to the first semicolon
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


# -------------------------
# JSON / JSONL IO + resume
# -------------------------
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if not raw:
            return []

        # JSON array or JSON object
        if raw[0] in "[{":
            obj = json.loads(raw)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
            raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

    # fallback: JSONL
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_item_id(rec: Dict[str, Any]) -> str:
    """
    Fix: avoid identical hashes by using broader fallback fields and stable JSON blob.
    """
    if rec.get("item_id"):
        return str(rec["item_id"])

    q = (rec.get("new_question") or rec.get("original_question") or rec.get("question") or "").strip()
    s = (
        rec.get("new_sql")
        or rec.get("new_gt_sql")
        or rec.get("original_sql")
        or rec.get("empty_gt_sql")
        or ""
    ).strip()

    blob = json.dumps({"q": q, "s": s}, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"hash_{h}"


def load_processed_ids(output_jsonl: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    done.add(iid)
            except Exception:
                continue
    return done


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--output_jsonl", required=True)

    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
        help="Local HF snapshot dir for Gemma-3-27b-it",
    )

    # selection
    ap.add_argument("--use_field", choices=["new_question", "original_question"], default="new_question")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)

    # evaluation
    ap.add_argument("--max_rows_compare", type=int, default=200)
    ap.add_argument("--require_target_sql", action="store_true")

    # AST controls
    ap.add_argument("--ast_dialect", default="sqlite")

    # resumability
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush_every", type=int, default=1)

    # reporting controls
    ap.add_argument("--print_fail_examples", type=int, default=0,
                    help="Print up to N example failures at the end (0 disables).")

    # vLLM controls
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    # generation params
    # NOTE: default reduced from 8192 to 512 as a more typical SQL generation cap.
    # You can still override on the CLI.
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    records = load_json_or_jsonl(args.input_jsonl)

    if args.limit and args.limit > 0:
        records = records[args.start : args.start + args.limit]
    else:
        records = records[args.start :]

    if args.require_target_sql:
        records = [r for r in records if (r.get("new_sql") or r.get("new_gt_sql") or r.get("original_sql") or r.get("empty_gt_sql"))]

    if not records:
        print("No records to process.")
        return

    processed = set()
    if args.resume:
        processed = load_processed_ids(args.output_jsonl)
        print(f"[resume] found {len(processed)} processed items in output.")

    todo = []
    for r in records:
        iid = compute_item_id(r)
        if args.resume and iid in processed:
            continue
        r["_item_id"] = iid
        todo.append(r)

    if not todo:
        print("Nothing to do (all items already processed).")
        return

    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    if not schema_cols:
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    # --- aggregate counters ---
    total = 0
    exact_yes = 0
    ast_yes = 0
    exec_strict_yes = 0
    exec_loose_yes = 0
    ast_err_cnt = 0
    exec_err_cnt = 0
    empty_target_cnt = 0
    empty_pred_cnt = 0
    vllm_empty_token_cnt = 0

    # store a few failures for printing
    fail_examples: List[Dict[str, Any]] = []

    written = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0 : b0 + args.batch_size]

            prompts = []
            questions_used: List[str] = []
            for r in batch:
                q = (r.get(args.use_field) or "").strip()
                if not q:
                    q = (r.get("original_question") or r.get("question") or "").strip()
                questions_used.append(q)
                prompts.append(build_prompt(schema_cols, q, args.table_name))

            outputs = llm.generate(prompts, sampling)

            for r, q, out in zip(batch, questions_used, outputs):
                iid = r["_item_id"]

                target_sql = (r.get("new_sql") or r.get("new_gt_sql") or "").strip()
                if not target_sql:
                    target_sql = (r.get("original_sql") or r.get("empty_gt_sql") or "").strip()

                gen_text = ""
                vllm_meta = None
                if out.outputs:
                    gen_text = (out.outputs[0].text or "").strip()
                    # DEBUG: inspect vLLM output structure
                    try:
                        vllm_meta = {
                            "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                            "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                            "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                        }
                        if vllm_meta["token_ids_len"] == 0:
                            vllm_empty_token_cnt += 1
                    except Exception:
                        vllm_meta = None

                pred_sql = extract_sql(gen_text)

                if not target_sql:
                    empty_target_cnt += 1
                if not pred_sql:
                    empty_pred_cnt += 1

                exact = False
                if pred_sql and target_sql:
                    exact = normalize_sql(pred_sql) == normalize_sql(target_sql)

                ast_ok = False
                ast_err = None
                if pred_sql and target_sql:
                    ast_ok, ast_err = ast_match_sql(pred_sql, target_sql, dialect=args.ast_dialect)
                    if ast_err:
                        ast_err_cnt += 1

                exec_match_strict = False
                exec_match_loose = False
                exec_err_strict = None
                exec_err_loose = None
                if pred_sql and target_sql:
                    exec_match_strict, exec_err_strict = results_match_strict(
                        conn, pred_sql, target_sql, max_rows=args.max_rows_compare
                    )
                    # Count only "real" execution errors, not column mismatch
                    if exec_err_strict and exec_err_strict.startswith("EXEC_ERROR"):
                        exec_err_cnt += 1

                    exec_match_loose, exec_err_loose = results_match_loose(
                        conn, pred_sql, target_sql, max_rows=args.max_rows_compare
                    )

                row = {
                    "item_id": iid,
                    "question_used": q,
                    "target_sql": target_sql,
                    "pred_sql": pred_sql,
                    "normalized_exact_match": bool(exact),
                    "ast_match": bool(ast_ok),
                    "ast_error": ast_err,
                    "execution_match_strict": bool(exec_match_strict),
                    "execution_error_strict": exec_err_strict,
                    "execution_match_loose": bool(exec_match_loose),
                    "execution_error_loose": exec_err_loose,
                    "model_raw_output": gen_text,
                    "vllm_meta": vllm_meta,
                }

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                if written % args.flush_every == 0:
                    out_f.flush()
                    os.fsync(out_f.fileno())

                # update aggregates
                total += 1
                if exact:
                    exact_yes += 1
                if ast_ok:
                    ast_yes += 1
                if exec_match_strict:
                    exec_strict_yes += 1
                if exec_match_loose:
                    exec_loose_yes += 1

                # collect failure examples (optional)
                if args.print_fail_examples > 0:
                    if (pred_sql and target_sql) and (not exec_match_loose):
                        if len(fail_examples) < args.print_fail_examples:
                            fail_examples.append({
                                "item_id": iid,
                                "question": q,
                                "target_sql": target_sql,
                                "pred_sql": pred_sql,
                                "ast_match": ast_ok,
                                "ast_error": ast_err,
                                "execution_error_strict": exec_err_strict,
                                "execution_error_loose": exec_err_loose,
                                "vllm_meta": vllm_meta,
                            })

            if written % 50 == 0:
                print(f"Processed {written}/{len(todo)}")

    conn.close()

    # ----- final summary -----
    print("\n================ FINAL SUMMARY ================")
    print(f"Output file: {args.output_jsonl}")
    print(f"Processed items: {total}")

    print(f"normalized_exact_match:   {exact_yes}/{total} ({pct(exact_yes, total):.2f}%)")
    print(f"ast_match:                {ast_yes}/{total} ({pct(ast_yes, total):.2f}%)")
    print(f"execution_match_strict:   {exec_strict_yes}/{total} ({pct(exec_strict_yes, total):.2f}%)")
    print(f"execution_match_loose:    {exec_loose_yes}/{total} ({pct(exec_loose_yes, total):.2f}%)")

    print(f"AST errors:               {ast_err_cnt}")
    print(f"Execution errors (real):  {exec_err_cnt}")
    print(f"Empty target_sql:         {empty_target_cnt}")
    print(f"Empty pred_sql:           {empty_pred_cnt}")
    print(f"vLLM zero-token outputs:  {vllm_empty_token_cnt}")
    print("==============================================\n")

    if args.print_fail_examples and fail_examples:
        print(f"------ Example failures (up to {args.print_fail_examples}) ------")
        for i, ex in enumerate(fail_examples, 1):
            print(f"\n[{i}] item_id={ex['item_id']}")
            print(f"Q: {ex['question']}")
            print(f"TARGET: {ex['target_sql']}")
            print(f"PRED:   {ex['pred_sql']}")
            if ex.get("execution_error_strict"):
                print(f"execution_error_strict: {ex['execution_error_strict']}")
            if ex.get("execution_error_loose"):
                print(f"execution_error_loose: {ex['execution_error_loose']}")
            if ex.get("ast_error"):
                print(f"ast_error: {ex['ast_error']}")
            print(f"ast_match: {ex['ast_match']}")
            if ex.get("vllm_meta") is not None:
                print(f"vllm_meta: {ex['vllm_meta']}")
        print("------------------------------------------------\n")

    print(f"Done. Appended {written} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()