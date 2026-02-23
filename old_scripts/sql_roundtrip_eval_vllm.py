#!/usr/bin/env python3
"""
sql_roundtrip_eval_vllm.py

Goal:
- Take natural-language questions from new_gt_questions_vllm.jsonl
- Ask Gemma-3-27b-it (via vLLM) to generate SQL for the clinical_trials table
- Evaluate whether it matches the existing "new_sql" in the JSONL

Evaluation:
- normalized_exact_match: string match after aggressive normalization
- execution_match: compare query outputs (as canonicalized JSON) with optional LIMIT
- optional: also record parse/SQL errors

Input JSONL expects fields like:
  item_id, new_question, new_sql
(or fallback to original_question/original_sql)

Writes JSONL output per record (append-safe, resumable):
  item_id, question_used, target_sql, pred_sql,
  normalized_exact_match, execution_match, errors, etc.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams


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
    - uppercase keywords lightly (optional-ish)
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
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    return [r[1] for r in rows]

def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int = 200) -> Tuple[List[str], List[Tuple[Any, ...]]]:
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
    out = []
    for r in rows:
        d = {}
        for c, v in zip(cols, r):
            # normalize types: keep ints/floats/strings, else stringify
            if isinstance(v, (int, float, str)) or v is None:
                d[c] = v
            else:
                d[c] = str(v)
        out.append(d)
    return out

def results_match(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    """
    Execute and compare outputs (columns + rows) up to max_rows.
    Returns (match, error_msg_if_any).
    """
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    # Compare columns exactly (order matters)
    if cols_a != cols_b:
        return False, "COL_MISMATCH"

    can_a = canonicalize_result(cols_a, rows_a)
    can_b = canonicalize_result(cols_b, rows_b)

    return can_a == can_b, None


# -------------------------
# Prompting
# -------------------------
def build_prompt(schema_cols: List[str], question: str, table: str) -> str:
    schema_text = "\n".join([f'- "{c}"' for c in schema_cols])
    return f"""You are a SQL generator. Write a single SQLite SELECT query for the table "{table}".

Rules:
- Output ONLY the SQL query (no backticks, no explanation).
- Use double quotes for column names exactly as listed.
- Use single quotes for string literals.
- Do NOT invent columns.
- The table name is exactly: {table}
- Avoid JOINs (only one table exists).
- If the question asks for certain fields, SELECT only those fields; otherwise select a reasonable set of relevant columns.

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
    # remove code fences if any
    t = re.sub(r"^```sql\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # take first statement up to semicolon (if multiple)
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


# -------------------------
# JSONL IO + resume
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def compute_item_id(rec: Dict[str, Any]) -> str:
    if rec.get("item_id"):
        return str(rec["item_id"])
    # fallback hash
    q = (rec.get("new_question") or rec.get("original_question") or "").strip()
    s = (rec.get("new_sql") or rec.get("original_sql") or "").strip()
    h = hashlib.sha1((q + "\n" + s).encode("utf-8")).hexdigest()[:12]
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
    ap.add_argument("--max_rows_compare", type=int, default=200, help="max rows to fetch for execution match")
    ap.add_argument("--require_target_sql", action="store_true", help="skip items missing new_sql")

    # resumability
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush_every", type=int, default=1)

    # vLLM controls
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    # generation params
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # load input
    records = load_jsonl(args.input_jsonl)

    # slice
    if args.limit and args.limit > 0:
        records = records[args.start : args.start + args.limit]
    else:
        records = records[args.start :]

    if args.require_target_sql:
        records = [r for r in records if (r.get("new_sql") or r.get("new_gt_sql"))]

    if not records:
        print("No records to process.")
        return

    # resume filtering
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

    # db + schema
    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    if not schema_cols:
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")

    # vLLM init
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

    written = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        # batch prompts
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0 : b0 + args.batch_size]

            prompts = []
            for r in batch:
                q = (r.get(args.use_field) or "").strip()
                if not q:
                    # fallback
                    q = (r.get("original_question") or r.get("question") or "").strip()
                prompts.append(build_prompt(schema_cols, q, args.table_name))

            outputs = llm.generate(prompts, sampling)

            for r, out in zip(batch, outputs):
                iid = r["_item_id"]
                question_used = (r.get(args.use_field) or r.get("original_question") or r.get("question") or "")

                target_sql = (r.get("new_sql") or r.get("new_gt_sql") or "").strip()
                if not target_sql:
                    target_sql = (r.get("original_sql") or r.get("empty_gt_sql") or "").strip()

                gen_text = ""
                if out.outputs:
                    gen_text = (out.outputs[0].text or "").strip()

                pred_sql_raw = extract_sql(gen_text)
                pred_sql = pred_sql_raw

                # exact match (normalized)
                exact = False
                if pred_sql and target_sql:
                    exact = normalize_sql(pred_sql) == normalize_sql(target_sql)

                # execution match
                exec_match = False
                exec_err = None
                if pred_sql and target_sql:
                    exec_match, exec_err = results_match(
                        conn, pred_sql, target_sql, max_rows=args.max_rows_compare
                    )

                row = {
                    "item_id": iid,
                    "question_used": question_used,
                    "target_sql": target_sql,
                    "pred_sql": pred_sql,
                    "normalized_exact_match": bool(exact),
                    "execution_match": bool(exec_match),
                    "execution_error": exec_err,
                    "model_raw_output": gen_text,
                }

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                if written % args.flush_every == 0:
                    out_f.flush()
                    os.fsync(out_f.fileno())

            if written % 50 == 0:
                print(f"Processed {written}/{len(todo)}")

    conn.close()
    print(f"Done. Appended {written} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()