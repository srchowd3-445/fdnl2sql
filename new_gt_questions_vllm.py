#!/usr/bin/env python3
"""
new_gt_questions_vllm.py (resumable)

- Reads input JSON (list) from empty_gt_replaced_v4_fixed.json
- For each item with non-null new_gt_sql:
  - Executes new_gt_sql (first 3 rows)
  - Calls Gemma via vLLM to rewrite the question to match new_gt_sql
  - Writes one JSON object per line to output_jsonl (append-safe)
- If --resume is set, it loads output_jsonl and skips already processed items.

Output JSONL fields:
  item_id, original_question, original_sql, new_question, new_sql, new_sql_results
"""

import argparse
import json
import os
import sqlite3
import hashlib
from typing import Any, Dict, List, Set

from vllm import LLM, SamplingParams


def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def execute_sql_preview(conn: sqlite3.Connection, sql: str, max_rows: int = 3) -> List[Dict[str, Any]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return [dict(zip(cols, r)) for r in rows]


def sql_has_any_row(conn: sqlite3.Connection, sql: str) -> bool:
    sql0 = strip_trailing_semicolon(sql)
    try:
        cur = conn.execute(f"SELECT 1 FROM ({sql0}) AS subq LIMIT 1;")
        return cur.fetchone() is not None
    except Exception:
        cur = conn.execute(sql0)
        return cur.fetchone() is not None


def build_prompt(original_question: str, new_sql: str) -> str:
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


def compute_item_id(record: Dict[str, Any]) -> str:
    """
    Prefer stable 'line_number' if present; else hash (question + new_gt_sql).
    """
    if "line_number" in record and record["line_number"] is not None:
        return f"line_{record['line_number']}"
    q = (record.get("question") or "").strip()
    s = (record.get("new_gt_sql") or "").strip()
    h = hashlib.sha1((q + "\n" + s).encode("utf-8")).hexdigest()[:12]
    return f"hash_{h}"


def load_processed_ids(output_jsonl: str) -> Set[str]:
    processed: Set[str] = set()
    if not os.path.exists(output_jsonl):
        return processed
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid:
                    processed.add(iid)
            except Exception:
                # ignore malformed lines
                continue
    return processed

import re

def extract_where_conditions(sql: str) -> List[str]:
    """
    Extracts AND-split WHERE conditions from a SQL string.
    Returns [] if no WHERE.
    """
    if not sql:
        return []
    sql0 = strip_trailing_semicolon(sql)

    m = re.search(r"\bWHERE\b", sql0, flags=re.IGNORECASE)
    if not m:
        return []

    after = sql0[m.end():].strip()

    # cut off GROUP BY / ORDER BY / LIMIT if present
    m2 = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b", after, flags=re.IGNORECASE)
    if m2:
        after = after[:m2.start()].strip()

    # split on AND
    parts = [p.strip() for p in re.split(r"\s+AND\s+", after, flags=re.IGNORECASE) if p.strip()]

    # normalize whitespace
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return parts


def diff_filters(original_sql: str, new_sql: str) -> Dict[str, List[str]]:
    """
    Returns filters_added / filters_removed based on WHERE condition set difference.
    """
    orig = set(extract_where_conditions(original_sql))
    new = set(extract_where_conditions(new_sql))

    return {
        "filters_added": sorted(list(new - orig)),
        "filters_removed": sorted(list(orig - new)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--db_path", required=True)

    # output as JSONL for resumability
    ap.add_argument("--output_jsonl", required=True, help="Append-safe JSONL output path")

    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
        help="Local HF snapshot directory for Gemma-3-27b-it",
    )

    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--require_nonempty", action="store_true")

    ap.add_argument("--max_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--resume", action="store_true", help="Resume by skipping already processed item_ids")
    ap.add_argument("--flush_every", type=int, default=1, help="Flush to disk every N writes (default 1)")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    conn = sqlite3.connect(args.db_path)

    eligible = [r for r in records if r.get("new_gt_sql")]

    if args.require_nonempty:
        filtered = []
        for r in eligible:
            try:
                if sql_has_any_row(conn, r["new_gt_sql"]):
                    filtered.append(r)
            except Exception:
                pass
        eligible = filtered

    eligible = eligible[args.start : args.start + args.limit]
    if not eligible:
        conn.close()
        print("No eligible items.")
        return

    processed_ids: Set[str] = set()
    if args.resume:
        processed_ids = load_processed_ids(args.output_jsonl)
        print(f"[resume] loaded {len(processed_ids)} processed item_ids from output.")

    # Filter out already processed
    todo = []
    for r in eligible:
        iid = compute_item_id(r)
        if args.resume and iid in processed_ids:
            continue
        r["_item_id"] = iid
        todo.append(r)

    if not todo:
        conn.close()
        print("Nothing to do (all items already processed).")
        return

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

    # We'll do generation in batches for efficiency
    prompts = [build_prompt(r.get("question", ""), r["new_gt_sql"]) for r in todo]
    outputs = llm.generate(prompts, sampling)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    written = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for r, out in zip(todo, outputs):
            iid = r["_item_id"]
            original_question = r.get("question", "")
            original_sql = r.get("empty_gt_sql", "")
            new_sql = r.get("new_gt_sql", "")

            # new SQL preview
            try:
                preview = execute_sql_preview(conn, new_sql, max_rows=3)
            except Exception as e:
                preview = [{"SQL_ERROR": str(e)}]

            # vLLM output text
            gen_text = ""
            if out.outputs:
                gen_text = (out.outputs[0].text or "").strip()
            new_question = gen_text.splitlines()[0].strip() if gen_text else ""

            filter_diff = diff_filters(original_sql, new_sql)

            row = {
                "item_id": iid,
                "original_question": original_question,
                "original_sql": original_sql,
                "new_question": new_question,
                "new_sql": new_sql,
                "filters_added": filter_diff["filters_added"],
                "filters_removed": filter_diff["filters_removed"],
                "new_sql_results": preview,
            }



            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written % args.flush_every == 0:
                out_f.flush()
                os.fsync(out_f.fileno())

            if written % 20 == 0:
                print(f"Written {written}/{len(todo)}")

    conn.close()
    print(f"Done. Appended {written} rows to: {args.output_jsonl}")


if __name__ == "__main__":
    main()