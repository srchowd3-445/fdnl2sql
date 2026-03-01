#!/usr/bin/env python3
"""
Validate decomposed SQL queries by executing them against a SQLite database.

Input format:
- JSON list where each row may contain:
  - idx
  - item_id
  - decomposed_query: {
      "query_1": {"question": "...", "sql": "..."},
      ...
    }

Output:
- A JSON list preserving original rows and adding:
  - decomposed_query_validation
  - decomposed_query_validation_summary
  - decomposed_query_all_runnable
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate decomposed SQL queries against SQLite.")
    ap.add_argument("--input_json", default="data/decompose_single_test.json")
    ap.add_argument("--db_path", default="data/database.db")
    ap.add_argument("--output_json", default="", help="Default: <input_json>.validation.json")
    ap.add_argument("--preview_rows", type=int, default=3, help="Preview rows to capture per successful query")
    return ap.parse_args()


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("input_json must contain a JSON list.")
    return [x for x in obj if isinstance(x, dict)]


def json_safe_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, tuple)):
        return [json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe_value(val) for k, val in v.items()}
    return str(v)


def sorted_query_keys(d: Dict[str, Any]) -> List[str]:
    def kf(k: str) -> Tuple[int, str]:
        m = re.match(r"query_(\d+)$", k or "")
        if m:
            return (0, f"{int(m.group(1)):08d}")
        return (1, str(k))

    return sorted([str(k) for k in d.keys()], key=kf)


def is_select_like(sql: str) -> bool:
    s = (sql or "").strip().lower()
    return bool(re.match(r"^(select|with)\b", s))


def validate_one_sql(conn: sqlite3.Connection, sql: str, preview_rows: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "runnable": False,
        "error": None,
        "columns": [],
        "preview_rows": [],
    }

    sql_text = (sql or "").strip()
    if not sql_text:
        out["error"] = "EMPTY_SQL"
        return out
    if not is_select_like(sql_text):
        out["error"] = "NON_SELECT_SQL"
        return out

    try:
        cur = conn.execute(sql_text)
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchmany(max(0, int(preview_rows)))
        out["runnable"] = True
        out["columns"] = [str(c) for c in cols]
        out["preview_rows"] = [json_safe_value(list(r)) for r in rows]
        return out
    except Exception as exc:
        out["error"] = str(exc).splitlines()[0]
        return out


def validate_row(row: Dict[str, Any], conn: sqlite3.Connection, preview_rows: int) -> Dict[str, Any]:
    decomposed = row.get("decomposed_query")
    if not isinstance(decomposed, dict):
        out = dict(row)
        out["decomposed_query_validation"] = {}
        out["decomposed_query_validation_summary"] = {
            "total_queries": 0,
            "runnable_queries": 0,
            "failed_queries": 0,
        }
        out["decomposed_query_all_runnable"] = False
        return out

    qkeys = sorted_query_keys(decomposed)
    val_map: Dict[str, Any] = {}
    runnable = 0
    failed = 0

    for qk in qkeys:
        qobj = decomposed.get(qk)
        if not isinstance(qobj, dict):
            val_map[qk] = {
                "runnable": False,
                "error": "INVALID_QUERY_OBJECT",
                "columns": [],
                "preview_rows": [],
            }
            failed += 1
            continue

        sql = str(qobj.get("sql") or "")
        question = str(qobj.get("question") or "")
        v = validate_one_sql(conn, sql, preview_rows)
        val_map[qk] = {
            "question": question,
            "sql": sql,
            **v,
        }
        if v.get("runnable"):
            runnable += 1
        else:
            failed += 1

    out = dict(row)
    out["decomposed_query_validation"] = val_map
    out["decomposed_query_validation_summary"] = {
        "total_queries": len(qkeys),
        "runnable_queries": runnable,
        "failed_queries": failed,
    }
    out["decomposed_query_all_runnable"] = len(qkeys) > 0 and failed == 0
    return out


def main() -> None:
    args = parse_args()
    input_path = args.input_json
    output_path = args.output_json or (args.input_json + ".validation.json")

    rows = load_rows(input_path)
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row

    try:
        out_rows: List[Dict[str, Any]] = []
        total_q = 0
        ok_q = 0
        bad_q = 0

        for row in rows:
            validated = validate_row(row, conn, args.preview_rows)
            out_rows.append(validated)
            s = validated.get("decomposed_query_validation_summary") or {}
            total_q += int(s.get("total_queries", 0) or 0)
            ok_q += int(s.get("runnable_queries", 0) or 0)
            bad_q += int(s.get("failed_queries", 0) or 0)

        ensure_parent(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)

        print("\n=== Decomposed Query Validation ===")
        print(f"Input rows:      {len(rows)}")
        print(f"Total subqueries:{total_q}")
        print(f"Runnable:        {ok_q}")
        print(f"Failed:          {bad_q}")
        print(f"Output JSON:     {output_path}")
        print("===================================\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

