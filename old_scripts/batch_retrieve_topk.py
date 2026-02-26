#!/usr/bin/env python3
"""Batch retrieval for NL->SQL seeds over a question dataset.

Loads candidates once, scores each question with retrieve_similar_queries.py logic,
and writes top-k ranked results per question to one JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Run top-k retrieval for every question in a dataset.")

    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    source.add_argument("--candidate-json")
    source.add_argument("--candidate-sqlite")

    ap.add_argument("--candidate-table", default="query_library")
    ap.add_argument("--candidate-question-col", default="question")
    ap.add_argument("--candidate-sql-col", default="sql")
    ap.add_argument("--candidate-id-col", default="id")

    ap.add_argument("--question-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--id-key", default="item_id")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1, help="Use -1 for all remaining")

    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--progress-every", type=int, default=100)

    ap.add_argument(
        "--output-json",
        default=str(root / "method" / "retrieval_top3_all_questions.json"),
    )

    return ap.parse_args()


def import_retriever(root: Path):
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import retrieve_similar_queries as rsq  # type: ignore

    return rsq


def load_candidates(args: argparse.Namespace, rsq) -> List[Any]:
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


def pick_question(row: Dict[str, Any], primary_key: str) -> tuple[str, Optional[str], str]:
    q = row.get(primary_key)
    if isinstance(q, str) and q.strip():
        return q.strip(), primary_key, ""

    for alt in ("natural_question", "question", "original_question"):
        v = row.get(alt)
        if isinstance(v, str) and v.strip():
            return v.strip(), alt, ""

    return "", None, f"MISSING_QUESTION_KEY:{primary_key}"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    rsq = import_retriever(root)

    questions_obj = json.loads(Path(args.question_json).read_text(encoding="utf-8"))
    if not isinstance(questions_obj, list):
        raise ValueError(f"Expected JSON list in {args.question_json}")

    start = max(0, int(args.start))
    rows = questions_obj[start:]
    if int(args.limit) > -1:
        rows = rows[: int(args.limit)]

    candidates = load_candidates(args, rsq)
    if not candidates:
        raise SystemExit("No candidates loaded. Check candidate source format/path.")

    out_rows: List[Dict[str, Any]] = []
    total = len(rows)
    prog_every = max(1, int(args.progress_every))

    for i, row in enumerate(rows):
        global_index = start + i

        if not isinstance(row, dict):
            out_rows.append(
                {
                    "question_index": global_index,
                    "item_id": None,
                    "question": "",
                    "question_key_used": None,
                    "error": "ROW_NOT_OBJECT",
                    "top_k": [],
                }
            )
            continue

        question, key_used, q_err = pick_question(row, args.question_key)
        if q_err:
            out_rows.append(
                {
                    "question_index": global_index,
                    "item_id": row.get(args.id_key),
                    "question": "",
                    "question_key_used": key_used,
                    "error": q_err,
                    "top_k": [],
                }
            )
            continue

        ranked = rsq.rank_candidates(question, candidates, top_k=max(1, int(args.top_k)))

        out_rows.append(
            {
                "question_index": global_index,
                "item_id": row.get(args.id_key),
                "question": question,
                "question_key_used": key_used,
                "error": None,
                "top_k": [
                    {
                        **asdict(m),
                        "candidate": asdict(m.candidate),
                    }
                    for m in ranked
                ],
            }
        )

        done = i + 1
        if done % prog_every == 0 or done == total:
            print(f"Processed {done}/{total}")

    payload: Dict[str, Any] = {
        "meta": {
            "question_json": args.question_json,
            "question_key": args.question_key,
            "id_key": args.id_key,
            "start": int(args.start),
            "limit": int(args.limit),
            "top_k": int(args.top_k),
            "seed_json": args.seed_json,
            "candidate_json": args.candidate_json,
            "candidate_sqlite": args.candidate_sqlite,
            "candidate_table": args.candidate_table,
            "candidate_question_col": args.candidate_question_col,
            "candidate_sql_col": args.candidate_sql_col,
            "candidate_id_col": args.candidate_id_col,
            "weights": {
                "lexical": rsq.W_LEXICAL,
                "char": rsq.W_CHAR,
                "literal": rsq.W_LITERAL,
                "operator": rsq.W_OPERATOR,
                "column": rsq.W_COLUMN,
            },
            "total_input_rows": len(questions_obj),
            "processed_rows": len(out_rows),
            "candidate_count": len(candidates),
        },
        "results": out_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    error_rows = sum(1 for r in out_rows if r.get("error"))
    print("Saved:", out_path)
    print("Processed rows:", len(out_rows))
    print("Rows with errors:", error_rows)


if __name__ == "__main__":
    main()
