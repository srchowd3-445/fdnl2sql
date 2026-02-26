#!/usr/bin/env python3
"""Sequential top-k retrieval for NL->SQL seed candidates.

Processes questions strictly one-by-one and saves results incrementally so runs
can be monitored/resumed safely.
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

    ap = argparse.ArgumentParser(description="Run top-k retrieval one question at a time.")

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
    ap.add_argument("--limit", type=int, default=-1, help="Use -1 for all remaining rows")

    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--save-every", type=int, default=1, help="Checkpoint frequency in processed rows")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--resume", type=int, choices=[0, 1], default=1)

    ap.add_argument(
        "--output-json",
        default=str(root / "method" / "retrieval_top3_all_questions_seq.json"),
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


def pick_question(row: Dict[str, Any], primary_key: str) -> tuple[str, Optional[str], Optional[str]]:
    q = row.get(primary_key)
    if isinstance(q, str) and q.strip():
        return q.strip(), primary_key, None

    for alt in ("natural_question", "question", "original_question"):
        v = row.get(alt)
        if isinstance(v, str) and v.strip():
            return v.strip(), alt, None

    return "", None, f"MISSING_QUESTION_KEY:{primary_key}"


def load_existing(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    rows = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        idx = r.get("question_index")
        if isinstance(idx, int):
            out[idx] = r
    return out


def save_payload(
    out_path: Path,
    args: argparse.Namespace,
    rsq,
    question_total: int,
    candidates_count: int,
    results_by_idx: Dict[int, Dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [results_by_idx[k] for k in sorted(results_by_idx)]

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
            "total_input_rows": question_total,
            "processed_rows": len(ordered),
            "candidate_count": candidates_count,
        },
        "results": ordered,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    rsq = import_retriever(root)

    questions_obj = json.loads(Path(args.question_json).read_text(encoding="utf-8"))
    if not isinstance(questions_obj, list):
        raise ValueError(f"Expected JSON list in {args.question_json}")

    start = max(0, int(args.start))
    end = len(questions_obj)
    if int(args.limit) > -1:
        end = min(end, start + int(args.limit))

    candidates = load_candidates(args, rsq)
    if not candidates:
        raise SystemExit("No candidates loaded. Check candidate source format/path.")

    out_path = Path(args.output_json)
    results_by_idx = load_existing(out_path) if int(args.resume) == 1 else {}

    processed_now = 0
    attempted = 0
    save_every = max(1, int(args.save_every))
    progress_every = max(1, int(args.progress_every))

    for idx in range(start, end):
        attempted += 1
        if idx in results_by_idx:
            continue

        row = questions_obj[idx]
        if not isinstance(row, dict):
            results_by_idx[idx] = {
                "question_index": idx,
                "item_id": None,
                "question": "",
                "question_key_used": None,
                "error": "ROW_NOT_OBJECT",
                "top_k": [],
            }
            processed_now += 1
        else:
            question, key_used, q_err = pick_question(row, args.question_key)
            if q_err:
                results_by_idx[idx] = {
                    "question_index": idx,
                    "item_id": row.get(args.id_key),
                    "question": "",
                    "question_key_used": key_used,
                    "error": q_err,
                    "top_k": [],
                }
                processed_now += 1
            else:
                ranked = rsq.rank_candidates(question, candidates, top_k=max(1, int(args.top_k)))
                results_by_idx[idx] = {
                    "question_index": idx,
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
                processed_now += 1

        if processed_now % save_every == 0:
            save_payload(
                out_path=out_path,
                args=args,
                rsq=rsq,
                question_total=len(questions_obj),
                candidates_count=len(candidates),
                results_by_idx=results_by_idx,
            )

        if attempted % progress_every == 0 or idx == end - 1:
            print(
                f"Attempted {attempted}/{end - start}; "
                f"newly processed this run: {processed_now}; "
                f"saved rows total: {len(results_by_idx)}"
            )

    save_payload(
        out_path=out_path,
        args=args,
        rsq=rsq,
        question_total=len(questions_obj),
        candidates_count=len(candidates),
        results_by_idx=results_by_idx,
    )

    errors = sum(1 for v in results_by_idx.values() if isinstance(v, dict) and v.get("error"))
    print("Saved:", out_path)
    print("Rows in output:", len(results_by_idx))
    print("Rows with errors:", errors)


if __name__ == "__main__":
    main()
