#!/usr/bin/env python3
"""Quick demo: retrieve best seed query matches for one natural question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import retrieve_similar_queries as rsq


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Demo retrieval using one question from natural_question_1500.json")
    ap.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    ap.add_argument("--natural-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--question-index", type=int, default=9)
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--preview-rows", type=int, default=5)
    ap.add_argument("--output-json", default=str(root / "results" / "retrieval_demo_output.json"))
    args = ap.parse_args()

    question = rsq.load_question_from_dataset(Path(args.natural_json), args.question_index, args.question_key)
    candidates = rsq.load_candidates_from_seed_json(Path(args.seed_json))
    ranked = rsq.rank_candidates(question, candidates, top_k=args.top_k)

    print(f"Question index: {args.question_index}")
    print("Question:")
    print(question)
    print("=" * 90)

    for m in ranked:
        print(f"Rank {m.rank} | score={m.total_score:.4f} | id={m.candidate.candidate_id}")
        print(f"  candidate_question: {m.candidate.question}")
        print(f"  candidate_sql: {m.candidate.sql}")
        print("-" * 90)

    preview = None
    if ranked:
        best = ranked[0]
        cols, rows, err = rsq.execute_sql_preview(Path(args.db_path), best.candidate.sql, args.preview_rows)
        preview = {
            "candidate_id": best.candidate.candidate_id,
            "columns": cols,
            "rows": rows,
            "error": err,
        }
        print("Top-1 SQL preview:")
        if err:
            print(f"  error: {err}")
        else:
            print(f"  columns: {cols}")
            for r in rows:
                print(f"  {r}")

    output = {
        "question_index": args.question_index,
        "question": question,
        "top_k": args.top_k,
        "ranked_results": [
            {
                "rank": m.rank,
                "score": m.total_score,
                "lexical": m.lexical_score,
                "char": m.char_score,
                "literal": m.literal_score,
                "operator": m.operator_score,
                "column": m.column_score,
                "candidate_id": m.candidate.candidate_id,
                "candidate_question": m.candidate.question,
                "candidate_sql": m.candidate.sql,
                "parent_question": m.candidate.parent_question,
                "source": m.candidate.source,
            }
            for m in ranked
        ],
        "top1_sql_preview": preview,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved demo output: {out_path}")


if __name__ == "__main__":
    main()
