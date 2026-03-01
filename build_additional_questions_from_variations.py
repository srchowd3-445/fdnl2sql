#!/usr/bin/env python3
"""
Generate additional SQL-question records not already present in natural_question_1500.json.

This script reuses the structural SQL variation strategy from empty_gt_build_final_variations.py
and enforces novelty against an existing dataset by:
- normalized gt_sql
- normalized question text

Output: JSON list containing only newly generated extra records.
"""

import argparse
import json
import random
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from empty_gt_build_final_variations import (
    build_base_records,
    load_dataset_jsonl,
    make_variant_question,
    mutate_drop_select_column,
    mutate_drop_two_where_conditions,
    mutate_drop_where_condition,
    mutate_keep_one_where_condition,
    mutate_keep_two_select_columns,
    mutate_relax_numeric,
    mutate_remove_where,
    mutate_swap_text_equality_value,
    normalize_sql,
    parse_sql_parts,
    query_returns_rows,
)


def normalize_question_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_existing_sets(existing_rows: List[Dict[str, Any]]) -> Tuple[set, set]:
    seen_sql = set()
    seen_q = set()

    for r in existing_rows:
        sql = normalize_sql(r.get("gt_sql", ""))
        if sql:
            seen_sql.add(sql)

        q = (
            (r.get("natural_question") or "").strip()
            or (r.get("question") or "").strip()
        )
        qn = normalize_question_text(q)
        if qn:
            seen_q.add(qn)

    return seen_sql, seen_q


def generate_extra_records(
    conn: sqlite3.Connection,
    table_name: str,
    base_records: List[Dict[str, Any]],
    seen_sql: set,
    seen_q: set,
    extra_count: int,
    seed: int,
    max_attempts: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)

    value_cache: Dict[str, List[str]] = {}
    variation_fns = [
        ("drop_select_column", mutate_drop_select_column),
        ("keep_two_select_columns", mutate_keep_two_select_columns),
        ("drop_where_condition", mutate_drop_where_condition),
        ("drop_two_where_conditions", mutate_drop_two_where_conditions),
        ("keep_one_where_condition", mutate_keep_one_where_condition),
        ("relax_numeric", mutate_relax_numeric),
        ("swap_text_equality_value", lambda p, r: mutate_swap_text_equality_value(conn, table_name, p, r, value_cache)),
        ("remove_where", mutate_remove_where),
    ]

    per_type = {name: 0 for name, _ in variation_fns}
    out: List[Dict[str, Any]] = []

    stats = {
        "target_extra": int(extra_count),
        "generated_extra": 0,
        "attempts": 0,
        "dup_sql_skipped": 0,
        "dup_question_skipped": 0,
        "non_working_sql_skipped": 0,
        "invalid_sql_skipped": 0,
    }

    while len(out) < extra_count and stats["attempts"] < max_attempts:
        stats["attempts"] += 1

        base = rng.choice(base_records)
        parts = parse_sql_parts(base.get("gt_sql", ""))
        if not parts:
            stats["invalid_sql_skipped"] += 1
            continue

        vname, vfn = rng.choice(variation_fns)
        res = vfn(parts, rng)
        if not res:
            continue

        new_sql, note, vmeta = res
        norm_sql = normalize_sql(new_sql)
        if not norm_sql:
            stats["invalid_sql_skipped"] += 1
            continue

        if norm_sql in seen_sql:
            stats["dup_sql_skipped"] += 1
            continue

        ok, err = query_returns_rows(conn, new_sql)
        if not ok:
            stats["non_working_sql_skipped"] += 1
            continue

        new_q = make_variant_question(base.get("question", ""), note)
        norm_q = normalize_question_text(new_q)
        if norm_q in seen_q:
            stats["dup_question_skipped"] += 1
            continue

        per_type[vname] += 1
        rec = {
            "item_id": f"extra_{len(out)+1}_{base.get('line_number')}_{vname}_{per_type[vname]}",
            "line_number": int(base.get("line_number", 0)),
            "question": new_q,
            "natural_question": new_q,
            "gt_sql": new_sql,
            "source": "variation_extra",
            "base_source": base.get("source", ""),
            "variation_type": vname,
            "variation_note": note,
            "row_exists": True,
            "metadata": {
                "parent_item_id": base.get("item_id"),
                "parent_sql": base.get("gt_sql"),
                "mutation": vmeta,
                "error": err,
                "generated_by": "build_additional_questions_from_variations.py",
            },
        }

        out.append(rec)
        seen_sql.add(norm_sql)
        seen_q.add(norm_q)
        stats["generated_extra"] += 1

    for k, v in per_type.items():
        stats[f"variation_{k}"] = int(v)

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", default="data/database.db")
    ap.add_argument("--input_jsonl", default="data/dataset.jsonl")
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--fixed_json", default="data/empty_gt_replaced_v5_fixed.json")

    ap.add_argument("--existing_json", default="data/natural_question_1500.json")
    ap.add_argument("--output_json", default="data/natural_question_additional_200.json")

    ap.add_argument("--extra_count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--max_attempts", type=int, default=40000)
    ap.add_argument("--strict_count", type=int, default=1)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db_path)

    dataset_rows = load_dataset_jsonl(args.input_jsonl)
    fixed_rows = load_json(args.fixed_json)
    if not isinstance(fixed_rows, list):
        raise ValueError("--fixed_json must be a JSON list")

    existing_rows = load_json(args.existing_json)
    if not isinstance(existing_rows, list):
        raise ValueError("--existing_json must be a JSON list")

    base_records, base_stats = build_base_records(conn, dataset_rows, fixed_rows)
    seen_sql, seen_q = build_existing_sets(existing_rows)

    extra_records, stats = generate_extra_records(
        conn=conn,
        table_name=args.table_name,
        base_records=base_records,
        seen_sql=seen_sql,
        seen_q=seen_q,
        extra_count=args.extra_count,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )

    conn.close()

    extra_records.sort(key=lambda r: r.get("item_id", ""))
    dump_json(args.output_json, extra_records)

    print("\n================ ADDITIONAL QUESTION BUILD SUMMARY ================")
    print(f"Existing dataset rows:            {len(existing_rows)}")
    print(f"Base records available:           {len(base_records)}")
    print(f"Base original_working:            {base_stats['original_working']}")
    print(f"Base fixed_working:               {base_stats['fixed_working']}")
    print(f"Target extra count:               {args.extra_count}")
    print(f"Generated extra count:            {len(extra_records)}")
    print(f"Attempts:                         {stats['attempts']}")
    print(f"Duplicate SQL skipped:            {stats['dup_sql_skipped']}")
    print(f"Duplicate question skipped:       {stats['dup_question_skipped']}")
    print(f"Non-working SQL skipped:          {stats['non_working_sql_skipped']}")
    print(f"Invalid SQL skipped:              {stats['invalid_sql_skipped']}")
    print(f"Variation counts: drop_select_column={stats['variation_drop_select_column']}, "
          f"keep_two_select_columns={stats['variation_keep_two_select_columns']}, "
          f"drop_where_condition={stats['variation_drop_where_condition']}, "
          f"drop_two_where_conditions={stats['variation_drop_two_where_conditions']}, "
          f"keep_one_where_condition={stats['variation_keep_one_where_condition']}, "
          f"relax_numeric={stats['variation_relax_numeric']}, "
          f"swap_text_equality_value={stats['variation_swap_text_equality_value']}, "
          f"remove_where={stats['variation_remove_where']}")
    print(f"Output JSON:                      {args.output_json}")
    print("===================================================================\n")

    if bool(args.strict_count) and len(extra_records) != int(args.extra_count):
        raise RuntimeError(
            f"Generated {len(extra_records)} records, expected {args.extra_count}. "
            "Increase --max_attempts, change --seed, or relax constraints."
        )


if __name__ == "__main__":
    main()
