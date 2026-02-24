#!/usr/bin/env python3
"""
Evaluate model_baseline output JSON using the exact same evaluation logic
from model_baseline.py (evaluate_pair / Hungarian or multiset).

Defaults target baseline output rows that contain:
  - predicted SQL key: baseline_pred_sql
  - gold SQL key: gt_sql
"""

import argparse
import json
import os
import sqlite3
from statistics import mean
from typing import Any, Dict, List

from model_baseline import evaluate_pair, extract_sql


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--item_id_key", default="item_id")
    ap.add_argument("--pred_sql_key", default="baseline_pred_sql")
    ap.add_argument("--gold_sql_key", default="gt_sql")
    ap.add_argument("--eval_key", default="baseline_exec_eval_recomputed")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--extract_sql", type=int, default=1)

    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--match_mode", choices=["hungarian", "multiset"], default="hungarian")
    ap.add_argument("--hungarian_max_rows", type=int, default=400)
    ap.add_argument("--cell_text_metric", choices=["rouge1_f1", "exact"], default="rouge1_f1")
    ap.add_argument("--print_worst", type=int, default=10)
    args = ap.parse_args()

    rows = load_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError("input_json must contain a JSON list")

    rows = rows[args.start:]
    if args.limit is not None and args.limit > -1:
        rows = rows[:args.limit]

    conn = sqlite3.connect(args.db_path)

    enriched: List[Dict[str, Any]] = []
    errors = 0
    no_common_cols = 0
    ok = 0
    fallback_cnt = 0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    jaccs: List[float] = []
    soft_scores: List[float] = []
    rem_hits = 0

    worst: List[Any] = []

    for row in rows:
        pred_sql = row.get(args.pred_sql_key, "")
        gold_sql = row.get(args.gold_sql_key, "")

        if bool(args.extract_sql):
            pred_sql = extract_sql(pred_sql)
            gold_sql = extract_sql(gold_sql)

        eval_row = evaluate_pair(
            conn=conn,
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            max_rows=args.max_rows,
            match_mode=args.match_mode,
            hungarian_max_rows=args.hungarian_max_rows,
            cell_text_metric=args.cell_text_metric,
        )

        out_row = dict(row)
        out_row[args.eval_key] = eval_row
        enriched.append(out_row)

        if eval_row.get("eval_ok"):
            ok += 1
            p = float(eval_row.get("precision", 0.0))
            r = float(eval_row.get("recall", 0.0))
            f1 = float(eval_row.get("f1", 0.0))
            j = float(eval_row.get("row_jaccard", 0.0))
            s = float(eval_row.get("soft_overlap_score", 0.0))

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            jaccs.append(j)
            soft_scores.append(s)
            if bool(eval_row.get("relaxed_em_unordered")):
                rem_hits += 1
            if eval_row.get("match_mode_used") != eval_row.get("match_mode_requested"):
                fallback_cnt += 1

            worst.append(
                (
                    row.get(args.item_id_key),
                    f1,
                    p,
                    r,
                    int(eval_row.get("pred_row_count", 0)),
                    int(eval_row.get("gold_row_count", 0)),
                    int(eval_row.get("common_cols_count", 0)),
                    str(eval_row.get("match_mode_used", "")),
                    eval_row.get("error"),
                )
            )
        else:
            errors += 1
            if eval_row.get("error") == "NO_COMMON_COLS":
                no_common_cols += 1

    conn.close()

    total = len(rows)
    relaxed_em = (rem_hits / ok) if ok else 0.0

    print("\n================ BASELINE ANSWER EVAL SUMMARY ================")
    print(f"Input:                         {args.input_json}")
    print(f"DB:                            {args.db_path}")
    print(f"Pred SQL key:                  {args.pred_sql_key}")
    print(f"Gold SQL key:                  {args.gold_sql_key}")
    print(f"Eval key written:              {args.eval_key}")
    print(f"Match mode requested:          {args.match_mode}")
    print(
        f"Cell text metric:              "
        f"{args.cell_text_metric if args.match_mode == 'hungarian' else 'n/a'}"
    )
    print(f"Items evaluated:               {total}")
    print(f"Eval OK:                       {ok}")
    print(f"Eval errors:                   {errors}")
    print(f"NO_COMMON_COLS:                {no_common_cols}")
    print(f"Mode fallbacks used:           {fallback_cnt}")

    if ok:
        print(f"Avg precision:                 {mean(precisions):.4f} ({format_pct(mean(precisions))})")
        print(f"Avg recall:                    {mean(recalls):.4f} ({format_pct(mean(recalls))})")
        print(f"Avg F1:                        {mean(f1s):.4f} ({format_pct(mean(f1s))})")
        print(f"Avg row_jaccard:               {mean(jaccs):.4f} ({format_pct(mean(jaccs))})")
        print(f"Avg soft_overlap_score:        {mean(soft_scores):.4f}")
        print(f"Relaxed EM (unordered rows):   {rem_hits}/{ok} ({format_pct(relaxed_em)})")

    k = max(0, int(args.print_worst))
    if k > 0 and worst:
        worst.sort(key=lambda x: x[1])
        print(f"Worst {min(k, len(worst))} by F1:")
        for item_id, f1, p, r, pr, gr, cc, mode_used, err in worst[:k]:
            print(
                f"  - {item_id}: f1={f1:.4f}, p={p:.4f}, r={r:.4f}, "
                f"pred_rows={pr}, gold_rows={gr}, common_cols={cc}, mode={mode_used}, err={err}"
            )

    if args.output_json:
        dump_json(args.output_json, enriched)
        print(f"Output JSON:                   {args.output_json}")

    print("==============================================================\n")


if __name__ == "__main__":
    main()
