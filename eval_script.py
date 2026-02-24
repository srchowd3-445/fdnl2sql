#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import pandas as pd


def read_progress_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return pd.DataFrame.from_records(rows)


def load_run_df(run_dir: Path) -> pd.DataFrame:
    eval_csv = run_dir / "evaluation.csv"
    prog_jsonl = run_dir / "progress.jsonl"

    if eval_csv.exists():
        return pd.read_csv(eval_csv)

    if prog_jsonl.exists():
        return read_progress_jsonl(prog_jsonl)

    raise FileNotFoundError(f"No evaluation.csv or progress.jsonl found in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dir",
        type=str,
        help="Path to the run folder containing evaluation.csv or progress.jsonl",
    )
    ap.add_argument("--top", type=int, default=0, help="If >0, show top-N by sql_jaccard")
    ap.add_argument("--bottom", type=int, default=0, help="If >0, show bottom-N by sql_jaccard")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_run_df(run_dir)

    # Coerce common columns
    if "relaxed_exact_match" in df.columns:
        rem = df["relaxed_exact_match"].map(
            lambda x: x if isinstance(x, bool) else (str(x).lower() == "true")
        )
    else:
        rem = pd.Series([None] * len(df))

    if "sql_jaccard" in df.columns:
        sj = pd.to_numeric(df["sql_jaccard"], errors="coerce")
    else:
        sj = pd.Series([None] * len(df), dtype="float")

    # "Defined" relaxed EM rows are those where relaxed_exact_match is not null
    defined_mask = rem.notna()

    relaxed_em_rate = float(rem[defined_mask].mean()) if defined_mask.any() else None
    avg_sql_jaccard = float(sj[defined_mask].mean()) if defined_mask.any() else None

    print("========================================")
    print("RELAXED EM SUMMARY")
    print(f"Run dir: {run_dir}")
    print(f"Rows total: {len(df)}")
    print(f"Rows with relaxed EM defined: {int(defined_mask.sum())}")
    print(f"Relaxed EM rate: {relaxed_em_rate if relaxed_em_rate is not None else 'NA'}")
    print(f"Avg sql_jaccard (defined rows): {avg_sql_jaccard if avg_sql_jaccard is not None else 'NA'}")
    if "relaxed_em_threshold" in df.columns:
        thr = pd.to_numeric(df["relaxed_em_threshold"], errors="coerce").dropna()
        if len(thr) > 0:
            print(f"Threshold (from file): {float(thr.iloc[0])}")
    print("========================================\n")

    # Per-row view
    cols = []
    for c in ["row_index", "question", "sql_jaccard", "relaxed_exact_match", "pred_exec_status", "gt_exec_status"]:
        if c in df.columns:
            cols.append(c)

    out = df.copy()
    if "question" in out.columns:
        out["question"] = out["question"].astype(str).str.replace("\n", " ").str.slice(0, 80)

    if cols:
        out_show = out[cols].copy()
    else:
        out_show = out.copy()

    # Sort by row_index if present
    if "row_index" in out_show.columns:
        out_show = out_show.sort_values("row_index")

    # Print main table
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
        print(out_show.to_string(index=False))

    # Optional top/bottom
    if args.top > 0 and "sql_jaccard" in out.columns:
        print("\n--- TOP by sql_jaccard ---")
        top = out.sort_values("sql_jaccard", ascending=False).head(args.top)
        top_cols = [c for c in ["row_index", "sql_jaccard", "relaxed_exact_match", "question"] if c in top.columns]
        print(top[top_cols].to_string(index=False))

    if args.bottom > 0 and "sql_jaccard" in out.columns:
        print("\n--- BOTTOM by sql_jaccard ---")
        bot = out.sort_values("sql_jaccard", ascending=True).head(args.bottom)
        bot_cols = [c for c in ["row_index", "sql_jaccard", "relaxed_exact_match", "question"] if c in bot.columns]
        print(bot[bot_cols].to_string(index=False))


if __name__ == "__main__":
    main()
