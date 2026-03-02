#!/usr/bin/env python3
"""Compare staged Qwen method batches against multiple Qwen baseline eval JSON files.

Outputs:
- Per-baseline detailed comparison JSON
- One combined summary JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from compare_fewshot_vs_staged_batches import (
    compare_one,
    compare_overall,
    load_eval,
    print_report,
    to_item_map,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    default_baselines = [
        root / "results" / "eval_baselines_v3" / "20260226_062742" / "eval_qwen3_30b_a3b_cot.json",
        root / "results" / "eval_baselines_v3" / "20260226_062742" / "eval_qwen3_30b_a3b_few_shot.json",
        root / "results" / "eval_baselines_v3" / "20260226_062742" / "eval_qwen3_30b_a3b_zero_shot.json",
    ]
    default_batches = [
        root / "method" / "batch_qwen3_30b" / "batch_1" / "batch_1.eval.json",
        root / "method" / "batch_qwen3_30b" / "batch_2" / "batch_2.eval.json",
        root / "method" / "batch_qwen3_30b" / "batch_3" / "batch_3.eval.json",
    ]

    ap = argparse.ArgumentParser(description="Evaluate Qwen method batches vs multiple Qwen baseline eval JSON files.")
    ap.add_argument(
        "--baseline-evals",
        nargs="+",
        default=[str(p) for p in default_baselines],
        help="Baseline eval JSON paths (e.g., cot/few-shot/zero-shot).",
    )
    ap.add_argument(
        "--batch-evals",
        nargs="+",
        default=[str(p) for p in default_batches],
        help="Method batch eval JSON paths.",
    )
    ap.add_argument(
        "--id-source",
        choices=["pred_nonempty", "all"],
        default="pred_nonempty",
        help="Question set per batch: rows with non-empty pred_sql, or all rows.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(root / "results" / "qwen_method_vs_baselines"),
        help="Directory to save comparison reports.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_objs: Dict[str, Dict[str, Any]] = {}
    batch_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for p in args.batch_evals:
        path = Path(p)
        obj = load_eval(path)
        batch_objs[path.stem] = obj
        batch_rows_by_name[path.stem] = obj["per_item"]

    summary_rows: List[Dict[str, Any]] = []

    for bpath in args.baseline_evals:
        baseline_path = Path(bpath)
        baseline_obj = load_eval(baseline_path)
        baseline_map = to_item_map(baseline_obj["per_item"])

        print("=" * 100)
        print(f"Baseline: {baseline_path.name}")

        per_batch_results: List[Dict[str, Any]] = []
        for mpath in args.batch_evals:
            mp = Path(mpath)
            obj = batch_objs[mp.stem]
            res = compare_one(
                batch_name=mp.stem,
                batch_rows=obj["per_item"],
                few_map=baseline_map,
                id_source=args.id_source,
            )
            per_batch_results.append(res)
            print_report(res)

        overall = compare_overall(
            batch_results=per_batch_results,
            batch_rows_by_name=batch_rows_by_name,
            few_map=baseline_map,
            id_source=args.id_source,
        )
        print_report(overall)

        report = {
            "baseline_eval": str(baseline_path),
            "batch_evals": [str(Path(x)) for x in args.batch_evals],
            "id_source": args.id_source,
            "results": per_batch_results,
            "overall": overall,
        }
        out_file = out_dir / f"{baseline_path.stem}_vs_method_batches.json"
        out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        overall_metrics = overall.get("metrics", {})
        em_block = overall_metrics.get("exact_exec_match", {}) if isinstance(overall_metrics, dict) else {}
        f1_block = overall_metrics.get("f1", {}) if isinstance(overall_metrics, dict) else {}
        ast_block = overall_metrics.get("sql_ast_similarity", {}) if isinstance(overall_metrics, dict) else {}
        summary_rows.append(
            {
                "baseline": baseline_path.stem,
                "common_ids": (overall.get("counts") or {}).get("common_ids"),
                "exact_exec_match_batch": em_block.get("batch"),
                "exact_exec_match_baseline": em_block.get("fewshot"),
                "exact_exec_match_delta": em_block.get("delta_batch_minus_fewshot"),
                "f1_batch": f1_block.get("batch"),
                "f1_baseline": f1_block.get("fewshot"),
                "f1_delta": f1_block.get("delta_batch_minus_fewshot"),
                "sql_ast_similarity_delta": ast_block.get("delta_batch_minus_fewshot"),
            }
        )

    summary = {
        "baseline_evals": [str(Path(x)) for x in args.baseline_evals],
        "batch_evals": [str(Path(x)) for x in args.batch_evals],
        "id_source": args.id_source,
        "summary_rows": summary_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Overall Summary (method - baseline)")
    for r in summary_rows:
        print(
            f"{r['baseline']}: "
            f"EM delta={float(r['exact_exec_match_delta'] or 0.0):+0.4f}, "
            f"F1 delta={float(r['f1_delta'] or 0.0):+0.4f}, "
            f"AST delta={float(r['sql_ast_similarity_delta'] or 0.0):+0.4f}"
        )
    print(f"Saved reports to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
