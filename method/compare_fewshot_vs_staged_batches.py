#!/usr/bin/env python3
"""Compare staged batch eval outputs against a few-shot eval on matching item_ids.

Each staged batch eval file (batch_1/2/3) is compared to the same few-shot eval file.
Comparison is done on the question set where the batch has non-empty `pred_sql`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BOOL_METRICS = [
    "exact_exec_match",
    "pred_exec_ok",
]

NUM_METRICS = [
    "sql_ast_similarity",
    "precision",
    "recall",
    "f1",
    "row_jaccard",
    "column_alignment_score",
    "chrf",
    "rouge_l_f1",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    default_few = root / "results" / "eval_baselines_v3" / "20260226_062742" / "eval_gemma27b_few_shot.json"
    default_batches = [
        root / "method" / "batch" / "batch_1" / "batch_1.eval.json",
        root / "method" / "batch" / "batch_2" / "batch_2.eval.json",
        root / "method" / "batch" / "batch_3" / "batch_3.eval.json",
    ]

    ap = argparse.ArgumentParser(description="Compare few-shot eval against staged batch eval files.")
    ap.add_argument("--fewshot-eval", default=str(default_few), help="Few-shot eval JSON path")
    ap.add_argument(
        "--batch-evals",
        nargs="+",
        default=[str(p) for p in default_batches],
        help="One or more staged batch eval JSON paths",
    )
    ap.add_argument(
        "--id-source",
        choices=["pred_nonempty", "all"],
        default="pred_nonempty",
        help="Question set to compare per batch: rows with non-empty pred_sql, or all rows",
    )
    ap.add_argument("--output-json", default="", help="Optional output JSON path")
    return ap.parse_args()


def load_eval(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("per_item"), list):
        raise ValueError(f"Invalid eval JSON format: {path}")
    return obj


def to_item_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        iid = str(r.get("item_id") or "").strip()
        if not iid:
            continue
        out[iid] = r
    return out


def nonempty_pred_sql(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("pred_sql") or "").strip())


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def get_num(row: Dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def get_bool(row: Dict[str, Any], key: str) -> int:
    return 1 if bool(row.get(key)) else 0


def scope_ids_from_rows(rows: Sequence[Dict[str, Any]], id_source: str) -> List[str]:
    if id_source == "pred_nonempty":
        scope_ids = [str(r.get("item_id")) for r in rows if isinstance(r, dict) and nonempty_pred_sql(r)]
    else:
        scope_ids = [str(r.get("item_id")) for r in rows if isinstance(r, dict) and str(r.get("item_id") or "").strip()]
    return [i for i in scope_ids if i]


def dedupe_keep_order(ids: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def compare_one(
    *,
    batch_name: str,
    batch_rows: Sequence[Dict[str, Any]],
    few_map: Dict[str, Dict[str, Any]],
    id_source: str,
) -> Dict[str, Any]:
    batch_scope_ids_raw = scope_ids_from_rows(batch_rows, id_source)
    batch_scope_ids = dedupe_keep_order(batch_scope_ids_raw)
    batch_map = to_item_map(batch_rows)

    common_ids = [i for i in batch_scope_ids if i in few_map and i in batch_map]
    missing_in_few = [i for i in batch_scope_ids if i not in few_map]

    metric_rows: Dict[str, Dict[str, float]] = {}
    for m in BOOL_METRICS:
        b_mean = mean(get_bool(batch_map[i], m) for i in common_ids)
        f_mean = mean(get_bool(few_map[i], m) for i in common_ids)
        metric_rows[m] = {
            "batch": b_mean,
            "fewshot": f_mean,
            "delta_batch_minus_fewshot": b_mean - f_mean,
        }

    for m in NUM_METRICS:
        b_vals = [get_num(batch_map[i], m) for i in common_ids]
        f_vals = [get_num(few_map[i], m) for i in common_ids]
        b_mean = mean(v for v in b_vals if v is not None)
        f_mean = mean(v for v in f_vals if v is not None)
        metric_rows[m] = {
            "batch": b_mean,
            "fewshot": f_mean,
            "delta_batch_minus_fewshot": b_mean - f_mean,
        }

    both = batch_only = few_only = neither = 0
    for i in common_ids:
        b = bool(batch_map[i].get("exact_exec_match"))
        f = bool(few_map[i].get("exact_exec_match"))
        if b and f:
            both += 1
        elif b and not f:
            batch_only += 1
        elif f and not b:
            few_only += 1
        else:
            neither += 1

    return {
        "batch": batch_name,
        "question_set_source": id_source,
        "counts": {
            "batch_scope_ids_raw": len(batch_scope_ids_raw),
            "batch_scope_ids": len(batch_scope_ids),
            "common_ids": len(common_ids),
            "missing_in_fewshot": len(missing_in_few),
            "deduped_scope_ids": len(batch_scope_ids_raw) - len(batch_scope_ids),
        },
        "exact_head_to_head": {
            "both_true": both,
            "batch_only_true": batch_only,
            "fewshot_only_true": few_only,
            "both_false": neither,
        },
        "metrics": metric_rows,
        "sample_common_ids": common_ids[:10],
    }


def compare_overall(
    *,
    batch_results: Sequence[Dict[str, Any]],
    batch_rows_by_name: Dict[str, Sequence[Dict[str, Any]]],
    few_map: Dict[str, Dict[str, Any]],
    id_source: str,
) -> Dict[str, Any]:
    scope_ids_raw: List[str] = []
    first_row_by_id: Dict[str, Dict[str, Any]] = {}
    first_batch_by_id: Dict[str, str] = {}

    for b in batch_results:
        name = str(b.get("batch") or "")
        rows = batch_rows_by_name.get(name, [])
        ids = scope_ids_from_rows(rows, id_source)
        row_map = to_item_map(rows)
        scope_ids_raw.extend(ids)
        for iid in ids:
            if iid in first_row_by_id:
                continue
            r = row_map.get(iid)
            if r is None:
                continue
            first_row_by_id[iid] = r
            first_batch_by_id[iid] = name

    scope_ids = dedupe_keep_order(scope_ids_raw)
    common_ids = [i for i in scope_ids if i in few_map and i in first_row_by_id]
    missing_in_few = [i for i in scope_ids if i not in few_map]

    metric_rows: Dict[str, Dict[str, float]] = {}
    for m in BOOL_METRICS:
        b_mean = mean(get_bool(first_row_by_id[i], m) for i in common_ids)
        f_mean = mean(get_bool(few_map[i], m) for i in common_ids)
        metric_rows[m] = {
            "batch": b_mean,
            "fewshot": f_mean,
            "delta_batch_minus_fewshot": b_mean - f_mean,
        }

    for m in NUM_METRICS:
        b_vals = [get_num(first_row_by_id[i], m) for i in common_ids]
        f_vals = [get_num(few_map[i], m) for i in common_ids]
        b_mean = mean(v for v in b_vals if v is not None)
        f_mean = mean(v for v in f_vals if v is not None)
        metric_rows[m] = {
            "batch": b_mean,
            "fewshot": f_mean,
            "delta_batch_minus_fewshot": b_mean - f_mean,
        }

    both = batch_only = few_only = neither = 0
    for i in common_ids:
        b = bool(first_row_by_id[i].get("exact_exec_match"))
        f = bool(few_map[i].get("exact_exec_match"))
        if b and f:
            both += 1
        elif b and not f:
            batch_only += 1
        elif f and not b:
            few_only += 1
        else:
            neither += 1

    return {
        "batch": "overall_combined",
        "question_set_source": id_source,
        "counts": {
            "batch_scope_ids_raw": len(scope_ids_raw),
            "batch_scope_ids": len(scope_ids),
            "common_ids": len(common_ids),
            "missing_in_fewshot": len(missing_in_few),
            "deduped_scope_ids": len(scope_ids_raw) - len(scope_ids),
        },
        "exact_head_to_head": {
            "both_true": both,
            "batch_only_true": batch_only,
            "fewshot_only_true": few_only,
            "both_false": neither,
        },
        "metrics": metric_rows,
        "sample_common_ids": common_ids[:10],
        "sample_id_sources": {i: first_batch_by_id.get(i) for i in common_ids[:10]},
    }


def print_report(result: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"Batch: {result['batch']}")
    print(f"ID source: {result['question_set_source']}")
    c = result["counts"]
    print(
        f"Scope IDs: {c['batch_scope_ids']} "
        f"(raw={c.get('batch_scope_ids_raw', c['batch_scope_ids'])}, deduped={c.get('deduped_scope_ids', 0)}) "
        f"| Common IDs: {c['common_ids']} | Missing in few-shot: {c['missing_in_fewshot']}"
    )
    h = result["exact_head_to_head"]
    print(
        "Exact head-to-head: "
        f"both={h['both_true']} batch_only={h['batch_only_true']} "
        f"fewshot_only={h['fewshot_only_true']} neither={h['both_false']}"
    )
    print("-" * 90)
    for key in BOOL_METRICS + NUM_METRICS:
        m = result["metrics"][key]
        print(
            f"{key:24s} "
            f"batch={m['batch']:.6f} "
            f"fewshot={m['fewshot']:.6f} "
            f"delta={m['delta_batch_minus_fewshot']:+.6f}"
        )


def main() -> None:
    args = parse_args()
    few_obj = load_eval(Path(args.fewshot_eval))
    few_map = to_item_map(few_obj["per_item"])

    all_results: List[Dict[str, Any]] = []
    batch_rows_by_name: Dict[str, Sequence[Dict[str, Any]]] = {}
    for p in args.batch_evals:
        path = Path(p)
        obj = load_eval(path)
        batch_rows_by_name[path.stem] = obj["per_item"]
        res = compare_one(
            batch_name=path.stem,
            batch_rows=obj["per_item"],
            few_map=few_map,
            id_source=args.id_source,
        )
        all_results.append(res)
        print_report(res)

    overall = compare_overall(
        batch_results=all_results,
        batch_rows_by_name=batch_rows_by_name,
        few_map=few_map,
        id_source=args.id_source,
    )
    print_report(overall)

    if args.output_json.strip():
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "fewshot_eval": str(Path(args.fewshot_eval)),
                    "batch_evals": [str(Path(x)) for x in args.batch_evals],
                    "id_source": args.id_source,
                    "results": all_results,
                    "overall": overall,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("=" * 90)
        print(f"Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
