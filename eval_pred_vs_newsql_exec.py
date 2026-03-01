#!/usr/bin/env python3
"""
Evaluate predicted SQL vs ground-truth SQL by executing both and comparing result sets.

Default fields assume your roundtrip output format:
- pred SQL key: pred_sql
- gold SQL key: new_sql

Default match mode is Hungarian (soft one-to-one row alignment on common columns).
Fallback to multiset mode is used automatically for very large row sets.

Metrics (per item):
- relaxed_em_unordered: exact row multiset equality after projection to common columns
- precision / recall / f1
- row_jaccard
- hard_overlap_rows: exact row overlaps after chosen matching
- soft_overlap_score: summed similarity score from matching

Example:
python3 eval_pred_vs_newsql_exec.py \
  --input_json data/empty_gt_fixed_v7_smoke.json \
  --db_path data/database.db \
  --output_json data/empty_gt_fixed_v7_smoke_exec_eval.json
"""

import argparse
import json
import os
import re
import sqlite3
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def extract_sql(text: str) -> str:
    """Best-effort SQL extraction from possible fenced/prose model output."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\bSELECT\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():].strip()

    # Keep first top-level SELECT block if repeated
    m2 = re.search(r"\n\s*SELECT\b", t, flags=re.IGNORECASE)
    if m2:
        t = t[:m2.start()].strip()

    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


def canonical_value(v: Any) -> Any:
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def tokenize_text(s: str) -> List[str]:
    s = normalize_text(s)
    return re.findall(r"[a-z0-9]+(?:[+/_-][a-z0-9]+)*", s)


def rouge1_f1(a: str, b: str) -> float:
    ta = tokenize_text(a)
    tb = tokenize_text(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum((ca & cb).values())
    p = overlap / len(ta)
    r = overlap / len(tb)
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def cell_similarity(a: Any, b: Any, text_metric: str = "rouge1_f1") -> float:
    a = canonical_value(a)
    b = canonical_value(b)

    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    # numeric exact equality
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 1.0 if float(a) == float(b) else 0.0

    if isinstance(a, str) or isinstance(b, str):
        sa = str(a)
        sb = str(b)
        if text_metric == "exact":
            return 1.0 if normalize_text(sa) == normalize_text(sb) else 0.0
        return rouge1_f1(sa, sb)

    return 1.0 if a == b else 0.0


def row_similarity(row_a: Tuple[Any, ...], row_b: Tuple[Any, ...], text_metric: str = "rouge1_f1") -> float:
    if len(row_a) != len(row_b):
        return 0.0
    if not row_a:
        return 1.0
    s = 0.0
    for va, vb in zip(row_a, row_b):
        s += cell_similarity(va, vb, text_metric=text_metric)
    return s / len(row_a)


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def hungarian_min_cost_square(cost: List[List[float]]) -> Tuple[List[int], float]:
    """Hungarian algorithm for square min-cost assignment. Returns assignment row->col."""
    n = len(cost)
    if n == 0:
        return [], 0.0
    for row in cost:
        if len(row) != n:
            raise ValueError("cost matrix must be square")

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1

    total = 0.0
    for i in range(n):
        j = assignment[i]
        if j >= 0:
            total += cost[i][j]
    return assignment, total


def multiset_metrics(pred_proj: List[Tuple[Any, ...]], gold_proj: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    cp = Counter(pred_proj)
    cg = Counter(gold_proj)

    overlap = sum((cp & cg).values())
    union = sum((cp | cg).values())

    p = safe_div(overlap, len(pred_proj)) if pred_proj else (1.0 if not gold_proj else 0.0)
    r = safe_div(overlap, len(gold_proj)) if gold_proj else (1.0 if not pred_proj else 0.0)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    jacc = safe_div(overlap, union) if union > 0 else 1.0

    return {
        "hard_overlap_rows": int(overlap),
        "soft_overlap_score": float(overlap),
        "precision": p,
        "recall": r,
        "f1": f1,
        "row_jaccard": jacc,
        "relaxed_em_unordered": (cp == cg),
    }


def hungarian_metrics(
    pred_proj: List[Tuple[Any, ...]],
    gold_proj: List[Tuple[Any, ...]],
    text_metric: str = "rouge1_f1",
) -> Dict[str, Any]:
    m = len(pred_proj)
    n = len(gold_proj)
    k = max(m, n)

    if k == 0:
        return {
            "hard_overlap_rows": 0,
            "soft_overlap_score": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "row_jaccard": 1.0,
            "relaxed_em_unordered": True,
        }

    sim = [[0.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = row_similarity(pred_proj[i], gold_proj[j], text_metric=text_metric)

    cost = [[1.0 - sim[i][j] for j in range(k)] for i in range(k)]
    assignment, _ = hungarian_min_cost_square(cost)

    soft_overlap = 0.0
    hard_overlap = 0
    for i in range(m):
        j = assignment[i]
        if 0 <= j < n:
            s = sim[i][j]
            soft_overlap += s
            if s >= 1.0 - 1e-12:
                hard_overlap += 1

    p = safe_div(soft_overlap, m) if m > 0 else (1.0 if n == 0 else 0.0)
    r = safe_div(soft_overlap, n) if n > 0 else (1.0 if m == 0 else 0.0)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    denom = (m + n - soft_overlap)
    jacc = safe_div(soft_overlap, denom) if denom > 0 else 1.0

    # Keep EM definition exact: same number of rows and all assigned rows exact matches.
    relaxed_em = (m == n and hard_overlap == m)

    return {
        "hard_overlap_rows": int(hard_overlap),
        "soft_overlap_score": float(soft_overlap),
        "precision": p,
        "recall": r,
        "f1": f1,
        "row_jaccard": jacc,
        "relaxed_em_unordered": relaxed_em,
    }


def evaluate_pair(
    conn: sqlite3.Connection,
    pred_sql: str,
    gold_sql: str,
    max_rows: int,
    match_mode: str,
    hungarian_max_rows: int,
    cell_text_metric: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred_exec_ok": False,
        "gold_exec_ok": False,
        "eval_ok": False,
        "error": None,
        "pred_row_count": 0,
        "gold_row_count": 0,
        "pred_col_count": 0,
        "gold_col_count": 0,
        "common_cols_count": 0,
        "common_columns": [],
        "overlap_rows": 0,
        "hard_overlap_rows": 0,
        "soft_overlap_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "row_jaccard": 0.0,
        "relaxed_em_unordered": False,
        "strict_projection_match": False,
        "match_mode_requested": match_mode,
        "match_mode_used": match_mode,
        "match_mode_fallback_reason": None,
        "cell_text_metric_requested": cell_text_metric,
        "cell_text_metric_used": cell_text_metric if match_mode == "hungarian" else None,
    }

    try:
        pred_cols, pred_rows = execute_sql_fetch(conn, pred_sql, max_rows=max_rows)
        out["pred_exec_ok"] = True
    except Exception as e:
        out["error"] = f"PRED_EXEC_ERROR: {e}"
        return out

    try:
        gold_cols, gold_rows = execute_sql_fetch(conn, gold_sql, max_rows=max_rows)
        out["gold_exec_ok"] = True
    except Exception as e:
        out["error"] = f"GOLD_EXEC_ERROR: {e}"
        return out

    out["pred_col_count"] = len(pred_cols)
    out["gold_col_count"] = len(gold_cols)
    out["pred_row_count"] = len(pred_rows)
    out["gold_row_count"] = len(gold_rows)
    out["strict_projection_match"] = pred_cols == gold_cols

    common_cols = [c for c in pred_cols if c in gold_cols]
    out["common_cols_count"] = len(common_cols)
    out["common_columns"] = common_cols

    if not common_cols:
        out["error"] = "NO_COMMON_COLS"
        return out

    pred_idx = [pred_cols.index(c) for c in common_cols]
    gold_idx = [gold_cols.index(c) for c in common_cols]

    pred_proj = [tuple(canonical_value(r[i]) for i in pred_idx) for r in pred_rows]
    gold_proj = [tuple(canonical_value(r[i]) for i in gold_idx) for r in gold_rows]

    mode_used = match_mode
    if match_mode == "hungarian" and max(len(pred_proj), len(gold_proj)) > max(1, hungarian_max_rows):
        mode_used = "multiset"
        out["match_mode_used"] = mode_used
        out["match_mode_fallback_reason"] = (
            f"result rows exceed --hungarian_max_rows={hungarian_max_rows}"
        )

    if mode_used == "hungarian":
        out["cell_text_metric_used"] = cell_text_metric
        m = hungarian_metrics(pred_proj, gold_proj, text_metric=cell_text_metric)
    else:
        out["cell_text_metric_used"] = None
        m = multiset_metrics(pred_proj, gold_proj)

    out["match_mode_used"] = mode_used
    out["hard_overlap_rows"] = int(m["hard_overlap_rows"])
    out["soft_overlap_score"] = float(m["soft_overlap_score"])
    out["overlap_rows"] = int(m["hard_overlap_rows"])
    out["precision"] = float(m["precision"])
    out["recall"] = float(m["recall"])
    out["f1"] = float(m["f1"])
    out["row_jaccard"] = float(m["row_jaccard"])
    out["relaxed_em_unordered"] = bool(m["relaxed_em_unordered"])
    out["eval_ok"] = True
    return out


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="JSON list with pred/gold SQL fields")
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--output_json", default="", help="Optional output JSON with per-item eval appended")

    ap.add_argument("--pred_sql_key", default="pred_sql")
    ap.add_argument("--gold_sql_key", default="new_sql")
    ap.add_argument("--item_id_key", default="item_id")

    ap.add_argument("--max_rows", type=int, default=10000, help="Row cap per query during eval")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--extract_sql", type=int, default=0, help="1=run best-effort SQL extraction on fields")
    ap.add_argument("--print_worst", type=int, default=10)

    ap.add_argument(
        "--match_mode",
        choices=["hungarian", "multiset"],
        default="hungarian",
        help="Row matching mode (default: hungarian).",
    )
    ap.add_argument(
        "--hungarian_max_rows",
        type=int,
        default=400,
        help="If max(pred_rows, gold_rows) exceeds this, fallback to multiset mode.",
    )
    ap.add_argument(
        "--cell_text_metric",
        choices=["rouge1_f1", "exact"],
        default="rouge1_f1",
        help="Cell-level text similarity metric for Hungarian mode (default: rouge1_f1).",
    )

    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("input_json must contain a JSON list")

    if args.limit is not None and args.limit > -1:
        rows = rows[args.start: args.start + args.limit]
    else:
        rows = rows[args.start:]

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

    for row in rows:
        pred_sql = row.get(args.pred_sql_key, "")
        gold_sql = row.get(args.gold_sql_key, "")

        if args.extract_sql:
            pred_sql = extract_sql(pred_sql)
            gold_sql = extract_sql(gold_sql)

        eval_row = evaluate_pair(
            conn,
            pred_sql,
            gold_sql,
            max_rows=args.max_rows,
            match_mode=args.match_mode,
            hungarian_max_rows=args.hungarian_max_rows,
            cell_text_metric=args.cell_text_metric,
        )
        out_row = dict(row)
        out_row["exec_eval"] = eval_row
        enriched.append(out_row)

        if eval_row["eval_ok"]:
            ok += 1
            precisions.append(float(eval_row["precision"]))
            recalls.append(float(eval_row["recall"]))
            f1s.append(float(eval_row["f1"]))
            jaccs.append(float(eval_row["row_jaccard"]))
            soft_scores.append(float(eval_row["soft_overlap_score"]))
            if bool(eval_row["relaxed_em_unordered"]):
                rem_hits += 1
            if eval_row.get("match_mode_used") != eval_row.get("match_mode_requested"):
                fallback_cnt += 1
        else:
            errors += 1
            if eval_row.get("error") == "NO_COMMON_COLS":
                no_common_cols += 1

    conn.close()

    total = len(rows)
    relaxed_em = (rem_hits / ok) if ok else 0.0

    print("\n================ EXEC EVAL SUMMARY ================")
    print(f"Input:                         {args.input_json}")
    print(f"DB:                            {args.db_path}")
    print(f"Match mode requested:          {args.match_mode}")
    print(f"Cell text metric:              {args.cell_text_metric if args.match_mode == "hungarian" else "n/a"}")
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

    # Print worst rows by F1 among successful evals
    worst: List[Tuple[Any, float, float, float, Optional[str], int, int, int, str]] = []
    for r in enriched:
        e = r.get("exec_eval", {})
        if not e.get("eval_ok"):
            continue
        worst.append((
            r.get(args.item_id_key),
            float(e.get("f1", 0.0)),
            float(e.get("precision", 0.0)),
            float(e.get("recall", 0.0)),
            e.get("error"),
            int(e.get("pred_row_count", 0)),
            int(e.get("gold_row_count", 0)),
            int(e.get("common_cols_count", 0)),
            str(e.get("match_mode_used", "")),
        ))
    worst.sort(key=lambda x: x[1])
    k = max(0, int(args.print_worst))
    if k > 0 and worst:
        print(f"Worst {min(k, len(worst))} by F1:")
        for item_id, f1, p, r, err, pr, gr, cc, mode_used in worst[:k]:
            print(
                f"  - {item_id}: f1={f1:.4f}, p={p:.4f}, r={r:.4f}, "
                f"pred_rows={pr}, gold_rows={gr}, common_cols={cc}, mode={mode_used}, err={err}"
            )

    if args.output_json:
        out_dir = os.path.dirname(args.output_json) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        print(f"Output JSON:                   {args.output_json}")

    print("====================================================\n")


if __name__ == "__main__":
    main()
