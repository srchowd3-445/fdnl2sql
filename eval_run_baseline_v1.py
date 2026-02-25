#!/usr/bin/env python3
"""Evaluate run_baselines outputs against GT SQL and execution results."""

import argparse
import json
import math
import os
import re
import sqlite3
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

try:
    import sqlglot
except Exception:
    sqlglot = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None

try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None


def load_data(path: str, fmt: str = "auto") -> List[Dict[str, Any]]:
    if fmt == "auto":
        with open(path, "r", encoding="utf-8") as f:
            first = ""
            for line in f:
                t = line.strip()
                if t:
                    first = t
                    break
        fmt = "json" if first.startswith("[") else "jsonl"

    if fmt == "json":
        obj = json.load(open(path, "r", encoding="utf-8"))
        return [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


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


def canonical_value(v: Any) -> Any:
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)


def normalize_num(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # reduce tiny float noise
        return round(v, 6)
    return v


def canonical_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    out = []
    for x in row:
        x = canonical_value(x)
        if isinstance(x, (int, float, bool)):
            x = normalize_num(x)
        out.append(x)
    return tuple(out)


def multiset_rows(rows: List[Tuple[Any, ...]]) -> Counter:
    return Counter(canonical_row(r) for r in rows)


def cell_similarity(a: Any, b: Any) -> float:
    a = canonical_value(a)
    b = canonical_value(b)
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af = float(a)
        bf = float(b)
        if math.isfinite(af) and math.isfinite(bf) and math.isclose(af, bf, rel_tol=1e-6, abs_tol=1e-6):
            return 1.0
        return 0.0
    return rouge1_f1(str(a), str(b))


def row_similarity(row_a: Tuple[Any, ...], row_b: Tuple[Any, ...]) -> float:
    if len(row_a) != len(row_b):
        return 0.0
    if not row_a:
        return 1.0
    s = 0.0
    for va, vb in zip(row_a, row_b):
        s += cell_similarity(va, vb)
    return s / len(row_a)


def hungarian_min_cost_square(cost: List[List[float]]) -> Tuple[List[int], float]:
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


def is_safe_readonly_sql(sql: str) -> Tuple[bool, str]:
    s = (sql or "").strip()
    if not s:
        return False, "empty"
    # Block multiple statements
    if ";" in s.rstrip(";"):
        return False, "multiple_statements"
    head = s.lstrip().split(None, 1)[0].lower()
    if head not in {"select", "with"}:
        return False, f"not_select_or_with:{head}"
    banned = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "replace",
    }
    toks = set(re.findall(r"[a-z_]+", s.lower()))
    if toks & banned:
        return False, f"contains_banned:{sorted(toks & banned)}"
    return True, ""


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = (sql or "").strip().rstrip(";")
    ok, why = is_safe_readonly_sql(sql0)
    if not ok:
        raise ValueError(f"UNSAFE_SQL:{why}")
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


# def ast_similarity_sqlglot(pred_sql: str, gt_sql: str) -> Tuple[Optional[float], Optional[str]]:
#     if not pred_sql or not gt_sql:
#         return None, "missing_sql"
#     if sqlglot is None:
#         return None, "sqlglot_not_installed"
#     try:
#         p = sqlglot.parse_one(pred_sql, read="sqlite")
#         g = sqlglot.parse_one(gt_sql, read="sqlite")
#     except Exception as e:
#         return None, f"parse_error:{e}"

#     def flatten_ast(expr: Any) -> List[str]:
#         toks: List[str] = []
#         for node in expr.walk():
#             cls = node.__class__.__name__
#             toks.append(f"node:{cls}")
#             for k, v in node.args.items():
#                 if isinstance(v, (str, int, float)):
#                     toks.append(f"{k}:{str(v).lower()}")
#         return toks

#     tp = flatten_ast(p)
#     tg = flatten_ast(g)
#     cp = Counter(tp)
#     cg = Counter(tg)
#     overlap = sum((cp & cg).values())
#     return safe_div(2 * overlap, len(tp) + len(tg)), None


def ast_similarity_sqlglot(pred_sql: str, gt_sql: str) -> Tuple[Optional[float], Optional[str]]:
    if not pred_sql or not gt_sql:
        return None, "missing_sql"
    if sqlglot is None:
        return None, "sqlglot_not_installed"
    try:
        p = sqlglot.parse_one(pred_sql, read="sqlite")
        g = sqlglot.parse_one(gt_sql, read="sqlite")
    except Exception as e:
        return None, f"parse_error:{e}"

    def flatten_ast(expr: Any) -> List[str]:
        toks: List[str] = []
        if expr is None:
            return toks
        if isinstance(expr, list):
            for x in expr:
                toks.extend(flatten_ast(x))
            return toks

        for node in expr.walk():
            cls = node._class_._name_
            toks.append(f"node:{cls}")
            for k, v in node.args.items():
                if isinstance(v, (str, int, float)):
                    toks.append(f"{k}:{str(v).lower()}")
        return toks

    def token_f1(a: List[str], b: List[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        ca = Counter(a)
        cb = Counter(b)
        overlap = sum((ca & cb).values())
        return safe_div(2 * overlap, len(a) + len(b))

    # Clause-level AST similarity with explicit weights.
    # Requested weights: SELECT=0.5, FROM=0.1, WHERE=0.4
    select_p = flatten_ast(p.args.get("expressions"))
    select_g = flatten_ast(g.args.get("expressions"))

    from_p = flatten_ast(p.args.get("from"))
    from_g = flatten_ast(g.args.get("from"))

    where_p = flatten_ast(p.args.get("where"))
    where_g = flatten_ast(g.args.get("where"))

    w_select = 0.5
    w_from = 0.0
    w_where = 0.5

    s_select = token_f1(select_p, select_g)
    s_from = token_f1(from_p, from_g)
    s_where = token_f1(where_p, where_g)

    weighted = (w_select * s_select) + (w_from * s_from) + (w_where * s_where)
    return weighted, None


def col_name_similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if na == nb:
        return 1.0
    return rouge1_f1(na, nb)


def column_value_similarity(
    pred_rows: List[Tuple[Any, ...]],
    gt_rows: List[Tuple[Any, ...]],
    i: int,
    j: int,
    sample_n: int = 50,
) -> float:
    if not pred_rows or not gt_rows:
        return 0.0
    n = min(sample_n, len(pred_rows), len(gt_rows))
    if n <= 0:
        return 0.0
    s = 0.0
    for t in range(n):
        try:
            s += cell_similarity(pred_rows[t][i], gt_rows[t][j])
        except Exception:
            continue
    return s / n if n else 0.0


def align_columns(
    pred_cols: List[str],
    gt_cols: List[str],
    pred_rows: List[Tuple[Any, ...]],
    gt_rows: List[Tuple[Any, ...]],
) -> Tuple[List[Tuple[int, int, float]], float]:
    m = len(pred_cols)
    n = len(gt_cols)
    k = max(m, n)
    if k == 0:
        return [], 1.0

    sim = [[0.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            name_s = col_name_similarity(pred_cols[i], gt_cols[j])
            val_s = column_value_similarity(pred_rows, gt_rows, i, j, sample_n=50)
            sim[i][j] = 0.7 * name_s + 0.3 * val_s

    cost = [[1.0 - sim[i][j] for j in range(k)] for i in range(k)]
    assignment, _ = hungarian_min_cost_square(cost)

    pairs: List[Tuple[int, int, float]] = []
    sim_sum = 0.0
    for i in range(m):
        j = assignment[i]
        if 0 <= j < n:
            s = sim[i][j]
            pairs.append((i, j, s))
            sim_sum += s
    denom = max(1, min(m, n))
    return pairs, sim_sum / denom


def project_rows(rows: List[Tuple[Any, ...]], indices: List[int]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    for r in rows:
        out.append(tuple(canonical_value(r[i]) for i in indices))
    return out


def row_to_text(row: Tuple[Any, ...]) -> str:
    return " | ".join("" if v is None else str(v) for v in row)


def rouge_l_f1_avg(pred_texts: List[str], ref_texts: List[str]) -> Optional[float]:
    if not pred_texts:
        return None
    if rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    vals = []
    for p, r in zip(pred_texts, ref_texts):
        vals.append(float(scorer.score(r, p)["rougeL"].fmeasure))
    return mean(vals) if vals else None


def chrf_corpus(pred_texts: List[str], ref_texts: List[str]) -> Optional[float]:
    if not pred_texts:
        return None
    if sacrebleu is None:
        return None
    score = sacrebleu.corpus_chrf(pred_texts, [ref_texts]).score
    return float(score)


def bertscore_f1_avg(pred_texts: List[str], ref_texts: List[str], enabled: bool) -> Optional[float]:
    if not enabled or not pred_texts:
        return None
    if bert_score_fn is None:
        return None
    _, _, f1 = bert_score_fn(pred_texts, ref_texts, lang="en", verbose=False)
    return float(f1.mean().item())


def has_limit_without_order(sql: str) -> bool:
    s = normalize_text(sql)
    s2 = f" {s} "
    return (" limit " in s2) and (" order by " not in s2)


def normalize_sql_table_refs(sql: str) -> str:
    s = sql or ""
    # vLLM runs produced table name as `clinical.trials` (dot). SQLite table is `clinical_trials`.
    s = re.sub(r"\bclinical\.trials\b", "clinical_trials", s, flags=re.IGNORECASE)
    s = re.sub(r'"clinical"\."trials"', "clinical_trials", s, flags=re.IGNORECASE)
    return s


def evaluate_execution(
    conn: sqlite3.Connection,
    pred_sql: str,
    gt_sql: str,
    *,
    max_rows: int,
    compute_bertscore: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred_exec_ok": False,
        "gt_exec_ok": False,
        "error": None,
        "pred_row_count": 0,
        "gt_row_count": 0,
        "pred_col_count": 0,
        "gt_col_count": 0,
        "column_alignment_score": 0.0,
        "matched_row_pairs": 0,
        "hard_overlap_rows": 0,
        "soft_overlap_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "row_jaccard": 0.0,
        "normalization_factor": 0.0,
        "chrf": None,
        "rouge_l_f1": None,
        "bertscore_f1": None,
        "exact_exec_match": False,
        "pred_limit_no_order": False,
        "gt_limit_no_order": False,
    }

    pred_sql = normalize_sql_table_refs(pred_sql)
    gt_sql = normalize_sql_table_refs(gt_sql)

    out["pred_limit_no_order"] = has_limit_without_order(pred_sql)
    out["gt_limit_no_order"] = has_limit_without_order(gt_sql)

    try:
        pred_cols, pred_rows = execute_sql_fetch(conn, pred_sql, max_rows=max_rows)
        out["pred_exec_ok"] = True
    except Exception as e:
        out["error"] = f"PRED_EXEC_ERROR: {e}"
        return out

    try:
        gt_cols, gt_rows = execute_sql_fetch(conn, gt_sql, max_rows=max_rows)
        out["gt_exec_ok"] = True
    except Exception as e:
        out["error"] = f"GT_EXEC_ERROR: {e}"
        return out

    out["pred_col_count"] = len(pred_cols)
    out["gt_col_count"] = len(gt_cols)
    out["pred_row_count"] = len(pred_rows)
    out["gt_row_count"] = len(gt_rows)

    if not pred_cols or not gt_cols:
        out["error"] = "EMPTY_RESULT_COLUMNS"
        return out

    col_pairs, col_score = align_columns(pred_cols, gt_cols, pred_rows, gt_rows)
    out["column_alignment_score"] = float(col_score)
    if not col_pairs:
        out["error"] = "NO_COLUMN_ALIGNMENT"
        return out

    pred_idx = [i for i, _, _ in col_pairs]
    gt_idx = [j for _, j, _ in col_pairs]
    pred_proj = project_rows(pred_rows, pred_idx)
    gt_proj = project_rows(gt_rows, gt_idx)

    # Standard strict exec-match metric (multiset equality after canonicalization)
    out["exact_exec_match"] = (multiset_rows(pred_proj) == multiset_rows(gt_proj))

    m = len(pred_proj)
    n = len(gt_proj)
    k = max(m, n)
    if k == 0:
        out.update(
            {
                "matched_row_pairs": 0,
                "hard_overlap_rows": 0,
                "soft_overlap_score": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "row_jaccard": 1.0,
                "normalization_factor": 1.0,
                "chrf": 100.0,
                "rouge_l_f1": 1.0,
                "bertscore_f1": 1.0 if compute_bertscore and bert_score_fn is not None else None,
                "exact_exec_match": True,
            }
        )
        return out

    sim = [[0.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = row_similarity(pred_proj[i], gt_proj[j])
    cost = [[1.0 - sim[i][j] for j in range(k)] for i in range(k)]
    assignment, _ = hungarian_min_cost_square(cost)

    hard_overlap = 0
    soft_overlap = 0.0
    matched_pred_texts: List[str] = []
    matched_gt_texts: List[str] = []
    for i in range(m):
        j = assignment[i]
        if 0 <= j < n:
            s = sim[i][j]
            soft_overlap += s
            if s >= 1.0 - 1e-12:
                hard_overlap += 1
            matched_pred_texts.append(row_to_text(pred_proj[i]))
            matched_gt_texts.append(row_to_text(gt_proj[j]))

    p = safe_div(soft_overlap, m) if m > 0 else (1.0 if n == 0 else 0.0)
    r = safe_div(soft_overlap, n) if n > 0 else (1.0 if m == 0 else 0.0)
    f1 = safe_div(2.0 * p * r, p + r) if (p + r) > 0 else 0.0
    denom = (m + n - soft_overlap)
    jacc = safe_div(soft_overlap, denom) if denom > 0 else 1.0

    out["matched_row_pairs"] = len(matched_pred_texts)
    out["hard_overlap_rows"] = int(hard_overlap)
    out["soft_overlap_score"] = float(soft_overlap)
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)
    out["row_jaccard"] = float(jacc)
    out["normalization_factor"] = safe_div(float(hard_overlap), float(max(1, m)))
    out["chrf"] = chrf_corpus(matched_pred_texts, matched_gt_texts)
    out["rouge_l_f1"] = rouge_l_f1_avg(matched_pred_texts, matched_gt_texts)
    out["bertscore_f1"] = bertscore_f1_avg(matched_pred_texts, matched_gt_texts, enabled=compute_bertscore)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate run_baselines results.")
    ap.add_argument("--pred_path", required=True, help="Output from run_baselines.py (json/jsonl)")
    ap.add_argument("--gt_path", default="data/natural_question_1500.json")
    ap.add_argument("--db_path", default="data/database.db")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--pred_format", choices=["auto", "json", "jsonl"], default="auto")
    ap.add_argument("--gt_format", choices=["auto", "json", "jsonl"], default="auto")
    ap.add_argument("--id_key", default="item_id")
    ap.add_argument("--pred_sql_key", default="pred_sql")
    ap.add_argument("--gt_sql_key", default="gt_sql")
    ap.add_argument("--question_key", default="natural_question")
    ap.add_argument("--confidence_key", default="confidence_overall")
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--compute_bertscore", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pred_rows = load_data(args.pred_path, args.pred_format)
    gt_rows = load_data(args.gt_path, args.gt_format)

    pred_map = {str(r.get(args.id_key)): r for r in pred_rows if r.get(args.id_key) is not None}
    gt_map = {str(r.get(args.id_key)): r for r in gt_rows if r.get(args.id_key) is not None}

    conn = sqlite3.connect(args.db_path)

    per_item: List[Dict[str, Any]] = []
    for item_id, gt in gt_map.items():
        pred = pred_map.get(item_id, {})
        join_method = "item_id" if pred else "missing"

        pred_sql = (pred.get(args.pred_sql_key) or "").strip()
        gt_sql = (gt.get(args.gt_sql_key) or "").strip()

        ast_sim, ast_err = ast_similarity_sqlglot(pred_sql, gt_sql)
        exec_eval = (
            evaluate_execution(
                conn=conn,
                pred_sql=pred_sql,
                gt_sql=gt_sql,
                max_rows=args.max_rows,
                compute_bertscore=bool(args.compute_bertscore),
            )
            if pred_sql and gt_sql
            else {
                "pred_exec_ok": False,
                "gt_exec_ok": False,
                "error": "MISSING_PRED_OR_GT_SQL",
                "pred_row_count": 0,
                "gt_row_count": 0,
                "pred_col_count": 0,
                "gt_col_count": 0,
                "column_alignment_score": 0.0,
                "matched_row_pairs": 0,
                "hard_overlap_rows": 0,
                "soft_overlap_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "row_jaccard": 0.0,
                "normalization_factor": 0.0,
                "chrf": None,
                "rouge_l_f1": None,
                "bertscore_f1": None,
                "exact_exec_match": False,
                "pred_limit_no_order": has_limit_without_order(pred_sql),
                "gt_limit_no_order": has_limit_without_order(gt_sql),
            }
        )

        per_item.append(
            {
                "item_id": item_id,
                "question": gt.get(args.question_key) or gt.get("natural_question") or pred.get("question_used"),
                "pred_sql": pred_sql,
                "gt_sql": gt_sql,
                "pred_confidence": pred.get(args.confidence_key),
                "pred_error": pred.get("error"),
                "join_method": join_method,
                "sql_ast_similarity": ast_sim,
                "sql_ast_error": ast_err,
                **exec_eval,
            }
        )

    conn.close()

    def avg_non_null(key: str) -> Optional[float]:
        vals = [x.get(key) for x in per_item if isinstance(x.get(key), (int, float))]
        return float(mean(vals)) if vals else None

    exec_ready = sum(1 for x in per_item if x.get("pred_exec_ok") and x.get("gt_exec_ok"))
    exact_match = sum(1 for x in per_item if x.get("exact_exec_match"))

    summary = {
        "total_gt_items": len(gt_map),
        "total_pred_items": len(pred_map),
        "evaluated_items": len(per_item),
        "pred_with_sql": sum(1 for x in per_item if (x.get("pred_sql") or "").strip()),
        "pred_exec_ok": sum(1 for x in per_item if x.get("pred_exec_ok")),
        "gt_exec_ok": sum(1 for x in per_item if x.get("gt_exec_ok")),
        "exec_eval_ok": sum(1 for x in per_item if x.get("pred_exec_ok") and x.get("gt_exec_ok") and not x.get("error")),
        "exec_exact_match": exact_match,
        "exec_exact_match_rate": safe_div(float(exact_match), float(max(1, exec_ready))),
        "avg_sql_ast_similarity": avg_non_null("sql_ast_similarity"),
        "avg_precision": avg_non_null("precision"),
        "avg_recall": avg_non_null("recall"),
        "avg_f1": avg_non_null("f1"),
        "avg_row_jaccard": avg_non_null("row_jaccard"),
        "avg_normalization_factor": avg_non_null("normalization_factor"),
        "avg_column_alignment_score": avg_non_null("column_alignment_score"),
        "avg_chrf": avg_non_null("chrf"),
        "avg_rouge_l_f1": avg_non_null("rouge_l_f1"),
        "avg_bertscore_f1": avg_non_null("bertscore_f1"),
        "avg_pred_confidence": avg_non_null("pred_confidence"),
        "pred_limit_no_order": sum(1 for x in per_item if x.get("pred_limit_no_order")),
        "gt_limit_no_order": sum(1 for x in per_item if x.get("gt_limit_no_order")),
    }

    out = {
        "meta": {
            "pred_path": args.pred_path,
            "gt_path": args.gt_path,
            "db_path": args.db_path,
            "max_rows": args.max_rows,
            "compute_bertscore": bool(args.compute_bertscore),
        },
        "summary": summary,
        "per_item": per_item,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", args.output_json)
    print("Summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()