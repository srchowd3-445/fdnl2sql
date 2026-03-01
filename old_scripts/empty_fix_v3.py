"""
empty_fix_v3.py

Fixes v2 SQL errors by using PARAMETERIZED queries when sampling DISTINCT candidates.
This prevents broken SQL caused by unescaped quotes / special characters in conditions.

Goal:
- Read dataset.jsonl with gt_sql
- Identify gt_sql queries that return empty
- Repair each empty query by REPLACING exactly one WHERE condition with a value that yields rows.
- Adds variability:
  - preferred columns first
  - deprioritize Cancer type / Name of ICI
  - relaxed context subset for candidate sampling
- Writes output JSON: data/empty_gt_replaced_v3.json
"""

import json
import re
import sqlite3
import random
from difflib import get_close_matches

# ===============================
# CONFIG
# ===============================
db_path = "data/database.db"
input_jsonl = "data/dataset.jsonl"
table_name = "clinical_trials"
output_json = "data/empty_gt_replaced_v3.json"

MAX_CANDIDATES_PER_COL = 25
PREFER_CLOSE_MATCH = True
TRY_ALL_CONDITIONS = True

# -------------------------------
# VARIABILITY CONFIG (V3)
# -------------------------------
RANDOM_SEED = 7
random.seed(RANDOM_SEED)

PREFERRED_COLS_FIRST = [
    "Trial phase",
    "Control arm",
    "Control regimen",
    "Type of control",
    "Monotherapy/combination",
    "Type of combination",
    "Lines of treatment",
    "Clinical setting",
    "Clincal setting in relation to surgery",
    "Primary endpoint",
    "Secondary endpoint",
    "Year",
    "Number of arms",
    "Total sample size",
]

DEPRIORITIZED_COLS = [
    "Cancer type",
    "Name of ICI",
    "Class of ICI",
    "Trial name",
    "Treatment regimen",
]

CANDIDATE_CONTEXT_KEEP_FRAC = 0.6


# ===============================
# SQL helpers
# ===============================
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def get_where_clause(sql: str):
    sql0 = strip_trailing_semicolon(sql)
    m = re.search(r"\bWHERE\b", sql0, flags=re.IGNORECASE)
    if not m:
        return sql0, None, ""

    prefix = sql0[:m.start()].rstrip()
    after_where = sql0[m.end():].strip()

    m2 = re.search(r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT)\b", after_where, flags=re.IGNORECASE)
    if m2:
        where_clause = after_where[:m2.start()].strip()
        suffix = after_where[m2.start():].strip()
    else:
        where_clause = after_where.strip()
        suffix = ""

    return prefix, where_clause, suffix


def split_conditions(where_clause: str):
    if not where_clause:
        return []
    return [p.strip() for p in re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE) if p.strip()]


def rebuild_sql(prefix: str, conditions, suffix: str):
    if not conditions:
        return prefix + (" " + suffix if suffix else "")
    return prefix + " WHERE " + " AND ".join(conditions) + (" " + suffix if suffix else "")


def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_condition(cond: str):
    """
    Parses simple predicates:
      "Col" = 'val'
      "Col" = 2018
      "Col" >= 4, >, <=, <

    Returns dict or None if unsupported.
    """
    m = re.match(r'^\s*"([^"]+)"\s*(=|>=|<=|>|<)\s*(.+?)\s*$', cond)
    if not m:
        return None

    col, op, raw_val = m.group(1), m.group(2), m.group(3).strip()

    # string literal?
    if re.match(r"^'.*'$", raw_val):
        sval = raw_val[1:-1]
        return {"col": col, "op": op, "type": "text", "val": sval, "text": cond}

    # numeric?
    try:
        nval = float(raw_val)
        return {"col": col, "op": op, "type": "num", "val": nval, "text": cond}
    except Exception:
        # unsupported complex token (e.g., bare words). Skip.
        return None


def execute_rowcount(conn: sqlite3.Connection, sql: str) -> int:
    """
    Return 1 if query has at least one row else 0.
    Uses LIMIT 1 wrapper for speed.
    """
    sql0 = strip_trailing_semicolon(sql)
    try:
        cur = conn.execute(f"SELECT 1 FROM ({sql0}) AS subq LIMIT 1;")
        return 1 if cur.fetchone() is not None else 0
    except Exception:
        cur = conn.execute(sql0)
        return 1 if cur.fetchone() is not None else 0


# ===============================
# Candidate ordering
# ===============================
def candidate_order(original_value, candidates):
    if not candidates:
        return []

    candidates_str = [str(c) for c in candidates]
    if PREFER_CLOSE_MATCH and original_value is not None:
        close = get_close_matches(
            str(original_value),
            candidates_str,
            n=min(len(candidates_str), MAX_CANDIDATES_PER_COL),
            cutoff=0.05
        )
        close_set = set(close)
        close_vals = [candidates[i] for i, s in enumerate(candidates_str) if s in close_set]
        rest_vals = [candidates[i] for i, s in enumerate(candidates_str) if s not in close_set]
        return close_vals + rest_vals
    return candidates


# ===============================
# SAFE DISTINCT candidate sampling (V3)
# ===============================
def build_param_where(conditions_parsed):
    """
    Build a parameterized WHERE clause from parsed conditions.
    Returns: (sql_fragment, params_list)
    """
    if not conditions_parsed:
        return "", []

    parts = []
    params = []
    for p in conditions_parsed:
        colq = safe_ident(p["col"])
        op = p["op"]
        parts.append(f"{colq} {op} ?")
        params.append(p["val"])
    return " AND ".join(parts), params


def distinct_values_param(conn, table, col, parsed_context_conditions, limit=5000):
    """
    Get DISTINCT values for col using a parameterized WHERE context.
    This avoids SQL syntax errors from unescaped strings.
    """
    colq = safe_ident(col)
    tblq = safe_ident(table)

    where_sql, params = build_param_where(parsed_context_conditions)

    sql = f"SELECT DISTINCT {colq} FROM {tblq} WHERE {colq} IS NOT NULL"
    if where_sql:
        sql += f" AND ({where_sql})"
    sql += f" LIMIT {int(limit)};"

    cur = conn.execute(sql, params)
    return [r[0] for r in cur.fetchall() if r[0] is not None]


# ===============================
# VARIABILITY HELPERS
# ===============================
def choose_context_subset(parsed_rest_conditions, keep_frac=0.6):
    """
    Choose a random subset of parsed conditions for candidate sampling.
    """
    if not parsed_rest_conditions:
        return []
    k = max(0, int(round(len(parsed_rest_conditions) * keep_frac)))
    if k >= len(parsed_rest_conditions):
        return parsed_rest_conditions[:]
    if k == 0:
        return []
    return random.sample(parsed_rest_conditions, k)


def order_conditions_for_try(conditions):
    """
    Order which condition to try replacing first.
    """
    parsed = []
    for idx, cond in enumerate(conditions):
        p = parse_condition(cond)
        parsed.append((idx, cond, p))

    def col_rank(p):
        if p is None:
            return 50
        col = p["col"]
        if col in PREFERRED_COLS_FIRST:
            return PREFERRED_COLS_FIRST.index(col)
        if col in DEPRIORITIZED_COLS:
            return 100 + DEPRIORITIZED_COLS.index(col)
        return 40

    parsed.sort(key=lambda x: col_rank(x[2]))

    grouped = {}
    for t in parsed:
        r = col_rank(t[2])
        grouped.setdefault(r, []).append(t)

    final = []
    for r in sorted(grouped.keys()):
        group = grouped[r]
        random.shuffle(group)
        final.extend(group)
    return final


# ===============================
# Replacement logic (V3)
# ===============================
def replace_one_condition_to_get_rows(conn, prefix, conditions, suffix):
    """
    Replace exactly one condition with an alternate value so the query returns rows.
    Uses parameterized DISTINCT sampling to avoid syntax errors.
    """
    sql0 = rebuild_sql(prefix, conditions, suffix)
    if execute_rowcount(conn, sql0) > 0:
        return sql0 + ";", {"replaced": None}

    ordered_conditions = order_conditions_for_try(conditions)

    for original_index, cond, p_target in ordered_conditions:
        if not p_target:
            continue

        target_col = p_target["col"]
        target_op = p_target["op"]

        # Parse all other conditions (rest), dropping those we can't parse safely
        rest_conditions_raw = [c for j, c in enumerate(conditions) if j != original_index]
        rest_parsed_all = [parse_condition(c) for c in rest_conditions_raw]
        rest_parsed_all = [p for p in rest_parsed_all if p is not None]

        # choose subset for candidate context
        context_parsed = choose_context_subset(rest_parsed_all, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

        # Case A: text equality
        if target_op == "=" and p_target["type"] == "text":
            try:
                candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
            except Exception:
                # If even param query fails (very rare), skip this target condition
                continue

            candidates = [c for c in candidates if str(c) != str(p_target["val"])]
            ordered_cands = candidate_order(p_target["val"], candidates)[:MAX_CANDIDATES_PER_COL]

            for cand in ordered_cands:
                cand_str = str(cand).replace("'", "''")  # for building the final SQL string
                new_cond = f'{safe_ident(target_col)} = \'{cand_str}\''

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed]
                    }

        # Case B: numeric equality
        elif target_op == "=" and p_target["type"] == "num":
            try:
                candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
            except Exception:
                continue

            nums = []
            for c in candidates:
                try:
                    nums.append(float(c))
                except Exception:
                    continue

            nums = sorted(set(nums))
            orig = float(p_target["val"])
            if orig in nums:
                nums.remove(orig)

            for cand in nums[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f'{safe_ident(target_col)} = {cand_repr}'

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed]
                    }

        # Case C: numeric inequality (relax threshold)
        elif p_target["type"] == "num" and target_op in (">=", ">", "<=", "<"):
            try:
                candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
            except Exception:
                continue

            nums = []
            for c in candidates:
                try:
                    nums.append(float(c))
                except Exception:
                    continue
            if not nums:
                continue

            nums = sorted(set(nums))
            orig = float(p_target["val"])

            if target_op in (">=", ">"):
                relaxed = sorted([n for n in nums if n < orig], reverse=True)
            else:
                relaxed = sorted([n for n in nums if n > orig])

            for cand in relaxed[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f'{safe_ident(target_col)} {target_op} {cand_repr}'

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed]
                    }

        else:
            continue

        if not TRY_ALL_CONDITIONS:
            break

    return None, {"replaced": None}


# ===============================
# MAIN
# ===============================
def main():
    conn = sqlite3.connect(db_path)

    total = 0
    empty_total = 0
    fixed_total = 0
    errors_total = 0

    out_records = []

    with open(input_jsonl, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            rec = json.loads(line)
            question = rec.get("question", "")
            gt_sql = (rec.get("gt_sql") or "").strip()

            if not gt_sql:
                errors_total += 1
                continue

            try:
                # Only process empty queries
                if execute_rowcount(conn, gt_sql) > 0:
                    continue

                empty_total += 1

                prefix, where_clause, suffix = get_where_clause(gt_sql)
                if where_clause is None:
                    out_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": None,
                        "difference": {"replaced": None, "error": "No WHERE clause found"}
                    })
                    continue

                conditions = split_conditions(where_clause)

                new_sql, diff = replace_one_condition_to_get_rows(conn, prefix, conditions, suffix)

                if new_sql is not None and execute_rowcount(conn, new_sql) > 0:
                    fixed_total += 1
                    out_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": new_sql,
                        "difference": diff
                    })
                else:
                    out_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": None,
                        "difference": diff
                    })

            except Exception as e:
                errors_total += 1
                out_records.append({
                    "line_number": line_number,
                    "question": question,
                    "empty_gt_sql": gt_sql,
                    "new_gt_sql": None,
                    "difference": {"replaced": None, "error": str(e)}
                })

    conn.close()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print("===== EMPTY FIX V3 REPORT =====")
    print(f"Total records processed: {total}")
    print(f"Empty GT queries found: {empty_total}")
    print(f"Fixed by replacing ONE condition: {fixed_total}")
    print(f"Errors encountered: {errors_total}")
    print(f"Saved output JSON to: {output_json}")
    print(f"(Random seed: {RANDOM_SEED}, context keep frac: {CANDIDATE_CONTEXT_KEEP_FRAC})")


if __name__ == "__main__":
    main()