"""
empty_fix_v2.py

Goal:
- Read dataset.jsonl with gt_sql
- Identify gt_sql queries that return empty
- Repair each empty query by REPLACING (not removing) exactly one WHERE condition
  with a different value that makes the query return at least 1 row.
- Adds variability:
  - tries "preferred" columns first
  - deprioritizes Cancer type / Name of ICI, etc.
  - uses a relaxed context when sampling candidate DISTINCT values
- Writes output JSON: data/empty_gt_replaced_v2.json
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
output_json = "data/empty_gt_replaced_v2.json"

MAX_CANDIDATES_PER_COL = 25   # how many alternative values to try per column
PREFER_CLOSE_MATCH = True     # try close-match candidates first (useful for typos)
TRY_ALL_CONDITIONS = True     # if False, only try replacing one condition

# -------------------------------
# VARIABILITY CONFIG (V2)
# -------------------------------
RANDOM_SEED = 7
random.seed(RANDOM_SEED)

# Prefer changing these columns earlier (edit to your taste)
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

# Avoid changing these too often (still allowed, just tried later)
DEPRIORITIZED_COLS = [
    "Cancer type",
    "Name of ICI",
    "Class of ICI",
    "Trial name",
    "Treatment regimen",
]

# How many "other conditions" to keep when generating candidate values (relaxed context)
# 1.0 means keep all (old behavior). Lower => more diversity but may drift.
CANDIDATE_CONTEXT_KEEP_FRAC = 0.6


# ===============================
# SQL helpers
# ===============================
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def get_where_clause(sql: str):
    """
    Returns (prefix_before_where, where_clause_string_or_None, suffix_after_where)
    Handles ORDER BY / GROUP BY / LIMIT after WHERE.
    """
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


def parse_condition(cond: str):
    """
    Supports:
      "Col" = 'val'
      "Col" = 2018
      "Col" >= 4, >, <=, <
    """
    m = re.match(r'^\s*"([^"]+)"\s*(=|>=|<=|>|<)\s*(.+?)\s*$', cond)
    if not m:
        return None

    col, op, raw_val = m.group(1), m.group(2), m.group(3).strip()

    if re.match(r"^'.*'$", raw_val):
        sval = raw_val[1:-1]
        nval = None
    else:
        sval = None
        try:
            nval = float(raw_val)
        except Exception:
            sval = raw_val.strip('"')
            nval = None

    return {"col": col, "op": op, "raw_val": raw_val, "sval": sval, "nval": nval, "text": cond}


def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


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
        # fallback: direct
        cur = conn.execute(sql0)
        return 1 if cur.fetchone() is not None else 0


# ===============================
# Candidate generation
# ===============================
def distinct_values(conn, table, col, extra_where_sql=None, limit=5000):
    """
    DISTINCT values for a column, optionally under a WHERE constraint.
    """
    colq = safe_ident(col)
    tblq = safe_ident(table)
    sql = f"SELECT DISTINCT {colq} FROM {tblq} WHERE {colq} IS NOT NULL"
    if extra_where_sql:
        sql += f" AND ({extra_where_sql})"
    sql += f" LIMIT {int(limit)};"
    cur = conn.execute(sql)
    return [r[0] for r in cur.fetchall() if r[0] is not None]


def candidate_order(original_value, candidates):
    """
    Order candidate values: close matches first (optional), then remaining.
    """
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
    else:
        return candidates


# ===============================
# VARIABILITY HELPERS (V2)
# ===============================
def choose_context_subset(rest_conditions, keep_frac=0.6):
    """
    Keep only a random subset of rest_conditions when computing candidate values.
    This increases candidate diversity for tight queries.
    """
    if not rest_conditions:
        return None
    k = max(0, int(round(len(rest_conditions) * keep_frac)))
    if k >= len(rest_conditions):
        return " AND ".join(rest_conditions)
    kept = random.sample(rest_conditions, k) if k > 0 else []
    return " AND ".join(kept) if kept else None


def order_conditions_for_try(conditions):
    """
    Reorder conditions so we *try* replacing preferred columns earlier,
    and deprioritize certain columns later.
    Also shuffles within a rank to increase variety.
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
        random.shuffle(group)  # shuffle within rank
        final.extend(group)

    return final


# ===============================
# Replacement logic (V2)
# ===============================
def replace_one_condition_to_get_rows(conn, prefix, conditions, suffix):
    """
    V2:
    - varied order of conditions attempted
    - relaxed candidate context when sampling DISTINCT values
    - replaces exactly ONE condition
    """
    sql0 = rebuild_sql(prefix, conditions, suffix)
    if execute_rowcount(conn, sql0) > 0:
        return sql0 + ";", {"replaced": None}

    ordered_conditions = order_conditions_for_try(conditions)

    for original_index, cond, p in ordered_conditions:
        if not p:
            continue

        col, op = p["col"], p["op"]

        # rest conditions exclude the target condition
        rest_conditions = [c for j, c in enumerate(conditions) if j != original_index]

        # relaxed context for candidate generation
        rest_where_for_candidates = choose_context_subset(
            rest_conditions,
            keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC
        )

        # 1) Equality STRING
        if op == "=" and p["sval"] is not None:
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where_for_candidates, limit=5000)
            candidates = [c for c in candidates if str(c) != str(p["sval"])]

            ordered_cands = candidate_order(p["sval"], candidates)[:MAX_CANDIDATES_PER_COL]

            for cand in ordered_cands:
                cand_str = str(cand).replace("'", "''")
                new_cond = f'{safe_ident(col)} = \'{cand_str}\''

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        },
                        "candidate_context_used": rest_where_for_candidates
                    }

        # 2) Equality NUMERIC
        elif op == "=" and p["nval"] is not None:
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where_for_candidates, limit=5000)
            num_cands = []
            for c in candidates:
                try:
                    num_cands.append(float(c))
                except Exception:
                    continue
            num_cands = sorted(set(num_cands))

            orig = float(p["nval"])
            if orig in num_cands:
                num_cands.remove(orig)

            for cand in num_cands[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f'{safe_ident(col)} = {cand_repr}'

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        },
                        "candidate_context_used": rest_where_for_candidates
                    }

        # 3) Inequalities: relax threshold
        elif p["nval"] is not None and op in (">=", ">", "<=", "<"):
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where_for_candidates, limit=5000)
            nums = []
            for c in candidates:
                try:
                    nums.append(float(c))
                except Exception:
                    continue
            if not nums:
                continue

            nums = sorted(set(nums))
            orig = float(p["nval"])

            if op in (">=", ">"):
                relaxed = sorted([n for n in nums if n < orig], reverse=True)
            else:
                relaxed = sorted([n for n in nums if n > orig])

            for cand in relaxed[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f'{safe_ident(col)} {op} {cand_repr}'

                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        },
                        "candidate_context_used": rest_where_for_candidates
                    }

        # unsupported condition type -> skip
        else:
            continue

        if not TRY_ALL_CONDITIONS:
            break

    return None, {"replaced": None}


# ===============================
# MAIN (empty_fix v2)
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

    print("===== EMPTY FIX V2 REPORT =====")
    print(f"Total records processed: {total}")
    print(f"Empty GT queries found: {empty_total}")
    print(f"Fixed by replacing ONE condition: {fixed_total}")
    print(f"Errors encountered: {errors_total}")
    print(f"Saved output JSON to: {output_json}")
    print(f"(Random seed: {RANDOM_SEED}, context keep frac: {CANDIDATE_CONTEXT_KEEP_FRAC})")


if __name__ == "__main__":
    main()