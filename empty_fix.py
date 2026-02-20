import json
import re
import sqlite3
from difflib import get_close_matches

# ===============================
# CONFIG
# ===============================
db_path = "data/database.db"
input_jsonl = "data/dataset.jsonl"
table_name = "clinical_trials"
output_json = "data/empty_gt_replaced.json"

MAX_CANDIDATES_PER_COL = 25   # how many alternative values to try per column
PREFER_CLOSE_MATCH = True     # try close-match candidates first (useful for typos)
TRY_ALL_CONDITIONS = True     # if False, only try replacing one condition (the tightest heuristic is not implemented)

# ===============================
# SQL helpers
# ===============================
def strip_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()

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
            # treat as bare token string
            sval = raw_val.strip('"')
            nval = None

    return {"col": col, "op": op, "raw_val": raw_val, "sval": sval, "nval": nval, "text": cond}

def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def execute_rowcount(conn: sqlite3.Connection, sql: str) -> int:
    """
    Return number of rows in result (capped cheaply).
    Uses LIMIT 1 check to avoid heavy fetch.
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

    # stringify for matching but keep original objects
    candidates_str = [str(c) for c in candidates]
    if PREFER_CLOSE_MATCH and original_value is not None:
        close = get_close_matches(str(original_value), candidates_str, n=min(len(candidates_str), MAX_CANDIDATES_PER_COL), cutoff=0.05)
        close_set = set(close)
        close_vals = [candidates[i] for i, s in enumerate(candidates_str) if s in close_set]
        rest_vals = [candidates[i] for i, s in enumerate(candidates_str) if s not in close_set]
        return close_vals + rest_vals
    else:
        return candidates

# ===============================
# Replacement logic
# ===============================
def replace_one_condition_to_get_rows(conn, prefix, conditions, suffix):
    """
    Try replacing exactly ONE condition's value (same column) so that query returns rows.
    Returns:
      new_sql or None,
      diff dict {replaced: {column, from, to}}
    """

    # Pre-check: if query already has rows, no replacement needed.
    sql0 = rebuild_sql(prefix, conditions, suffix)
    if execute_rowcount(conn, sql0) > 0:
        return sql0 + ";", {"replaced": None}

    # Try each condition, replacing it with alternative values that exist under the other constraints.
    for i, cond in enumerate(conditions):
        p = parse_condition(cond)
        if not p:
            continue

        col, op = p["col"], p["op"]

        # Build "rest of WHERE" excluding this condition
        rest_conditions = conditions[:i] + conditions[i+1:]
        rest_where = " AND ".join(rest_conditions) if rest_conditions else None

        # Equality STRING replacement: "Col" = 'val'
        if op == "=" and p["sval"] is not None:
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where, limit=5000)
            # remove the original value itself if present
            candidates = [c for c in candidates if str(c) != str(p["sval"])]

            ordered = candidate_order(p["sval"], candidates)[:MAX_CANDIDATES_PER_COL]

            for cand in ordered:
                cand_str = str(cand).replace("'", "''")
                new_cond = f'{safe_ident(col)} = \'{cand_str}\''

                trial = conditions[:i] + [new_cond] + conditions[i+1:]
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        }
                    }

        # Equality NUMERIC replacement: "Col" = 2018
        elif op == "=" and p["nval"] is not None:
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where, limit=5000)
            # keep numeric-like
            num_cands = []
            for c in candidates:
                try:
                    num_cands.append(float(c))
                except Exception:
                    continue
            num_cands = sorted(set(num_cands))
            if p["nval"] in num_cands:
                num_cands.remove(p["nval"])

            ordered = num_cands[:MAX_CANDIDATES_PER_COL]
            for cand in ordered:
                # keep integers clean
                if float(cand).is_integer():
                    cand_repr = str(int(cand))
                else:
                    cand_repr = str(cand)

                new_cond = f'{safe_ident(col)} = {cand_repr}'
                trial = conditions[:i] + [new_cond] + conditions[i+1:]
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        }
                    }

        # Inequalities: attempt to relax / shift threshold
        # Example: "Year" >= 2018 -> try >= 2017, >=2016 ... or <= if it was too strict
        elif p["nval"] is not None and op in (">=", ">", "<=", "<"):
            candidates = distinct_values(conn, table_name, col, extra_where_sql=rest_where, limit=5000)
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

            # Heuristic: choose a "less restrictive" threshold.
            # If it's >= or >, try smaller thresholds; if it's <= or <, try larger thresholds.
            if op in (">=", ">"):
                # smaller first (closest below orig)
                relaxed = [n for n in nums if n < orig]
                relaxed = sorted(relaxed, reverse=True)  # closest below first
            else:
                relaxed = [n for n in nums if n > orig]
                relaxed = sorted(relaxed)  # closest above first

            relaxed = relaxed[:MAX_CANDIDATES_PER_COL]

            for cand in relaxed:
                if float(cand).is_integer():
                    cand_repr = str(int(cand))
                else:
                    cand_repr = str(cand)

                new_cond = f'{safe_ident(col)} {op} {cand_repr}'
                trial = conditions[:i] + [new_cond] + conditions[i+1:]
                trial_sql = rebuild_sql(prefix, trial, suffix)

                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {
                            "column": col,
                            "from": cond,
                            "to": new_cond
                        }
                    }

        # otherwise: unsupported condition type
        else:
            continue

        if not TRY_ALL_CONDITIONS:
            break

    return None, {"replaced": None}

# ===============================
# MAIN
# ===============================
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
            if execute_rowcount(conn, gt_sql) > 0:
                continue  # not empty

            empty_total += 1

            prefix, where_clause, suffix = get_where_clause(gt_sql)
            if where_clause is None:
                # no WHERE to repair by replacement
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

print("===== EMPTY GT CONDITION REPLACEMENT REPORT =====")
print(f"Total records processed: {total}")
print(f"Empty GT queries found: {empty_total}")
print(f"Fixed by replacing ONE condition: {fixed_total}")
print(f"Errors encountered: {errors_total}")
print(f"Saved output JSON to: {output_json}")