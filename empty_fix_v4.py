#!/usr/bin/env python3
"""
empty_fix_v5_nodrop_bruteforce_fixedsplit.py

Fixes EMPTY gt_sql queries WITHOUT removing any WHERE predicates.
Only changes values/thresholds inside existing predicates.

Pipeline:
  1) Replace ONE predicate value
  2) Replace TWO predicate values (bounded)
  3) Replace MANY predicate values (greedy)
  4) GUARANTEED brute-force: ROW-BACKED fix (rewrites EVERY predicate to match a real row)

Critical fix vs prior versions:
  - Robust WHERE splitting: do NOT split on AND inside quoted strings
    (prevents breaking 'Head and Neck' into 'Head' + 'Neck')

Outputs:
  data/empty_gt_replaced_v5_fixed.json
  data/empty_gt_unfixed_v5.json
"""

import json
import re
import sqlite3
import random
import traceback
from difflib import get_close_matches
from typing import List, Optional, Dict, Any, Tuple

# ===============================
# CONFIG
# ===============================
db_path = "data/database.db"
input_jsonl = "data/dataset.jsonl"
table_name = "clinical_trials"

output_json_fixed = "data/empty_gt_replaced_v5_fixed.json"
output_json_unfixed = "data/empty_gt_unfixed_v5.json"

MAX_CANDIDATES_PER_COL = 25
PREFER_CLOSE_MATCH = True
TRY_ALL_CONDITIONS = True

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

ENABLE_TWO_CONDITION_REPLACE = True
MAX_PAIR_TRIES = 250

ENABLE_REPLACE_MANY = True
REPLACE_MANY_MAX_CHANGES = 4

ENABLE_ROW_BACKED_FIX = True

# When row-backed chooses a row: LIMIT 1 can be too deterministic.
# Optionally randomize: ORDER BY RANDOM() (slower but more variety).
ROW_BACKED_RANDOM_ROW = False


# ===============================
# SAFETY: always unpack safely
# ===============================
def safe_call2(fn, *args, fn_name="func"):
    try:
        res = fn(*args)
        if not isinstance(res, tuple) or len(res) != 2:
            return None, {
                "replaced": None,
                "error": f"{fn_name} returned {type(res).__name__}, expected (new_sql, diff). value={res!r}",
                "traceback": None,
                "stage": fn_name,
            }
        return res
    except Exception as e:
        return None, {
            "replaced": None,
            "error": f"{fn_name} exception: {str(e)}",
            "traceback": traceback.format_exc(),
            "stage": fn_name,
        }


# ===============================
# SQL helpers
# ===============================
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def get_where_clause(sql: str):
    """
    Returns (prefix_before_where, where_clause_or_None, suffix_after_where)
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


def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def rebuild_sql(prefix: str, conditions, suffix: str):
    if not conditions:
        return prefix + (" " + suffix if suffix else "")
    return prefix + " WHERE " + " AND ".join(conditions) + (" " + suffix if suffix else "")


def execute_rowcount(conn: sqlite3.Connection, sql: str) -> int:
    sql0 = strip_trailing_semicolon(sql)
    try:
        cur = conn.execute(f"SELECT 1 FROM ({sql0}) AS subq LIMIT 1;")
        return 1 if cur.fetchone() is not None else 0
    except Exception:
        cur = conn.execute(sql0)
        return 1 if cur.fetchone() is not None else 0


# ===============================
# IMPORTANT: robust WHERE splitting (do NOT split on AND inside quotes)
# ===============================
def split_conditions(where_clause: str) -> List[str]:
    """
    Split WHERE clause on AND tokens that are OUTSIDE single quotes.
    Keeps predicates intact when values contain 'and' (e.g. 'Head and Neck').

    Assumptions:
    - SQL uses single quotes for string literals
    - We don't handle nested parentheses boolean logic perfectly; but works well
      for the dataset format (flat AND chain).
    """
    if not where_clause:
        return []

    s = where_clause.strip()
    out = []
    buf = []
    in_quote = False

    i = 0
    n = len(s)

    def flush():
        part = "".join(buf).strip()
        if part:
            out.append(part)

    while i < n:
        ch = s[i]

        if ch == "'":
            # handle doubled quotes '' inside string
            if in_quote and i + 1 < n and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue

        # detect AND outside quotes with word boundaries
        if not in_quote:
            if (i + 3 <= n and s[i:i+3].upper() == "AND"):
                left_ok = (i == 0) or (not s[i-1].isalnum() and s[i-1] != "_")
                right_ok = (i + 3 == n) or (not s[i+3].isalnum() and s[i+3] != "_")
                if left_ok and right_ok:
                    flush()
                    buf = []
                    i += 3
                    # skip whitespace after AND
                    while i < n and s[i].isspace():
                        i += 1
                    continue

        buf.append(ch)
        i += 1

    flush()
    return out


def parse_condition(cond: str) -> Optional[Dict[str, Any]]:
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
        return {"col": col, "op": op, "type": "text", "val": sval, "text": cond}

    try:
        nval = float(raw_val)
        return {"col": col, "op": op, "type": "num", "val": nval, "text": cond}
    except Exception:
        return None


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
# SAFE DISTINCT candidate sampling (parameterized)
# ===============================
def build_param_where(conditions_parsed):
    if not conditions_parsed:
        return "", []

    parts = []
    params = []
    for p in conditions_parsed:
        colq = safe_ident(p["col"])
        parts.append(f"{colq} {p['op']} ?")
        params.append(p["val"])
    return " AND ".join(parts), params


def distinct_values_param(conn, table, col, parsed_context_conditions, limit=5000):
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
# Variability helpers
# ===============================
def choose_context_subset(parsed_rest_conditions, keep_frac=0.6):
    if not parsed_rest_conditions:
        return []
    k = max(0, int(round(len(parsed_rest_conditions) * keep_frac)))
    if k >= len(parsed_rest_conditions):
        return parsed_rest_conditions[:]
    if k == 0:
        return []
    return random.sample(parsed_rest_conditions, k)


def order_conditions_for_try(conditions):
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
# Fix method 1: replace ONE predicate value
# ===============================
def replace_one_condition_to_get_rows(conn, prefix, conditions, suffix):
    sql0 = rebuild_sql(prefix, conditions, suffix)
    if execute_rowcount(conn, sql0) > 0:
        return sql0 + ";", {"replaced": None, "method": "REPLACE_1_ALREADY_OK"}

    ordered_conditions = order_conditions_for_try(conditions)

    for i, cond, p in ordered_conditions:
        if not p:
            continue

        col, op, typ, val = p["col"], p["op"], p["type"], p["val"]

        rest_parsed = []
        for j, c in enumerate(conditions):
            if j == i:
                continue
            pj = parse_condition(c)
            if pj:
                rest_parsed.append(pj)

        context = choose_context_subset(rest_parsed, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

        if op == "=" and typ == "text":
            cands = distinct_values_param(conn, table_name, col, context, limit=5000)
            cands = [x for x in cands if str(x) != str(val)]
            for cand in candidate_order(val, cands)[:MAX_CANDIDATES_PER_COL]:
                s = str(cand).replace("'", "''")
                new_cond = f"{safe_ident(col)} = '{s}'"
                trial = conditions[:]
                trial[i] = new_cond
                if execute_rowcount(conn, rebuild_sql(prefix, trial, suffix)) > 0:
                    return rebuild_sql(prefix, trial, suffix) + ";", {
                        "replaced": {"column": col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context],
                        "method": "REPLACE_1",
                    }

        elif op == "=" and typ == "num":
            cands = distinct_values_param(conn, table_name, col, context, limit=5000)
            nums = []
            for x in cands:
                try:
                    nums.append(float(x))
                except Exception:
                    pass
            nums = sorted(set(nums))
            if float(val) in nums:
                nums.remove(float(val))

            for cand in nums[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f"{safe_ident(col)} = {cand_repr}"
                trial = conditions[:]
                trial[i] = new_cond
                if execute_rowcount(conn, rebuild_sql(prefix, trial, suffix)) > 0:
                    return rebuild_sql(prefix, trial, suffix) + ";", {
                        "replaced": {"column": col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context],
                        "method": "REPLACE_1",
                    }

        elif typ == "num" and op in (">=", ">", "<=", "<"):
            cands = distinct_values_param(conn, table_name, col, context, limit=5000)
            nums = []
            for x in cands:
                try:
                    nums.append(float(x))
                except Exception:
                    pass
            nums = sorted(set(nums))
            if not nums:
                continue

            orig = float(val)
            if op in (">=", ">"):
                relaxed = sorted([n for n in nums if n < orig], reverse=True)
            else:
                relaxed = sorted([n for n in nums if n > orig])

            for cand in relaxed[:MAX_CANDIDATES_PER_COL]:
                cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                new_cond = f"{safe_ident(col)} {op} {cand_repr}"
                trial = conditions[:]
                trial[i] = new_cond
                if execute_rowcount(conn, rebuild_sql(prefix, trial, suffix)) > 0:
                    return rebuild_sql(prefix, trial, suffix) + ";", {
                        "replaced": {"column": col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context],
                        "method": "REPLACE_1",
                    }

        if not TRY_ALL_CONDITIONS:
            break

    return None, {"replaced": None, "method": "REPLACE_1_FAILED"}


# ===============================
# Fix method 2: replace TWO predicate values (bounded)
# ===============================
def replace_two_conditions_to_get_rows(conn, prefix, conditions, suffix):
    parsed = [(idx, conditions[idx], parse_condition(conditions[idx])) for idx in range(len(conditions))]
    parsed = [(i, c, p) for (i, c, p) in parsed if p is not None]
    if len(parsed) < 2:
        return None, {"replaced": None, "method": "REPLACE_2_NOT_ENOUGH_PARSEABLE"}

    ordered_idxs = [t[0] for t in order_conditions_for_try(conditions) if t[2] is not None]
    tries = 0

    def candidates_for(p, context):
        col, op, typ, val = p["col"], p["op"], p["type"], p["val"]
        if op == "=" and typ == "text":
            vals = distinct_values_param(conn, table_name, col, context, limit=5000)
            vals = [v for v in vals if str(v) != str(val)]
            return candidate_order(val, vals)[:max(8, MAX_CANDIDATES_PER_COL // 2)]
        if op == "=" and typ == "num":
            vals = distinct_values_param(conn, table_name, col, context, limit=5000)
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            nums = sorted(set(nums))
            if float(val) in nums:
                nums.remove(float(val))
            return nums[:max(8, MAX_CANDIDATES_PER_COL // 2)]
        if typ == "num" and op in (">=", ">", "<=", "<"):
            vals = distinct_values_param(conn, table_name, col, context, limit=5000)
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            nums = sorted(set(nums))
            orig = float(val)
            if op in (">=", ">"):
                relaxed = sorted([n for n in nums if n < orig], reverse=True)
            else:
                relaxed = sorted([n for n in nums if n > orig])
            return relaxed[:max(8, MAX_CANDIDATES_PER_COL // 2)]
        return []

    def cond_from(p, v):
        colq = safe_ident(p["col"])
        if p["op"] == "=" and p["type"] == "text":
            s = str(v).replace("'", "''")
            return f"{colq} = '{s}'"
        if p["op"] == "=" and p["type"] == "num":
            vv = float(v)
            vv = int(vv) if float(vv).is_integer() else vv
            return f"{colq} = {vv}"
        if p["type"] == "num" and p["op"] in (">=", ">", "<=", "<"):
            vv = float(v)
            vv = int(vv) if float(vv).is_integer() else vv
            return f"{colq} {p['op']} {vv}"
        return None

    for a_pos in range(len(ordered_idxs)):
        for b_pos in range(a_pos + 1, len(ordered_idxs)):
            i = ordered_idxs[a_pos]
            j = ordered_idxs[b_pos]
            pi = parse_condition(conditions[i])
            pj = parse_condition(conditions[j])
            if not pi or not pj:
                continue

            rest_parsed = []
            for k, c in enumerate(conditions):
                if k in (i, j):
                    continue
                pk = parse_condition(c)
                if pk:
                    rest_parsed.append(pk)

            context = choose_context_subset(rest_parsed, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

            cand_i = candidates_for(pi, context)
            cand_j = candidates_for(pj, context)
            if not cand_i or not cand_j:
                continue

            for vi in cand_i:
                for vj in cand_j:
                    tries += 1
                    if tries > MAX_PAIR_TRIES:
                        return None, {"replaced": None, "pair_search_capped": True, "method": "REPLACE_2_CAPPED"}

                    new_i = cond_from(pi, vi)
                    new_j = cond_from(pj, vj)
                    if not new_i or not new_j:
                        continue

                    trial = conditions[:]
                    trial[i] = new_i
                    trial[j] = new_j
                    if execute_rowcount(conn, rebuild_sql(prefix, trial, suffix)) > 0:
                        return rebuild_sql(prefix, trial, suffix) + ";", {
                            "replaced": {
                                "pair": [
                                    {"column": pi["col"], "from": conditions[i], "to": new_i},
                                    {"column": pj["col"], "from": conditions[j], "to": new_j},
                                ]
                            },
                            "candidate_context_used": [pp["text"] for pp in context],
                            "method": "REPLACE_2",
                        }

    return None, {"replaced": None, "method": "REPLACE_2_FAILED"}


# ===============================
# Fix method 3: replace MANY predicate values (greedy)
# ===============================
def replace_many_values_keep_all_preds(conn, prefix, conditions, suffix, max_changes=4):
    def has_rows(conds):
        return execute_rowcount(conn, rebuild_sql(prefix, conds, suffix)) > 0

    if has_rows(conditions):
        return rebuild_sql(prefix, conditions, suffix) + ";", {"method": "REPLACE_MANY_ALREADY_OK", "changes": []}

    conds = conditions[:]
    changes = []

    for _ in range(max_changes):
        made_progress = False
        ordered = order_conditions_for_try(conds)

        for idx, old_cond, p in ordered:
            if not p:
                continue
            col, op, typ, val = p["col"], p["op"], p["type"], p["val"]

            rest_parsed = []
            for j, c in enumerate(conds):
                if j == idx:
                    continue
                pj = parse_condition(c)
                if pj:
                    rest_parsed.append(pj)

            context = choose_context_subset(rest_parsed, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

            if op == "=" and typ == "text":
                try:
                    cands = distinct_values_param(conn, table_name, col, context, limit=5000)
                except Exception:
                    continue
                cands = [x for x in cands if str(x) != str(val)]
                for cand in candidate_order(val, cands)[:MAX_CANDIDATES_PER_COL]:
                    s = str(cand).replace("'", "''")
                    new_cond = f"{safe_ident(col)} = '{s}'"
                    trial = conds[:]
                    trial[idx] = new_cond
                    if has_rows(trial):
                        changes.append({"column": col, "from": old_cond, "to": new_cond})
                        conds = trial
                        made_progress = True
                        break

            elif op == "=" and typ == "num":
                try:
                    cands = distinct_values_param(conn, table_name, col, context, limit=5000)
                except Exception:
                    continue
                nums = []
                for x in cands:
                    try:
                        nums.append(float(x))
                    except Exception:
                        pass
                nums = sorted(set(nums))
                if float(val) in nums:
                    nums.remove(float(val))

                for cand in nums[:MAX_CANDIDATES_PER_COL]:
                    cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                    new_cond = f"{safe_ident(col)} = {cand_repr}"
                    trial = conds[:]
                    trial[idx] = new_cond
                    if has_rows(trial):
                        changes.append({"column": col, "from": old_cond, "to": new_cond})
                        conds = trial
                        made_progress = True
                        break

            elif typ == "num" and op in (">=", ">", "<=", "<"):
                try:
                    cands = distinct_values_param(conn, table_name, col, context, limit=5000)
                except Exception:
                    continue
                nums = []
                for x in cands:
                    try:
                        nums.append(float(x))
                    except Exception:
                        pass
                nums = sorted(set(nums))
                if not nums:
                    continue

                orig = float(val)
                if op in (">=", ">"):
                    relaxed = sorted([n for n in nums if n < orig], reverse=True)
                else:
                    relaxed = sorted([n for n in nums if n > orig])

                for cand in relaxed[:MAX_CANDIDATES_PER_COL]:
                    cand_repr = str(int(cand)) if float(cand).is_integer() else str(cand)
                    new_cond = f"{safe_ident(col)} {op} {cand_repr}"
                    trial = conds[:]
                    trial[idx] = new_cond
                    if has_rows(trial):
                        changes.append({"column": col, "from": old_cond, "to": new_cond})
                        conds = trial
                        made_progress = True
                        break

            if made_progress:
                break

        if has_rows(conds):
            return rebuild_sql(prefix, conds, suffix) + ";", {"method": "REPLACE_MANY", "changes": changes}

        if not made_progress:
            break

    return None, {"method": "REPLACE_MANY_FAILED", "changes": changes}


# ===============================
# Fix method 4: GUARANTEED brute-force row-backed fix
# ===============================
def fetch_one_row_any(conn, table):
    if ROW_BACKED_RANDOM_ROW:
        cur = conn.execute(f'SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT 1;')
    else:
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT 1;')

    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if not row:
        return None
    return dict(zip(cols, row))


def build_condition_from_row(p, row):
    col = p["col"]
    op = p["op"]
    typ = p["type"]

    if col not in row or row[col] is None:
        return None

    colq = safe_ident(col)
    rv = row[col]

    if typ == "text" and op == "=":
        s = str(rv).replace("'", "''")
        return f"{colq} = '{s}'"

    # numeric
    try:
        rnum = float(rv)
    except Exception:
        return None

    if op == "=":
        v = int(rnum) if float(rnum).is_integer() else rnum
        return f"{colq} = {v}"

    # inequality thresholds that INCLUDE the row
    if op in (">=", ">"):
        v = (rnum - 1e-9) if op == ">" else rnum
    elif op in ("<=", "<"):
        v = (rnum + 1e-9) if op == "<" else rnum
    else:
        return None

    v = int(v) if float(v).is_integer() else v
    return f"{colq} {op} {v}"


def brute_force_row_backed_fix(conn, prefix, conditions, suffix):
    row = fetch_one_row_any(conn, table_name)
    if not row:
        return None, {"method": "ROW_BACKED_FAILED", "error": "No rows in table"}

    new_conds = []
    changes = []

    for old_cond in conditions:
        p = parse_condition(old_cond)
        if not p:
            return None, {"method": "ROW_BACKED_FAILED", "error": f"Unparseable predicate: {old_cond}"}

        new_cond = build_condition_from_row(p, row)
        if not new_cond:
            return None, {"method": "ROW_BACKED_FAILED", "error": f"Cannot build predicate for: {old_cond}"}

        new_conds.append(new_cond)

        old_norm = re.sub(r"\s+", " ", old_cond.strip())
        new_norm = re.sub(r"\s+", " ", new_cond.strip())
        if old_norm != new_norm:
            changes.append({"column": p["col"], "from": old_cond, "to": new_cond})

    new_sql = rebuild_sql(prefix, new_conds, suffix) + ";"
    if execute_rowcount(conn, new_sql) > 0:
        return new_sql, {"method": "ROW_BACKED_FIX", "changes": changes}

    return None, {"method": "ROW_BACKED_FAILED", "error": "Built SQL still empty", "changes": changes}


# ===============================
# MAIN
# ===============================
def main():
    conn = sqlite3.connect(db_path)

    total = 0
    empty_total = 0

    fixed_total = 0
    fixed_by_1 = 0
    fixed_by_2 = 0
    fixed_by_many = 0
    fixed_by_row = 0

    unfixed_total = 0

    fixed_records = []
    unfixed_records = []

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
                continue

            # Skip queries that already return rows
            if execute_rowcount(conn, gt_sql) > 0:
                continue

            empty_total += 1

            prefix, where_clause, suffix = get_where_clause(gt_sql)
            if where_clause is None:
                unfixed_total += 1
                unfixed_records.append({
                    "line_number": line_number,
                    "question": question,
                    "empty_gt_sql": gt_sql,
                    "new_gt_sql": None,
                    "difference": {"error": "No WHERE clause found"},
                })
                continue

            conditions = split_conditions(where_clause)
            if not conditions:
                unfixed_total += 1
                unfixed_records.append({
                    "line_number": line_number,
                    "question": question,
                    "empty_gt_sql": gt_sql,
                    "new_gt_sql": None,
                    "difference": {"error": "Empty WHERE clause"},
                })
                continue

            stage_diffs = {}

            # 1) Replace one
            new_sql, diff = safe_call2(
                replace_one_condition_to_get_rows,
                conn, prefix, conditions, suffix,
                fn_name="replace_one_condition_to_get_rows"
            )
            stage_diffs["replace_1"] = diff
            if new_sql and execute_rowcount(conn, new_sql) > 0:
                fixed_total += 1
                fixed_by_1 += 1
                fixed_records.append({
                    "line_number": line_number,
                    "question": question,
                    "empty_gt_sql": gt_sql,
                    "new_gt_sql": new_sql,
                    "difference": diff,
                })
                continue

            # 2) Replace two
            if ENABLE_TWO_CONDITION_REPLACE:
                new_sql2, diff2 = safe_call2(
                    replace_two_conditions_to_get_rows,
                    conn, prefix, conditions, suffix,
                    fn_name="replace_two_conditions_to_get_rows"
                )
                stage_diffs["replace_2"] = diff2
                if new_sql2 and execute_rowcount(conn, new_sql2) > 0:
                    fixed_total += 1
                    fixed_by_2 += 1
                    fixed_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": new_sql2,
                        "difference": diff2,
                    })
                    continue

            # 3) Replace many
            if ENABLE_REPLACE_MANY:
                new_sql3, diff3 = safe_call2(
                    replace_many_values_keep_all_preds,
                    conn, prefix, conditions, suffix, REPLACE_MANY_MAX_CHANGES,
                    fn_name="replace_many_values_keep_all_preds"
                )
                stage_diffs["replace_many"] = diff3
                if new_sql3 and execute_rowcount(conn, new_sql3) > 0:
                    fixed_total += 1
                    fixed_by_many += 1
                    fixed_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": new_sql3,
                        "difference": diff3,
                    })
                    continue

            # 4) Guaranteed brute force
            if ENABLE_ROW_BACKED_FIX:
                new_sql4, diff4 = safe_call2(
                    brute_force_row_backed_fix,
                    conn, prefix, conditions, suffix,
                    fn_name="brute_force_row_backed_fix"
                )
                stage_diffs["row_backed"] = diff4
                if new_sql4 and execute_rowcount(conn, new_sql4) > 0:
                    fixed_total += 1
                    fixed_by_row += 1
                    fixed_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": new_sql4,
                        "difference": diff4,
                    })
                    continue

            # unfixed
            unfixed_total += 1
            unfixed_records.append({
                "line_number": line_number,
                "question": question,
                "empty_gt_sql": gt_sql,
                "new_gt_sql": None,
                "difference": stage_diffs,
            })

    conn.close()

    with open(output_json_fixed, "w", encoding="utf-8") as f:
        json.dump(fixed_records, f, ensure_ascii=False, indent=2)

    with open(output_json_unfixed, "w", encoding="utf-8") as f:
        json.dump(unfixed_records, f, ensure_ascii=False, indent=2)

    print("===== EMPTY FIX V5 (NO-DROP + BRUTE FORCE + SAFE SPLIT) REPORT =====")
    print(f"Total records processed: {total}")
    print(f"Empty GT queries found: {empty_total}")
    print(f"Fixed total: {fixed_total}")
    print(f"  - Fixed by 1-condition replace: {fixed_by_1}")
    print(f"  - Fixed by 2-condition replace: {fixed_by_2}")
    print(f"  - Fixed by replace-many: {fixed_by_many}")
    print(f"  - Fixed by row-backed brute force: {fixed_by_row}")
    print(f"Unfixed total: {unfixed_total}")
    print(f"Saved fixed JSON to: {output_json_fixed}")
    print(f"Saved unfixed JSON to: {output_json_unfixed}")
    print(f"(ROW_BACKED_RANDOM_ROW={ROW_BACKED_RANDOM_ROW})")


if __name__ == "__main__":
    main()