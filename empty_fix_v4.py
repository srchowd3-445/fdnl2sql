"""
empty_fix_v4_fixed.py

V4 with robustness fixes:
- Safe-unpack wrapper around repair functions (prevents "cannot unpack NoneType")
- Traceback captured in output JSON for debugging
- Defensive checks so every stage either returns a valid tuple or is treated as failure gracefully
"""

import json
import re
import sqlite3
import random
import traceback
from difflib import get_close_matches

# ===============================
# CONFIG
# ===============================
db_path = "data/database.db"
input_jsonl = "data/dataset.jsonl"
table_name = "clinical_trials"
output_json = "data/empty_gt_replaced_v4_fixed.json"

MAX_CANDIDATES_PER_COL = 25
PREFER_CLOSE_MATCH = True
TRY_ALL_CONDITIONS = True

# -------------------------------
# VARIABILITY CONFIG
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

# -------------------------------
# BRUTE-FORCE / GUARANTEE CONFIG (V4)
# -------------------------------
ENABLE_TWO_CONDITION_REPLACE = True
MAX_PAIR_TRIES = 250

ANCHOR_FALLBACK = True
ANCHOR_COLS = [
    "Cancer type",
    "Trial phase",
    "Class of ICI",
    "Name of ICI",
    "Control arm",
    "Monotherapy/combination",
    "Type of combination",
    "Clinical setting",
    "Primary endpoint",
]


# ===============================
# SAFETY: always unpack safely
# ===============================
def safe_call2(fn, *args, fn_name="func"):
    """
    Ensures fn returns a 2-tuple. If not, returns (None, {error...}) instead of crashing.
    """
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


def execute_rowcount(conn: sqlite3.Connection, sql: str) -> int:
    sql0 = strip_trailing_semicolon(sql)
    try:
        cur = conn.execute(f"SELECT 1 FROM ({sql0}) AS subq LIMIT 1;")
        return 1 if cur.fetchone() is not None else 0
    except Exception:
        cur = conn.execute(sql0)
        return 1 if cur.fetchone() is not None else 0


def fetch_one_row_dict(conn: sqlite3.Connection, sql: str):
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return None
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip(cols, row))


def build_eq_condition(col: str, val):
    colq = safe_ident(col)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return f"{colq} = {val}"
    sval = str(val).replace("'", "''")
    return f"{colq} = '{sval}'"


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
# 1-condition replace
# ===============================
def replace_one_condition_to_get_rows(conn, prefix, conditions, suffix):
    sql0 = rebuild_sql(prefix, conditions, suffix)
    if execute_rowcount(conn, sql0) > 0:
        return sql0 + ";", {"replaced": None, "method": "REPLACE_1_ALREADY_OK"}

    ordered_conditions = order_conditions_for_try(conditions)

    for original_index, cond, p_target in ordered_conditions:
        if not p_target:
            continue

        target_col = p_target["col"]
        target_op = p_target["op"]

        rest_conditions_raw = [c for j, c in enumerate(conditions) if j != original_index]
        rest_parsed_all = [parse_condition(c) for c in rest_conditions_raw]
        rest_parsed_all = [p for p in rest_parsed_all if p is not None]
        context_parsed = choose_context_subset(rest_parsed_all, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

        if target_op == "=" and p_target["type"] == "text":
            candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
            candidates = [c for c in candidates if str(c) != str(p_target["val"])]
            ordered_cands = candidate_order(p_target["val"], candidates)[:MAX_CANDIDATES_PER_COL]

            for cand in ordered_cands:
                cand_str = str(cand).replace("'", "''")
                new_cond = f"{safe_ident(target_col)} = '{cand_str}'"
                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)
                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed],
                        "method": "REPLACE_1"
                    }

        elif target_op == "=" and p_target["type"] == "num":
            candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
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
                new_cond = f"{safe_ident(target_col)} = {cand_repr}"
                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)
                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed],
                        "method": "REPLACE_1"
                    }

        elif p_target["type"] == "num" and target_op in (">=", ">", "<=", "<"):
            candidates = distinct_values_param(conn, table_name, target_col, context_parsed, limit=5000)
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
                new_cond = f"{safe_ident(target_col)} {target_op} {cand_repr}"
                trial = conditions[:]
                trial[original_index] = new_cond
                trial_sql = rebuild_sql(prefix, trial, suffix)
                if execute_rowcount(conn, trial_sql) > 0:
                    return trial_sql + ";", {
                        "replaced": {"column": target_col, "from": cond, "to": new_cond},
                        "candidate_context_used": [pp["text"] for pp in context_parsed],
                        "method": "REPLACE_1"
                    }

        if not TRY_ALL_CONDITIONS:
            break

    return None, {"replaced": None, "method": "REPLACE_1_FAILED"}


# ===============================
# 2-condition bounded brute force
# ===============================
def replace_two_conditions_to_get_rows(conn, prefix, conditions, suffix):
    parsed = [(idx, cond, parse_condition(cond)) for idx, cond in enumerate(conditions)]
    parsed = [(i, c, p) for (i, c, p) in parsed if p is not None]
    if len(parsed) < 2:
        return None, {"replaced": None, "method": "REPLACE_2_NOT_ENOUGH_PARSEABLE"}

    ordered = order_conditions_for_try(conditions)
    ordered_idxs = [t[0] for t in ordered if t[2] is not None]

    tries = 0

    def candidates_for(p, context_parsed):
        col = p["col"]
        op = p["op"]
        if op == "=" and p["type"] == "text":
            vals = distinct_values_param(conn, table_name, col, context_parsed, limit=5000)
            vals = [v for v in vals if str(v) != str(p["val"])]
            return candidate_order(p["val"], vals)[:max(8, MAX_CANDIDATES_PER_COL // 2)]
        if op == "=" and p["type"] == "num":
            vals = distinct_values_param(conn, table_name, col, context_parsed, limit=5000)
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            nums = sorted(set(nums))
            orig = float(p["val"])
            if orig in nums:
                nums.remove(orig)
            return nums[:max(8, MAX_CANDIDATES_PER_COL // 2)]
        if p["type"] == "num" and op in (">=", ">", "<=", "<"):
            vals = distinct_values_param(conn, table_name, col, context_parsed, limit=5000)
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            nums = sorted(set(nums))
            orig = float(p["val"])
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

            rest_raw = [c for k, c in enumerate(conditions) if k not in (i, j)]
            rest_parsed = [parse_condition(c) for c in rest_raw]
            rest_parsed = [p for p in rest_parsed if p is not None]
            context_parsed = choose_context_subset(rest_parsed, keep_frac=CANDIDATE_CONTEXT_KEEP_FRAC)

            cand_i = candidates_for(pi, context_parsed)
            cand_j = candidates_for(pj, context_parsed)
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
                    trial_sql = rebuild_sql(prefix, trial, suffix)

                    if execute_rowcount(conn, trial_sql) > 0:
                        return trial_sql + ";", {
                            "replaced": {
                                "pair": [
                                    {"column": pi["col"], "from": conditions[i], "to": new_i},
                                    {"column": pj["col"], "from": conditions[j], "to": new_j},
                                ]
                            },
                            "candidate_context_used": [pp["text"] for pp in context_parsed],
                            "method": "REPLACE_2"
                        }

    return None, {"replaced": None, "method": "REPLACE_2_FAILED"}


# ===============================
# Anchor fallback (guarantee)
# ===============================
def anchor_fallback_sql(conn, original_sql):
    prefix, where_clause, suffix = get_where_clause(original_sql)
    base_sql = prefix + (" " + suffix if suffix else "")

    row = fetch_one_row_dict(conn, base_sql)
    if not row:
        if execute_rowcount(conn, base_sql) > 0:
            return base_sql + ";", {"fallback": "DROP_WHERE_BASE", "method": "ANCHOR_FALLBACK"}
        return None, {"fallback": "NO_ROW_FROM_BASE", "method": "ANCHOR_FALLBACK"}

    anchors = [c for c in ANCHOR_COLS if c in row and row[c] is not None]
    random.shuffle(anchors)

    for a in anchors[:10]:
        cond = build_eq_condition(a, row[a])
        if not cond:
            continue
        trial_sql = prefix + " WHERE " + cond + (" " + suffix if suffix else "")
        if execute_rowcount(conn, trial_sql) > 0:
            return trial_sql + ";", {"fallback": "ANCHOR_1", "anchors": [cond], "method": "ANCHOR_FALLBACK"}

    for i in range(min(8, len(anchors))):
        for j in range(i + 1, min(8, len(anchors))):
            c1 = build_eq_condition(anchors[i], row[anchors[i]])
            c2 = build_eq_condition(anchors[j], row[anchors[j]])
            if not c1 or not c2:
                continue
            where = f"{c1} AND {c2}"
            trial_sql = prefix + " WHERE " + where + (" " + suffix if suffix else "")
            if execute_rowcount(conn, trial_sql) > 0:
                return trial_sql + ";", {"fallback": "ANCHOR_2", "anchors": [c1, c2], "method": "ANCHOR_FALLBACK"}

    if execute_rowcount(conn, base_sql) > 0:
        return base_sql + ";", {"fallback": "DROP_WHERE_BASE", "method": "ANCHOR_FALLBACK"}

    return None, {"fallback": "FAILED", "method": "ANCHOR_FALLBACK"}


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
    fixed_by_anchor = 0
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
                continue

            try:
                if execute_rowcount(conn, gt_sql) > 0:
                    continue

                empty_total += 1

                prefix, where_clause, suffix = get_where_clause(gt_sql)

                # If no WHERE, try anchor fallback directly
                if where_clause is None:
                    if ANCHOR_FALLBACK:
                        new_sql, diff = safe_call2(anchor_fallback_sql, conn, gt_sql, fn_name="anchor_fallback_sql")
                        if new_sql and execute_rowcount(conn, new_sql) > 0:
                            fixed_total += 1
                            fixed_by_anchor += 1
                        else:
                            errors_total += 1
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
                            "difference": {"replaced": None, "error": "No WHERE clause found"}
                        })
                    continue

                conditions = split_conditions(where_clause)

                # 1) Replace one
                new_sql, diff = safe_call2(
                    replace_one_condition_to_get_rows,
                    conn, prefix, conditions, suffix,
                    fn_name="replace_one_condition_to_get_rows"
                )
                if new_sql and execute_rowcount(conn, new_sql) > 0:
                    fixed_total += 1
                    fixed_by_1 += 1
                    out_records.append({
                        "line_number": line_number,
                        "question": question,
                        "empty_gt_sql": gt_sql,
                        "new_gt_sql": new_sql,
                        "difference": diff
                    })
                    continue

                # 2) Replace two
                if ENABLE_TWO_CONDITION_REPLACE:
                    new_sql2, diff2 = safe_call2(
                        replace_two_conditions_to_get_rows,
                        conn, prefix, conditions, suffix,
                        fn_name="replace_two_conditions_to_get_rows"
                    )
                    if new_sql2 and execute_rowcount(conn, new_sql2) > 0:
                        fixed_total += 1
                        fixed_by_2 += 1
                        out_records.append({
                            "line_number": line_number,
                            "question": question,
                            "empty_gt_sql": gt_sql,
                            "new_gt_sql": new_sql2,
                            "difference": diff2
                        })
                        continue

                # 3) Anchor fallback
                if ANCHOR_FALLBACK:
                    new_sql3, diff3 = safe_call2(anchor_fallback_sql, conn, gt_sql, fn_name="anchor_fallback_sql")
                    if new_sql3 and execute_rowcount(conn, new_sql3) > 0:
                        fixed_total += 1
                        fixed_by_anchor += 1
                        out_records.append({
                            "line_number": line_number,
                            "question": question,
                            "empty_gt_sql": gt_sql,
                            "new_gt_sql": new_sql3,
                            "difference": diff3
                        })
                        continue

                # failed (no exception, just no solution)
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
                    "difference": {
                        "replaced": None,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "stage": "main_loop"
                    }
                })

    conn.close()

    out_records = [r for r in out_records if r.get("new_gt_sql")]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print("===== EMPTY FIX V4 FIXED REPORT =====")
    print(f"Total records processed: {total}")
    print(f"Empty GT queries found: {empty_total}")
    print(f"Fixed total: {fixed_total}")
    print(f"  - Fixed by 1-condition replace: {fixed_by_1}")
    print(f"  - Fixed by 2-condition replace: {fixed_by_2}")
    print(f"  - Fixed by anchor fallback: {fixed_by_anchor}")
    print(f"Errors encountered: {errors_total}")
    print(f"Saved output JSON to: {output_json}")
    print(f"(Random seed: {RANDOM_SEED}, context keep frac: {CANDIDATE_CONTEXT_KEEP_FRAC}, max pair tries: {MAX_PAIR_TRIES})")


if __name__ == "__main__":
    main()