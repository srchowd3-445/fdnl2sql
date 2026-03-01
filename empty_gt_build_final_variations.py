#!/usr/bin/env python3
"""
Build a final 1500-item SQL dataset by combining:
1) Original queries that already return rows
2) The fixed-empty queries (390)
3) Additional structural SQL variations (remove columns / alter conditions)

Defaults are aligned to the project:
  db_path = "data/database.db"
  input_jsonl = "data/dataset.jsonl"
  table_name = "clinical_trials"

Output:
  data/empty_gt_replaced_final.json
"""

import argparse
import json
import random
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    s = strip_trailing_semicolon(sql)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def quote_ident(name: str) -> str:
    return '"' + (name or "").replace('"', '""') + '"'


def sql_string_literal(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


def split_top_level_commas(s: str) -> List[str]:
    """Split commas while respecting single quotes and parenthesis nesting."""
    out: List[str] = []
    buf: List[str] = []
    in_quote = False
    depth = 0

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]

        if ch == "'":
            # handle doubled single-quote escape
            if in_quote and i + 1 < n and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue

        if not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    out.append(part)
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out


def split_conditions(where_clause: str) -> List[str]:
    """Split on AND outside single quotes."""
    if not where_clause:
        return []

    s = where_clause.strip()
    out: List[str] = []
    buf: List[str] = []
    in_quote = False

    i = 0
    n = len(s)

    def flush() -> None:
        part = "".join(buf).strip()
        if part:
            out.append(part)

    while i < n:
        ch = s[i]

        if ch == "'":
            if in_quote and i + 1 < n and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue

        if not in_quote and i + 3 <= n and s[i:i + 3].upper() == "AND":
            left_ok = (i == 0) or (not s[i - 1].isalnum() and s[i - 1] != "_")
            right_ok = (i + 3 == n) or (not s[i + 3].isalnum() and s[i + 3] != "_")
            if left_ok and right_ok:
                flush()
                buf = []
                i += 3
                while i < n and s[i].isspace():
                    i += 1
                continue

        buf.append(ch)
        i += 1

    flush()
    return out


def parse_sql_parts(sql: str) -> Optional[Dict[str, Any]]:
    """
    Parse a query of the form:
      SELECT ... FROM ... [WHERE ...] [ORDER BY/GROUP BY/LIMIT ...]
    Returns parts needed for structural rewrites.
    """
    sql0 = strip_trailing_semicolon(sql)
    m = re.match(r"^\s*SELECT\s+(.+?)\s+FROM\s+(.+)$", sql0, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    select_part = m.group(1).strip()
    rest = m.group(2).strip()

    m_where = re.search(r"\bWHERE\b", rest, flags=re.IGNORECASE)
    if m_where:
        from_part = rest[:m_where.start()].strip()
        after_where = rest[m_where.end():].strip()

        m_suffix = re.search(r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT)\b", after_where, flags=re.IGNORECASE)
        if m_suffix:
            where_clause = after_where[:m_suffix.start()].strip()
            suffix = after_where[m_suffix.start():].strip()
        else:
            where_clause = after_where
            suffix = ""
    else:
        from_part = rest
        where_clause = ""
        suffix = ""

    select_exprs = split_top_level_commas(select_part)
    conditions = split_conditions(where_clause) if where_clause else []

    return {
        "select_exprs": select_exprs,
        "from_part": from_part,
        "conditions": conditions,
        "suffix": suffix,
    }


def build_sql(select_exprs: List[str], from_part: str, conditions: List[str], suffix: str) -> str:
    q = f"SELECT {', '.join(select_exprs)} FROM {from_part}"
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    if suffix:
        q += " " + suffix
    return q.strip() + ";"


def query_returns_rows(conn: sqlite3.Connection, sql: str) -> Tuple[bool, Optional[str]]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return False, "EMPTY_SQL"
    try:
        cur = conn.execute(f"SELECT 1 FROM ({sql0}) AS subq LIMIT 1;")
        return cur.fetchone() is not None, None
    except Exception as e:
        return False, str(e)


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["line_number"] = i
            out.append(row)
    return out


def condition_label(cond: str) -> str:
    cond = re.sub(r"\s+", " ", cond).strip()
    return cond[:120]


def make_variant_question(base_question: str, note: str) -> str:
    q = (base_question or "").strip()
    if not q:
        q = "Which clinical trials match the criteria"
    if not q.endswith("?"):
        q += "?"
    q = q.rstrip("?").strip()
    return f"{q} ({note})?"


def mutate_drop_select_column(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    cols = parts["select_exprs"]
    if len(cols) <= 1:
        return None

    idx = rng.randrange(len(cols))
    removed = cols[idx]
    new_cols = [c for i, c in enumerate(cols) if i != idx]
    sql = build_sql(new_cols, parts["from_part"], parts["conditions"], parts["suffix"])
    note = "return fewer columns"
    meta = {"removed_select": removed}
    return sql, note, meta


def mutate_drop_where_condition(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    if len(conds) < 2:
        return None

    idx = rng.randrange(len(conds))
    removed = conds[idx]
    new_conds = [c for i, c in enumerate(conds) if i != idx]
    sql = build_sql(parts["select_exprs"], parts["from_part"], new_conds, parts["suffix"])
    note = f"without condition {condition_label(removed)}"
    meta = {"removed_condition": removed}
    return sql, note, meta


def mutate_drop_two_where_conditions(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    if len(conds) < 3:
        return None

    idxs = sorted(rng.sample(range(len(conds)), 2))
    removed = [conds[i] for i in idxs]
    new_conds = [c for i, c in enumerate(conds) if i not in idxs]
    sql = build_sql(parts["select_exprs"], parts["from_part"], new_conds, parts["suffix"])
    note = "with fewer filtering conditions"
    meta = {"removed_conditions": removed}
    return sql, note, meta


def mutate_relax_numeric(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    num_candidates: List[Tuple[int, str, str, float]] = []

    for i, c in enumerate(conds):
        m = re.match(r'^\s*"([^"]+)"\s*(>=|>|<=|<|=)\s*(-?\d+(?:\.\d+)?)\s*$', c)
        if m:
            num_candidates.append((i, m.group(1), m.group(2), float(m.group(3))))

    if not num_candidates:
        return None

    i, col, op, val = rng.choice(num_candidates)
    delta = 1.0 if abs(val) <= 20 else (5.0 if abs(val) <= 100 else 10.0)

    if op in (">", ">="):
        new_op = ">="
        new_val = val - delta
    elif op in ("<", "<="):
        new_op = "<="
        new_val = val + delta
    else:  # '='
        new_op = ">="
        new_val = val - delta

    if abs(new_val - round(new_val)) < 1e-9:
        vtxt = str(int(round(new_val)))
    else:
        vtxt = f"{new_val:.3f}".rstrip("0").rstrip(".")

    new_cond = f'"{col}" {new_op} {vtxt}'
    new_conds = list(conds)
    old_cond = new_conds[i]
    new_conds[i] = new_cond

    sql = build_sql(parts["select_exprs"], parts["from_part"], new_conds, parts["suffix"])
    note = f"with a relaxed numeric threshold on {col}"
    meta = {"replaced_condition": {"from": old_cond, "to": new_cond}}
    return sql, note, meta


def mutate_remove_where(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    if not conds:
        return None

    sql = build_sql(parts["select_exprs"], parts["from_part"], [], parts["suffix"])
    note = "with broader criteria"
    meta = {"removed_where": True, "removed_condition_count": len(conds)}
    return sql, note, meta


def mutate_keep_one_where_condition(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    if len(conds) < 2:
        return None

    keep_idx = rng.randrange(len(conds))
    kept = conds[keep_idx]
    new_conds = [kept]
    sql = build_sql(parts["select_exprs"], parts["from_part"], new_conds, parts["suffix"])
    note = "using only one filter condition"
    meta = {"kept_condition": kept, "dropped_count": len(conds) - 1}
    return sql, note, meta


def mutate_keep_two_select_columns(parts: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    cols = parts["select_exprs"]
    if len(cols) < 3:
        return None

    idxs = sorted(rng.sample(range(len(cols)), 2))
    kept_cols = [cols[i] for i in idxs]
    sql = build_sql(kept_cols, parts["from_part"], parts["conditions"], parts["suffix"])
    note = "return only two key columns"
    meta = {"kept_select": kept_cols, "dropped_count": len(cols) - 2}
    return sql, note, meta


def mutate_swap_text_equality_value(
    conn: sqlite3.Connection,
    table_name: str,
    parts: Dict[str, Any],
    rng: random.Random,
    value_cache: Dict[str, List[str]],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    conds = parts["conditions"]
    candidates: List[Tuple[int, str, str]] = []

    for i, c in enumerate(conds):
        m = re.match(r'^\s*"([^"]+)"\s*=\s*\'([^\']*)\'\s*$', c)
        if m and m.group(2).strip():
            candidates.append((i, m.group(1), m.group(2)))

    if not candidates:
        return None

    rng.shuffle(candidates)
    qtable = quote_ident(table_name)
    for idx, col, old_val in candidates:
        if col not in value_cache:
            qcol = quote_ident(col)
            try:
                cur = conn.execute(
                    f"SELECT DISTINCT {qcol} FROM {qtable} "
                    f"WHERE {qcol} IS NOT NULL LIMIT 1000;"
                )
                vals = [str(r[0]) for r in cur.fetchall() if str(r[0]).strip()]
            except Exception:
                vals = []
            value_cache[col] = vals

        alternatives = [v for v in value_cache[col] if v != old_val]
        if not alternatives:
            continue

        new_val = rng.choice(alternatives)
        new_cond = f'{quote_ident(col)} = {sql_string_literal(new_val)}'
        new_conds = list(conds)
        old_cond = new_conds[idx]
        new_conds[idx] = new_cond

        short_val = new_val if len(new_val) <= 40 else (new_val[:37] + "...")
        note = f'with a different {col} value ({short_val})'
        meta = {"replaced_condition": {"from": old_cond, "to": new_cond}}
        sql = build_sql(parts["select_exprs"], parts["from_part"], new_conds, parts["suffix"])
        return sql, note, meta

    return None


def build_base_records(
    conn: sqlite3.Connection,
    dataset_rows: List[Dict[str, Any]],
    fixed_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    fixed_by_line = {int(r["line_number"]): r for r in fixed_rows if "line_number" in r}

    base_records: List[Dict[str, Any]] = []
    counts = {
        "original_working": 0,
        "fixed_working": 0,
        "skipped_original_not_working": 0,
        "skipped_fixed_not_working": 0,
    }

    for row in dataset_rows:
        line_number = int(row["line_number"])
        question = row.get("question", "")

        if line_number in fixed_by_line:
            fr = fixed_by_line[line_number]
            sql = fr.get("new_gt_sql", "")
            ok, err = query_returns_rows(conn, sql)
            if not ok:
                counts["skipped_fixed_not_working"] += 1
                continue

            base_records.append({
                "item_id": f"line_{line_number}",
                "line_number": line_number,
                "question": question,
                "gt_sql": sql,
                "source": "fixed_empty",
                "variation_type": "none",
                "variation_note": "",
                "row_exists": True,
                "metadata": {
                    "difference": fr.get("difference"),
                    "empty_gt_sql": fr.get("empty_gt_sql"),
                    "error": err,
                },
            })
            counts["fixed_working"] += 1
        else:
            sql = row.get("gt_sql", "")
            ok, err = query_returns_rows(conn, sql)
            if not ok:
                counts["skipped_original_not_working"] += 1
                continue

            base_records.append({
                "item_id": f"line_{line_number}",
                "line_number": line_number,
                "question": question,
                "gt_sql": sql,
                "source": "original_working",
                "variation_type": "none",
                "variation_note": "",
                "row_exists": True,
                "metadata": {
                    "error": err,
                },
            })
            counts["original_working"] += 1

    return base_records, counts


def generate_variations(
    conn: sqlite3.Connection,
    table_name: str,
    base_records: List[Dict[str, Any]],
    target_total: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)

    seen_sql = {normalize_sql(r["gt_sql"]) for r in base_records if r.get("gt_sql")}
    out = list(base_records)

    value_cache: Dict[str, List[str]] = {}

    variation_fns = [
        ("drop_select_column", mutate_drop_select_column),
        ("keep_two_select_columns", mutate_keep_two_select_columns),
        ("drop_where_condition", mutate_drop_where_condition),
        ("drop_two_where_conditions", mutate_drop_two_where_conditions),
        ("keep_one_where_condition", mutate_keep_one_where_condition),
        ("relax_numeric", mutate_relax_numeric),
        ("swap_text_equality_value", lambda p, r: mutate_swap_text_equality_value(conn, table_name, p, r, value_cache)),
        ("remove_where", mutate_remove_where),
    ]

    stats = {
        "target_total": target_total,
        "added_variations": 0,
        "attempts": 0,
        "duplicates": 0,
        "invalid_sql": 0,
        "non_working_sql": 0,
    }
    per_type = {name: 0 for name, _ in variation_fns}

    max_attempts = max(5000, target_total * 80)

    while len(out) < target_total and stats["attempts"] < max_attempts:
        stats["attempts"] += 1

        base = rng.choice(base_records)
        parts = parse_sql_parts(base["gt_sql"])
        if not parts:
            continue

        vname, vfn = rng.choice(variation_fns)
        res = vfn(parts, rng)
        if not res:
            continue

        new_sql, note, vmeta = res
        norm = normalize_sql(new_sql)
        if not norm or norm in seen_sql:
            stats["duplicates"] += 1
            continue

        ok, err = query_returns_rows(conn, new_sql)
        if not ok:
            stats["non_working_sql"] += 1
            continue

        seen_sql.add(norm)
        per_type[vname] += 1

        var_idx = per_type[vname]
        rec = {
            "item_id": f"line_{base['line_number']}_var_{vname}_{var_idx}",
            "line_number": base["line_number"],
            "question": make_variant_question(base.get("question", ""), note),
            "gt_sql": new_sql,
            "source": "variation",
            "base_source": base.get("source", ""),
            "variation_type": vname,
            "variation_note": note,
            "row_exists": True,
            "metadata": {
                "parent_item_id": base.get("item_id"),
                "parent_sql": base.get("gt_sql"),
                "mutation": vmeta,
                "error": err,
            },
        }
        out.append(rec)
        stats["added_variations"] += 1

    stats.update({f"variation_{k}": v for k, v in per_type.items()})
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", default="data/database.db")
    ap.add_argument("--input_jsonl", default="data/dataset.jsonl")
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--fixed_json", default="data/empty_gt_replaced_v5_fixed.json")
    ap.add_argument("--output_json", default="data/empty_gt_replaced_final.json")
    ap.add_argument("--target_total", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db_path)

    dataset_rows = load_dataset_jsonl(args.input_jsonl)
    with open(args.fixed_json, "r", encoding="utf-8") as f:
        fixed_rows = json.load(f)

    base_records, base_stats = build_base_records(conn, dataset_rows, fixed_rows)
    final_records, var_stats = generate_variations(
        conn=conn,
        table_name=args.table_name,
        base_records=base_records,
        target_total=args.target_total,
        seed=args.seed,
    )

    conn.close()

    # deterministic order: base first by line, then variations by item_id
    base_part = [r for r in final_records if r.get("source") != "variation"]
    var_part = [r for r in final_records if r.get("source") == "variation"]
    base_part.sort(key=lambda r: (int(r.get("line_number", 0)), r.get("source", "")))
    var_part.sort(key=lambda r: r.get("item_id", ""))
    final_records = base_part + var_part

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    print("\n================ FINAL BUILD SUMMARY ================")
    print(f"Input dataset rows:              {len(dataset_rows)}")
    print(f"Fixed rows provided:             {len(fixed_rows)}")
    print(f"Base original_working:           {base_stats['original_working']}")
    print(f"Base fixed_working:              {base_stats['fixed_working']}")
    print(f"Skipped original not working:    {base_stats['skipped_original_not_working']}")
    print(f"Skipped fixed not working:       {base_stats['skipped_fixed_not_working']}")
    print(f"Base total included:             {len(base_records)}")
    print(f"Target total:                    {var_stats['target_total']}")
    print(f"Added variations:                {var_stats['added_variations']}")
    print(f"Variation attempts:              {var_stats['attempts']}")
    print(f"Variation duplicates skipped:    {var_stats['duplicates']}")
    print(f"Variation non-working skipped:   {var_stats['non_working_sql']}")
    print(f"Variation by type: drop_select_column={var_stats['variation_drop_select_column']}, "
          f"keep_two_select_columns={var_stats['variation_keep_two_select_columns']}, "
          f"drop_where_condition={var_stats['variation_drop_where_condition']}, "
          f"drop_two_where_conditions={var_stats['variation_drop_two_where_conditions']}, "
          f"keep_one_where_condition={var_stats['variation_keep_one_where_condition']}, "
          f"relax_numeric={var_stats['variation_relax_numeric']}, "
          f"swap_text_equality_value={var_stats['variation_swap_text_equality_value']}, "
          f"remove_where={var_stats['variation_remove_where']}")
    print(f"Final output records:            {len(final_records)}")
    print(f"Output JSON:                     {args.output_json}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
