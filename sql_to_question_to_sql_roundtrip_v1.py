#!/usr/bin/env python3
"""
sql_to_question_to_sql_roundtrip_v3.py

Flip-the-script pipeline: SQL -> question -> SQL -> eval.

Adds the “plumbing fixes”:
- SQL->Question prompt tightened (one-line output, no "Question:" prefixes)
- sanitize_generated_question() to drop junk outputs ("Question:", "---", "Allowed column names...")
- extract_sql() robust to prose before SQL and code fences; extracts from first SELECT onward
- build_question_to_sql_prompt(): defines core_inline; enforces quoting rule explicitly
- allowed_cols_per_item + auto_quote_allowed_columns() to repair missing quotes on spaced column names
- bump default q_max_tokens to 120 (you can override)

Output:
- JSON LIST to --output_json (default: empty_gt_fixed_v7.json)
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

try:
    from litellm import completion as litellm_completion
except Exception:
    litellm_completion = None

# pip install sqlglot
import sqlglot
from sqlglot import exp


# -------------------------
# Helpers: SQL normalization / schema / execution
# -------------------------
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    s = strip_trailing_semicolon(sql)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*=\s*", "=", s)
    s = re.sub(r"\s*>=\s*", ">=", s)
    s = re.sub(r"\s*<=\s*", "<=", s)
    s = re.sub(r"\s*>\s*", ">", s)
    s = re.sub(r"\s*<\s*", "<", s)
    return s


def fetch_schema(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}");')
    rows = cur.fetchall()
    return [r[1] for r in rows]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def get_table_rowcount(conn: sqlite3.Connection, table: str) -> Optional[int]:
    try:
        cur = conn.execute(f'SELECT COUNT(*) FROM "{table}";')
        return int(cur.fetchone()[0])
    except Exception:
        return None


def fetch_sample_rows(conn: sqlite3.Connection, table: str, limit: int = 3) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    try:
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)};')
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return cols, rows
    except Exception:
        return [], []


def print_schema_sanity(
    conn: sqlite3.Connection,
    db_path: str,
    table: str,
    schema_cols: List[str],
    sample_rows: int = 3,
    show_cols: int = 40,
) -> None:
    print("\n================ SCHEMA SANITY CHECK ================")
    print(f"DB:    {db_path}")
    print(f"Table: {table}")
    print(f"Table exists: {table_exists(conn, table)}")
    print(f"Num columns: {len(schema_cols)}")
    if schema_cols:
        head = schema_cols[: min(show_cols, len(schema_cols))]
        print(f"First {len(head)} columns:")
        for c in head:
            print(f"  - {c}")
        if len(schema_cols) > show_cols:
            print(f"... ({len(schema_cols) - show_cols} more columns not shown)")
    rc = get_table_rowcount(conn, table)
    if rc is not None:
        print(f"Row count: {rc}")
    cols, rows = fetch_sample_rows(conn, table, limit=sample_rows)
    if cols and rows:
        max_show = min(10, len(cols))
        print(f"Sample rows: {len(rows)} (showing {min(sample_rows, len(rows))})")
        print(f"Sample row keys (first {max_show} cols): {cols[:max_show]}")
        r0 = rows[0]
        print("Sample row[0] preview:")
        for i in range(max_show):
            v = r0[i]
            s = str(v)
            if len(s) > 120:
                s = s[:120] + "..."
            print(f"  {cols[i]} = {s}")
    else:
        print("Could not fetch sample rows (table empty or query failed).")
    print("=====================================================\n")


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int = 200) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


def execute_sql_preview(conn: sqlite3.Connection, sql: str, max_rows: int = 3) -> List[Dict[str, Any]]:
    cols, rows = execute_sql_fetch(conn, sql, max_rows=max_rows)
    return [dict(zip(cols, r)) for r in rows]


def canonicalize_result(cols: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d: Dict[str, Any] = {}
        for c, v in zip(cols, r):
            if isinstance(v, (int, float, str)) or v is None:
                d[c] = v
            else:
                d[c] = str(v)
        out.append(d)
    return out


def results_match_strict(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    if cols_a != cols_b:
        return False, "COL_MISMATCH"

    can_a = canonicalize_result(cols_a, rows_a)
    can_b = canonicalize_result(cols_b, rows_b)
    return can_a == can_b, None


def results_match_loose(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    common = [c for c in cols_a if c in cols_b]
    if not common:
        return False, "NO_COMMON_COLS"

    idx_a = [cols_a.index(c) for c in common]
    idx_b = [cols_b.index(c) for c in common]

    proj_a = [tuple(r[i] for i in idx_a) for r in rows_a]
    proj_b = [tuple(r[i] for i in idx_b) for r in rows_b]
    return proj_a == proj_b, None


# -------------------------
# AST eval (full + root decompositions)
# -------------------------
def canonicalize_sql_ast(sql: str, dialect: str = "sqlite") -> Optional[dict]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    return tree.dump()


def ast_match_sql(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        a = canonicalize_sql_ast(sql_a, dialect=dialect)
        b = canonicalize_sql_ast(sql_b, dialect=dialect)
        if a is None or b is None:
            return False, "EMPTY_SQL"
        return a == b, None
    except Exception as e:
        return False, f"AST_ERROR: {e}"


def select_projection_set(sql: str, dialect: str = "sqlite") -> Optional[List[str]]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    sel = tree.args.get("expressions") or []
    cols = []
    for e in sel:
        cols.append(re.sub(r"\s+", " ", e.sql(dialect=dialect, pretty=False)).strip())
    return sorted(cols)


def projection_match_set(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        pa = select_projection_set(sql_a, dialect=dialect)
        pb = select_projection_set(sql_b, dialect=dialect)
        if pa is None or pb is None:
            return False, "EMPTY_SQL"
        return pa == pb, None
    except Exception as e:
        return False, f"PROJ_ERROR: {e}"


def from_clause(sql: str, dialect: str = "sqlite") -> Optional[str]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    frm = tree.args.get("from") or tree.args.get("from_")
    return frm.sql(dialect=dialect, pretty=False) if frm else None


def from_match(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        fa = from_clause(sql_a, dialect=dialect)
        fb = from_clause(sql_b, dialect=dialect)
        if fa is None or fb is None:
            return False, "EMPTY_SQL"
        return fa == fb, None
    except Exception as e:
        return False, f"FROM_ERROR: {e}"


# WHERE match fingerprint (AND-order-insensitive)
def flatten_and(expr_: exp.Expression) -> List[exp.Expression]:
    if isinstance(expr_, exp.And):
        return flatten_and(expr_.left) + flatten_and(expr_.right)
    return [expr_]


def where_clause_fingerprint(sql: str, dialect: str = "sqlite") -> Optional[List[str]]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    where = tree.args.get("where")
    if not where or not where.this:
        return []
    parts = flatten_and(where.this)
    norm = [p.sql(dialect=dialect, pretty=False) for p in parts]
    norm = [re.sub(r"\s+", " ", s).strip() for s in norm]
    return sorted(norm)


def where_match_commutative(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        fa = where_clause_fingerprint(sql_a, dialect=dialect)
        fb = where_clause_fingerprint(sql_b, dialect=dialect)
        if fa is None or fb is None:
            return False, "EMPTY_SQL"
        return fa == fb, None
    except Exception as e:
        return False, f"WHERE_FINGERPRINT_ERROR: {e}"


def set_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return (len(sa & sb) / len(u)) if u else 0.0


def ast_relaxed_components(
    sql_a: str,
    sql_b: str,
    dialect: str = "sqlite",
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Less-strict AST similarity using root-level components:
    - projection_jaccard: SELECT projection overlap as set similarity
    - where_jaccard: AND-flattened WHERE overlap as set similarity
    - from_sim: 1 if FROM clause matches else 0
    Weighted score: 0.40 * projection + 0.50 * where + 0.10 * from
    """
    try:
        pa = select_projection_set(sql_a, dialect=dialect)
        pb = select_projection_set(sql_b, dialect=dialect)
        wa = where_clause_fingerprint(sql_a, dialect=dialect)
        wb = where_clause_fingerprint(sql_b, dialect=dialect)
        fa = from_clause(sql_a, dialect=dialect)
        fb = from_clause(sql_b, dialect=dialect)

        if pa is None or pb is None or wa is None or wb is None or fa is None or fb is None:
            return None, "EMPTY_SQL"

        proj_j = set_jaccard(pa, pb)
        where_j = set_jaccard(wa, wb)
        from_sim = 1.0 if fa == fb else 0.0

        score = 0.50 * proj_j + 0.40 * where_j + 0.10 * from_sim
        return {
            "score": score,
            "projection_jaccard": proj_j,
            "where_jaccard": where_j,
            "from_sim": from_sim,
        }, None
    except Exception as e:
        return None, f"AST_RELAXED_ERROR: {e}"


# -------------------------
# Prompt builders + sanitizers
# -------------------------
def build_sql_to_question_prompt(new_sql: str, preview_rows: List[Dict[str, Any]]) -> str:
    preview_json = json.dumps(preview_rows, ensure_ascii=False, indent=2)
    return f"""Write ONE natural-language question that the following SQL answers.

Rules:
- Output ONLY the question text on ONE line.
- Do NOT include prefixes like "Question:" and do NOT use quotes.
- Do NOT mention SQL, tables, columns, databases, or "query".
- The question must be specific enough that this SQL would be the correct answer.

SQL:
{new_sql}

Example rows:
{preview_json}
"""


def sanitize_generated_question(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r'^(question\s*:)\s*', '', q, flags=re.IGNORECASE).strip()
    q = q.strip('"').strip("'").strip()
    if q in {"---", "-", ""}:
        return ""
    if q.lower().startswith("allowed column"):
        return ""
    return q


def extract_first_line(text: str) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0].strip()


def extract_sql(text: str) -> str:
    """
    Robustly extract SQL:
    - Strip fences
    - If prose exists, keep from first SELECT onward
    - If model repeats queries, keep only the first top-level SELECT block
    - Keep first statement
    """
    if not text:
        return ""
    t = text.strip()

    # Drop opening fence like ```sql or ```sqlite or ```anything
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    # Drop closing fence
    t = re.sub(r"\s*```$", "", t).strip()

    # If model included prose, keep from first SELECT onward
    m = re.search(r"\bSELECT\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():].strip()

    # If model emitted another top-level SELECT, keep only the first query block.
    m2 = re.search(r"\n\s*SELECT\b", t, flags=re.IGNORECASE)
    if m2:
        t = t[:m2.start()].strip()

    # Keep first statement
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"

    return t.strip()


def extract_identifiers_from_sql(sql: str) -> List[str]:
    # Works for your quoted-column style: "Column name"
    return sorted(set(re.findall(r'"([^"]+)"', sql or "")))


def has_where_clause(sql: str, dialect: str = "sqlite") -> Optional[bool]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    try:
        tree = sqlglot.parse_one(sql0, read=dialect)
    except Exception:
        return None
    where = tree.args.get("where")
    return bool(where and where.this)


def sql_has_parse_error(sql: str, dialect: str = "sqlite") -> bool:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return True
    try:
        sqlglot.parse_one(sql0, read=dialect)
        return False
    except Exception:
        return True


def should_retry_sql_prediction(
    pred_sql: str,
    hint_sql: str,
    *,
    retry_on_missing_where: bool = True,
    retry_on_parse_error: bool = True,
    retry_on_multi_select: bool = True,
    dialect: str = "sqlite",
) -> bool:
    if not (pred_sql or "").strip():
        return True

    if retry_on_multi_select and re.search(r"\n\s*SELECT\b", pred_sql, flags=re.IGNORECASE):
        return True

    if retry_on_parse_error and sql_has_parse_error(pred_sql, dialect=dialect):
        return True

    if retry_on_missing_where:
        hint_has_where = has_where_clause(hint_sql, dialect=dialect)
        pred_has_where = has_where_clause(pred_sql, dialect=dialect)
        if hint_has_where is True and pred_has_where is False:
            return True

    return False


def auto_quote_allowed_columns(sql: str, allowed_cols: List[str]) -> str:
    """
    Repair missing quotes for column names, especially those with spaces (e.g. Cancer type).
    Only attempts to quote columns in allowed_cols. Conservative replacement.
    """
    if not sql:
        return sql
    fixed = sql
    for col in sorted(allowed_cols, key=len, reverse=True):
        quoted = f'"{col}"'
        if quoted in fixed:
            continue
        pattern = r'(?<!")\b' + re.escape(col) + r'\b(?!")'
        fixed = re.sub(pattern, quoted, fixed)
    return fixed


def build_question_to_sql_prompt(
    schema_cols: List[str],
    question: str,
    table: str,
    hint_sql: str,
    core_cols: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Stronger + shorter schema:
    - Allowed columns = columns used in hint_sql + core context columns.
    - Schema is inlined to avoid huge prompts and EOS-with-empty-output issues.
    Returns: (prompt, allowed_cols)
    """
    if core_cols is None:
        core_cols = ["NCT", "Author", "Year", "Cancer type", "Trial phase"]

    used = set(extract_identifiers_from_sql(hint_sql))
    core = set(core_cols)

    allowed = [c for c in schema_cols if (c in used or c in core)]
    if not allowed:
        allowed = list(schema_cols)

    schema_inline = ", ".join([f'"{c}"' for c in allowed])
    core_inline = ", ".join([f'"{c}"' for c in core_cols])

    prompt = f"""You are a SQL generator. Write one SQLite SELECT query over "{table}".

Rules:
- Output ONLY the SQL query (no prose, no code fences).
- The query MUST start with SELECT.
- Use ONLY column names from this allowed list: {schema_inline}
- IMPORTANT: Put double quotes around EVERY column name exactly as shown in the allowed list.
- Use single quotes for string literals.
- For categorical filters, copy exact values as they appear in the data (match capitalization).
- If the question includes any constraints (entity/value/time/phase/etc.), include ALL of them in WHERE.
- Do NOT add filters, values, years, authors, phases, or trial names that are not explicitly in the question.
- SELECT: include the columns needed to answer the question, plus these context columns if present: {core_inline}

Question:
{question}
"""
    return prompt, allowed


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _litellm_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if t is None:
                    t = item.get("content")
                if t is None:
                    t = item.get("value")
                if t is not None:
                    parts.append(str(t))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def generate_batch_with_backend(
    backend: str,
    prompts: List[str],
    llm: Any,
    sampling: Any,
    *,
    litellm_model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    litellm_timeout: float,
    litellm_num_retries: int,
    vertex_project: str,
    vertex_location: str,
    stop: Optional[List[str]] = None,
) -> Tuple[List[str], List[Optional[Dict[str, Any]]]]:
    texts: List[str] = []
    metas: List[Optional[Dict[str, Any]]] = []

    if backend == "vllm":
        outs = llm.generate(prompts, sampling)
        for out in outs:
            gen_text = (out.outputs[0].text or "").strip() if out.outputs else ""
            texts.append(gen_text)
            try:
                meta = {
                    "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                    "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                    "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                } if out.outputs else None
            except Exception:
                meta = None
            metas.append(meta)
        return texts, metas

    if litellm_completion is None:
        raise RuntimeError("litellm is not installed. Install with: pip install litellm")

    for prompt in prompts:
        try:
            req: Dict[str, Any] = {
                "model": litellm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "num_retries": litellm_num_retries,
            }
            if max_tokens is not None and max_tokens > 0:
                req["max_tokens"] = int(max_tokens)
            if litellm_timeout and litellm_timeout > 0:
                req["timeout"] = litellm_timeout
            if stop:
                req["stop"] = stop
            if vertex_project:
                req["vertex_project"] = vertex_project
            if vertex_location:
                req["vertex_location"] = vertex_location

            resp = litellm_completion(**req)

            choices = _obj_get(resp, "choices", []) or []
            choice0 = choices[0] if choices else {}
            message = _obj_get(choice0, "message", {}) or {}
            content = _obj_get(message, "content", "")
            gen_text = _litellm_content_to_text(content).strip()

            usage = _obj_get(resp, "usage", {}) or {}
            completion_tokens = _obj_get(usage, "completion_tokens", None)

            texts.append(gen_text)
            metas.append({
                "finish_reason": _obj_get(choice0, "finish_reason", None),
                "stop_reason": _obj_get(choice0, "stop_reason", None),
                "token_ids_len": completion_tokens,
            })
        except Exception as e:
            texts.append("")
            metas.append({
                "finish_reason": "error",
                "stop_reason": None,
                "token_ids_len": None,
                "error": str(e),
            })

    return texts, metas


# -------------------------
# ID
# -------------------------
def compute_item_id(record: Dict[str, Any]) -> str:
    if "line_number" in record and record["line_number"] is not None:
        return f"line_{record['line_number']}"
    s = (record.get("new_gt_sql") or "").strip()
    q = (record.get("question") or "").strip()
    h = hashlib.sha1((s + "\n" + q).encode("utf-8")).hexdigest()[:12]
    return f"hash_{h}"


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to JSON list input")
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--output_json", default="empty_gt_fixed_v7.json")

    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
        help="Local HF snapshot directory for Gemma-3-27b-it",
    )

    ap.add_argument("--llm_backend", choices=["vllm", "litellm"], default="vllm")
    ap.add_argument("--litellm_model", default="vertex_ai/gemini-2.5-flash")
    ap.add_argument("--vertex_project", default="", help="GCP project for Vertex AI (or set VERTEX_PROJECT)")
    ap.add_argument("--vertex_location", default="", help="Vertex location (or set VERTEX_LOCATION)")
    ap.add_argument("--litellm_timeout", type=float, default=120.0)
    ap.add_argument("--litellm_num_retries", type=int, default=2)

    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all")

    # SQL -> question params
    ap.add_argument("--q_max_tokens", type=int, default=0, help="Stage-1 question max tokens. <=0 disables cap for litellm; vllm falls back to 256.")
    ap.add_argument("--q_temperature", type=float, default=0.2)
    ap.add_argument("--q_top_p", type=float, default=0.9)

    # question -> SQL params
    ap.add_argument("--sql_max_tokens", type=int, default=512)
    ap.add_argument("--sql_temperature", type=float, default=0.0)
    ap.add_argument("--sql_top_p", type=float, default=1.0)
    ap.add_argument("--sql_min_tokens", type=int, default=16)
    ap.add_argument("--sql_retry_attempts", type=int, default=1)
    ap.add_argument("--sql_retry_temperature", type=float, default=0.2)
    ap.add_argument("--sql_retry_top_p", type=float, default=0.95)
    ap.add_argument("--sql_use_stop_semicolon", type=int, default=0, help="1=pass stop=[';'] to API, 0=disabled")
    ap.add_argument("--sql_retry_on_missing_where", type=int, default=1, help="Retry if gold has WHERE but prediction does not")
    ap.add_argument("--sql_retry_on_parse_error", type=int, default=1, help="Retry when generated SQL fails parsing")
    ap.add_argument("--sql_retry_on_multi_select", type=int, default=1, help="Retry when generation contains repeated top-level SELECT")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_rows_compare", type=int, default=200)
    ap.add_argument("--preview_rows", type=int, default=3)

    # Combined scoring controls
    ap.add_argument("--relaxed_overall_threshold", type=float, default=0.60)
    ap.add_argument("--relaxed_weight_ast", type=float, default=0.70)
    ap.add_argument("--relaxed_weight_exec_loose", type=float, default=0.30)

    ap.add_argument("--schema_sanity_only", action="store_true")

    args = ap.parse_args()

    if args.llm_backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.vertex_project:
        os.environ.setdefault("VERTEX_PROJECT", args.vertex_project)
    else:
        args.vertex_project = os.getenv("VERTEX_PROJECT", "")

    if args.vertex_location:
        os.environ.setdefault("VERTEX_LOCATION", args.vertex_location)
    else:
        args.vertex_location = os.getenv("VERTEX_LOCATION", "")

    # DB + schema sanity
    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    print_schema_sanity(conn, args.db_path, args.table_name, schema_cols, sample_rows=3, show_cols=40)
    if args.schema_sanity_only:
        conn.close()
        return
    if not schema_cols:
        conn.close()
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")

    # Load input JSON list
    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        conn.close()
        raise ValueError("Input must be a JSON LIST of records.")

    # Filter + slice
    eligible = [r for r in records if r.get("new_gt_sql")]
    if args.limit is not None and args.limit > -1:
        eligible = eligible[args.start : args.start + args.limit]
    else:
        eligible = eligible[args.start :]

    if not eligible:
        conn.close()
        print("No eligible items (missing new_gt_sql).")
        return

    # Prepare LLM backend
    llm: Any = None
    q_sampling: Any = None
    sql_sampling: Any = None
    sql_retry_sampling: Any = None

    if args.llm_backend == "vllm":
        if LLM is None or SamplingParams is None:
            conn.close()
            raise RuntimeError("vllm is not installed/importable. Install vllm or use --llm_backend litellm.")

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

        q_max_tokens_vllm = args.q_max_tokens if args.q_max_tokens and args.q_max_tokens > 0 else 256
        q_sampling = SamplingParams(
            max_tokens=q_max_tokens_vllm,
            temperature=args.q_temperature,
            top_p=args.q_top_p,
        )
        sql_sampling_kwargs = dict(
            max_tokens=args.sql_max_tokens,
            temperature=args.sql_temperature,
            top_p=args.sql_top_p,
        )
        if args.sql_min_tokens and args.sql_min_tokens > 0:
            sql_sampling_kwargs["min_tokens"] = args.sql_min_tokens
        try:
            sql_sampling = SamplingParams(**sql_sampling_kwargs)
        except TypeError:
            sql_sampling_kwargs.pop("min_tokens", None)
            sql_sampling = SamplingParams(**sql_sampling_kwargs)

        retry_sampling_kwargs = dict(
            max_tokens=args.sql_max_tokens,
            temperature=args.sql_retry_temperature,
            top_p=args.sql_retry_top_p,
        )
        if args.sql_min_tokens and args.sql_min_tokens > 0:
            retry_sampling_kwargs["min_tokens"] = args.sql_min_tokens
        try:
            sql_retry_sampling = SamplingParams(**retry_sampling_kwargs)
        except TypeError:
            retry_sampling_kwargs.pop("min_tokens", None)
            sql_retry_sampling = SamplingParams(**retry_sampling_kwargs)
    else:
        if litellm_completion is None:
            conn.close()
            raise RuntimeError("litellm is not installed/importable. Install with: pip install litellm")
        if args.litellm_model.startswith("vertex_ai/") and not args.vertex_project:
            conn.close()
            raise RuntimeError(
                "For Vertex models set --vertex_project (or VERTEX_PROJECT). "
                "Example: --litellm_model vertex_ai/gemini-2.5-flash --vertex_project <project-id>"
            )

    sql_stop_tokens = [";"] if bool(args.sql_use_stop_semicolon) else None

    # Prepare items + SQL previews
    items: List[Dict[str, Any]] = []
    previews: List[List[Dict[str, Any]]] = []

    for r in eligible:
        item_id = compute_item_id(r)
        new_sql = r.get("new_gt_sql", "")
        original_question = r.get("question", "")
        original_sql = r.get("empty_gt_sql", "") or r.get("original_sql", "")

        try:
            preview = execute_sql_preview(conn, new_sql, max_rows=args.preview_rows)
        except Exception as e:
            preview = [{"SQL_ERROR": str(e)}]

        items.append({
            "item_id": item_id,
            "original_question": original_question,
            "original_sql": original_sql,
            "new_sql": new_sql,
        })
        previews.append(preview)

    # Stage 1: SQL -> question prompts
    q_prompts: List[str] = [build_sql_to_question_prompt(it["new_sql"], prev) for it, prev in zip(items, previews)]

    generated_questions: List[str] = [""] * len(items)
    q_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    q_stage1_max_tokens: Optional[int]
    if args.q_max_tokens and args.q_max_tokens > 0:
        q_stage1_max_tokens = int(args.q_max_tokens)
    elif args.llm_backend == "litellm":
        q_stage1_max_tokens = None
    else:
        q_stage1_max_tokens = 256

    for b0 in range(0, len(items), args.batch_size):
        batch_prompts = q_prompts[b0 : b0 + args.batch_size]
        batch_texts, batch_meta = generate_batch_with_backend(
            args.llm_backend,
            batch_prompts,
            llm,
            q_sampling,
            litellm_model=args.litellm_model,
            temperature=args.q_temperature,
            top_p=args.q_top_p,
            max_tokens=q_stage1_max_tokens,
            litellm_timeout=args.litellm_timeout,
            litellm_num_retries=args.litellm_num_retries,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            stop=None,
        )
        for i, gen_text in enumerate(batch_texts):
            idx = b0 + i
            q1 = sanitize_generated_question(extract_first_line(gen_text))
            if not q1:
                q1 = "Which clinical trials match the given criteria?"
            generated_questions[idx] = q1
            q_meta[idx] = batch_meta[i] if i < len(batch_meta) else None

    # Stage 2: question -> SQL prompts (filtered schema + stronger SELECT/quoting rule)
    sql_prompts: List[str] = []
    allowed_cols_per_item: List[List[str]] = []
    for it, q in zip(items, generated_questions):
        p, allowed_cols = build_question_to_sql_prompt(schema_cols, q, args.table_name, hint_sql=it["new_sql"])
        sql_prompts.append(p)
        allowed_cols_per_item.append(allowed_cols)

    pred_sqls: List[str] = [""] * len(items)
    raw_sql_outputs: List[str] = [""] * len(items)
    sql_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    for b0 in range(0, len(items), args.batch_size):
        batch_prompts = sql_prompts[b0 : b0 + args.batch_size]
        batch_texts, batch_meta = generate_batch_with_backend(
            args.llm_backend,
            batch_prompts,
            llm,
            sql_sampling,
            litellm_model=args.litellm_model,
            temperature=args.sql_temperature,
            top_p=args.sql_top_p,
            max_tokens=args.sql_max_tokens,
            litellm_timeout=args.litellm_timeout,
            litellm_num_retries=args.litellm_num_retries,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            stop=sql_stop_tokens,
        )
        for i, gen_text in enumerate(batch_texts):
            idx = b0 + i
            raw_sql_outputs[idx] = gen_text

            pred = extract_sql(gen_text)
            if not pred and gen_text and "SELECT" not in gen_text.upper():
                pred = extract_sql("SELECT " + gen_text)
            pred = auto_quote_allowed_columns(pred, allowed_cols_per_item[idx])
            pred_sqls[idx] = pred

            sql_meta[idx] = batch_meta[i] if i < len(batch_meta) else None

    # Retry empty SQL generations with a stronger start cue.
    for _ in range(max(0, args.sql_retry_attempts)):
        retry_indices = [
            i for i, p in enumerate(pred_sqls)
            if should_retry_sql_prediction(
                p,
                items[i]["new_sql"],
                retry_on_missing_where=bool(args.sql_retry_on_missing_where),
                retry_on_parse_error=bool(args.sql_retry_on_parse_error),
                retry_on_multi_select=bool(args.sql_retry_on_multi_select),
                dialect="sqlite",
            )
        ]
        if not retry_indices:
            break

        retry_prompts = []
        for i in retry_indices:
            extra = [
                "Start directly with SELECT and output only SQL.",
                "Do not add any filters that are not explicitly asked in the question.",
            ]
            if bool(args.sql_retry_on_missing_where) and has_where_clause(items[i]["new_sql"], dialect="sqlite") is True:
                extra.append("Include a WHERE clause with every constraint from the question.")
            retry_prompts.append(sql_prompts[i] + "\n" + "\n".join(extra) + "\nSELECT")

        for b0 in range(0, len(retry_indices), args.batch_size):
            batch_indices = retry_indices[b0 : b0 + args.batch_size]
            batch_prompts = retry_prompts[b0 : b0 + args.batch_size]
            batch_texts, batch_meta = generate_batch_with_backend(
                args.llm_backend,
                batch_prompts,
                llm,
                sql_retry_sampling,
                litellm_model=args.litellm_model,
                temperature=args.sql_retry_temperature,
                top_p=args.sql_retry_top_p,
                max_tokens=args.sql_max_tokens,
                litellm_timeout=args.litellm_timeout,
                litellm_num_retries=args.litellm_num_retries,
                vertex_project=args.vertex_project,
                vertex_location=args.vertex_location,
                stop=sql_stop_tokens,
            )

            for j, gen_text in enumerate(batch_texts):
                idx = batch_indices[j]

                pred = extract_sql(gen_text)
                if not pred and gen_text and "SELECT" not in gen_text.upper():
                    pred = extract_sql("SELECT " + gen_text)
                pred = auto_quote_allowed_columns(pred, allowed_cols_per_item[idx])

                if pred:
                    pred_sqls[idx] = pred
                    raw_sql_outputs[idx] = gen_text
                    sql_meta[idx] = batch_meta[j] if j < len(batch_meta) else None

    # Evaluation + output rows
    out_rows: List[Dict[str, Any]] = []

    total = 0
    exact_yes = 0
    ast_yes = 0
    exec_strict_yes = 0
    exec_loose_yes = 0

    where_yes = 0
    proj_yes = 0
    from_yes = 0

    empty_pred_cnt = 0
    exec_err_cnt = 0
    ast_err_cnt = 0
    gt_self_ast_yes = 0
    gt_self_ast_err_cnt = 0

    overall_strict_yes = 0
    relaxed_overall_sum = 0.0
    relaxed_overall_yes = 0

    ast_relaxed_sum = 0.0
    ast_relaxed_err_cnt = 0

    for i, it in enumerate(items):
        total += 1
        new_sql = it["new_sql"]
        pred_sql = pred_sqls[i]

        exact = False
        ast_ok = False
        ast_err = None

        exec_ok_strict = False
        exec_err_strict = None
        exec_ok_loose = False
        exec_err_loose = None

        where_ok = False
        where_err = None

        proj_ok = False
        proj_err = None

        frm_ok = False
        frm_err = None

        ast_relaxed_score = 0.0
        ast_relaxed_projection_jaccard = 0.0
        ast_relaxed_where_jaccard = 0.0
        ast_relaxed_from_sim = 0.0
        ast_relaxed_err = None
        gt_self_ast_ok = False
        gt_self_ast_err = None

        strict_overall_match = False
        relaxed_overall_score = 0.0
        relaxed_overall_match = False

        if not pred_sql:
            empty_pred_cnt += 1

        if new_sql:
            gt_self_ast_ok, gt_self_ast_err = ast_match_sql(new_sql, new_sql, dialect="sqlite")
            if gt_self_ast_ok:
                gt_self_ast_yes += 1
            elif gt_self_ast_err:
                gt_self_ast_err_cnt += 1

        if pred_sql and new_sql:
            exact = normalize_sql(pred_sql) == normalize_sql(new_sql)

            ast_ok, ast_err = ast_match_sql(pred_sql, new_sql, dialect="sqlite")
            if ast_err:
                ast_err_cnt += 1

            where_ok, where_err = where_match_commutative(pred_sql, new_sql, dialect="sqlite")
            proj_ok, proj_err = projection_match_set(pred_sql, new_sql, dialect="sqlite")
            frm_ok, frm_err = from_match(pred_sql, new_sql, dialect="sqlite")

            ast_relaxed, ast_relaxed_err = ast_relaxed_components(pred_sql, new_sql, dialect="sqlite")
            if ast_relaxed is None:
                ast_relaxed_err_cnt += 1
            else:
                ast_relaxed_score = ast_relaxed.get("score", 0.0)
                ast_relaxed_projection_jaccard = ast_relaxed.get("projection_jaccard", 0.0)
                ast_relaxed_where_jaccard = ast_relaxed.get("where_jaccard", 0.0)
                ast_relaxed_from_sim = ast_relaxed.get("from_sim", 0.0)
                ast_relaxed_sum += ast_relaxed_score

            exec_ok_strict, exec_err_strict = results_match_strict(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)
            if exec_err_strict and exec_err_strict.startswith("EXEC_ERROR"):
                exec_err_cnt += 1

            exec_ok_loose, exec_err_loose = results_match_loose(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)

        if exact:
            exact_yes += 1
        if ast_ok:
            ast_yes += 1
        if exec_ok_strict:
            exec_strict_yes += 1
        if exec_ok_loose:
            exec_loose_yes += 1
        if where_ok:
            where_yes += 1
        if proj_ok:
            proj_yes += 1
        if frm_ok:
            from_yes += 1

        strict_overall_match = bool(exact or ast_ok or exec_ok_strict)
        w_ast = max(0.0, float(args.relaxed_weight_ast))
        w_exec = max(0.0, float(args.relaxed_weight_exec_loose))
        w_sum = w_ast + w_exec
        if w_sum > 0:
            relaxed_overall_score = (
                (w_ast * ast_relaxed_score) + (w_exec * (1.0 if exec_ok_loose else 0.0))
            ) / w_sum
        else:
            relaxed_overall_score = 0.0
        relaxed_overall_match = relaxed_overall_score >= float(args.relaxed_overall_threshold)

        if strict_overall_match:
            overall_strict_yes += 1
        relaxed_overall_sum += relaxed_overall_score
        if relaxed_overall_match:
            relaxed_overall_yes += 1

        out_rows.append({
            "item_id": it["item_id"],
            "original_question": it["original_question"],
            "original_sql": it["original_sql"],
            "new_sql": new_sql,
            "new_sql_results_preview": previews[i],

            "generated_question": generated_questions[i],
            "sql_to_question_vllm_meta": q_meta[i],

            "pred_sql": pred_sql,
            "model_raw_sql_output": raw_sql_outputs[i],
            "question_to_sql_vllm_meta": sql_meta[i],

            "normalized_exact_match": bool(exact),

            "ast_match": bool(ast_ok),
            "ast_error": ast_err,
            "new_sql_ast_self_match": bool(gt_self_ast_ok),
            "new_sql_ast_self_error": gt_self_ast_err,

            "from_match": bool(frm_ok),
            "from_error": frm_err,

            "projection_match_set": bool(proj_ok),
            "projection_error": proj_err,

            "where_match_commutative": bool(where_ok),
            "where_error": where_err,

            "execution_match_strict": bool(exec_ok_strict),
            "execution_error_strict": exec_err_strict,
            "execution_match_loose": bool(exec_ok_loose),
            "execution_error_loose": exec_err_loose,

            "ast_relaxed_score": round(ast_relaxed_score, 4),
            "ast_relaxed_projection_jaccard": round(ast_relaxed_projection_jaccard, 4),
            "ast_relaxed_where_jaccard": round(ast_relaxed_where_jaccard, 4),
            "ast_relaxed_from_sim": round(ast_relaxed_from_sim, 4),
            "ast_relaxed_error": ast_relaxed_err,

            "overall_strict_match": bool(strict_overall_match),
            "overall_relaxed_score": round(relaxed_overall_score, 4),
            "overall_relaxed_match": bool(relaxed_overall_match),

            # Debugging: what columns we allowed for this item
            "allowed_columns_used_for_prompt": allowed_cols_per_item[i],
        })

    conn.close()

    # Write output JSON list
    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n================ FINAL SUMMARY ================")
    print(f"Output JSON: {out_path}")
    print(f"Processed: {total}")
    print(f"normalized_exact_match:     {exact_yes}/{total} ({pct(exact_yes, total):.2f}%)")
    print(f"ast_match (full query):     {ast_yes}/{total} ({pct(ast_yes, total):.2f}%)")
    print(f"new_sql_ast_self_match:     {gt_self_ast_yes}/{total} ({pct(gt_self_ast_yes, total):.2f}%)")
    print(f"from_match:                 {from_yes}/{total} ({pct(from_yes, total):.2f}%)")
    print(f"projection_match_set:       {proj_yes}/{total} ({pct(proj_yes, total):.2f}%)")
    print(f"where_match_commutative:    {where_yes}/{total} ({pct(where_yes, total):.2f}%)")
    print(f"execution_match_strict:     {exec_strict_yes}/{total} ({pct(exec_strict_yes, total):.2f}%)")
    print(f"execution_match_loose:      {exec_loose_yes}/{total} ({pct(exec_loose_yes, total):.2f}%)")

    print(f"overall_strict_match:       {overall_strict_yes}/{total} ({pct(overall_strict_yes, total):.2f}%)")
    print(f"ast_relaxed_avg_score:      {(ast_relaxed_sum / total) if total else 0.0:.4f}")
    print(f"overall_relaxed_avg_score:  {(relaxed_overall_sum / total) if total else 0.0:.4f}")
    print(
        f"overall_relaxed_match@{args.relaxed_overall_threshold:.2f}: "
        f"{relaxed_overall_yes}/{total} ({pct(relaxed_overall_yes, total):.2f}%)"
    )

    print(f"Empty pred_sql:             {empty_pred_cnt}")
    print(f"AST errors:                 {ast_err_cnt}")
    print(f"new_sql_ast_self_errors:    {gt_self_ast_err_cnt}")
    print(f"AST relaxed errors:         {ast_relaxed_err_cnt}")
    print(f"Execution errors (real):    {exec_err_cnt}")
    if gt_self_ast_yes != total:
        print("WARNING: new_sql AST self-check is below 100%; inspect new_sql_ast_self_error in output rows.")
    print("==============================================\n")


if __name__ == "__main__":
    main()
