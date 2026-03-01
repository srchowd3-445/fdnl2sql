#!/usr/bin/env python3
"""
CAT2 SQL (vLLM) - prompts from folder, schema from Excel, batched LLM calls first, then execute/evaluate.

Pipeline:
1) Load schema from --schema-xlsx (or fallback)
2) Load prompts from --prompts-dir (*.txt)
3) Read input data from --data-xlsx (questions + optional GT SQL)
4) Batch ALL LLM generations first (vLLM offline batched inference)
5) Execute predicted SQL (and GT optionally)
6) Evaluate + save artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm.auto import tqdm

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# --- Optional deps
try:
    from sqlalchemy import create_engine, text
    HAVE_SQLALCHEMY = True
except Exception:
    HAVE_SQLALCHEMY = False

try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except Exception:
    HAVE_VLLM = False

try:
    from openpyxl import Workbook
    from openpyxl.worksheet.table import Table, TableStyleInfo
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False


# ----------------------------------------------------------------------------
# Settings (kept minimal; NO hardcoded path hunting)
# ----------------------------------------------------------------------------
class Settings(BaseSettings):
    # Optional env defaults (you can remove these too if you want everything via CLI)
    cat2_engine_uri: str = Field(default="", validation_alias="CAT2_ENGINE_URI")
    cat2_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", validation_alias="CAT2_MODEL")
    cat2_dtype: str = Field(default="bfloat16", validation_alias="CAT2_DTYPE")

    class Config:
        extra = "ignore"


# ----------------------------------------------------------------------------
# Pydantic models for structure
# ----------------------------------------------------------------------------
class ModelJSON(BaseModel):
    sql: str = ""
    answer: str = ""
    assumptions: str = ""


class SQLFix(BaseModel):
    missing: str
    replaced_with: str
    reason: Optional[str] = None


class RowManifest(BaseModel):
    row_index: int
    question: str

    pred_sql: str
    pred_sql_final: str
    pred_sql_fixes: List[SQLFix] = Field(default_factory=list)
    pred_answer: str = ""
    pred_assumptions: str = ""

    ground_truth_sql: str = ""
    gt_sql_final: str = ""
    gt_sql_fixes: List[SQLFix] = Field(default_factory=list)

    sql_jaccard: Optional[float] = None
    relaxed_exact_match: Optional[bool] = None
    relaxed_em_threshold: float

    pred_exec_status: str
    pred_exec_ms: float
    pred_exec_error: Optional[str] = None
    pred_result_rows: Optional[int] = None
    pred_result_hash: str = ""

    gt_exec_status: str
    gt_exec_ms: float
    gt_exec_error: Optional[str] = None
    gt_result_rows: Optional[int] = None
    gt_result_hash: str = ""

    results_equal_exact: Optional[bool] = None
    results_equal_normalized: Optional[bool] = None
    rowcount_delta_pred_minus_gt: Optional[int] = None
    column_overlap_ratio: Optional[float] = None

    row_artifacts_dir: str


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1 :] if nl != -1 else ""
    s = s.strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s.strip()


def _normalize_json_quotes(s: str) -> str:
    return (s or "").replace("“", '"').replace("”", '"').replace("„", '"').replace("’", "'").replace("‘", "'")


def parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = _normalize_json_quotes(strip_code_fences(text))
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = _normalize_json_quotes(raw[start : end + 1])
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def extract_first_select(text: str) -> str:
    t = strip_code_fences(text or "")
    m = re.search(r"(?is)\b(with|select)\b.*?(;|\Z)", t)
    return m.group(0).strip() if m else ""


def sanitize_excel_sql(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.replace('""', '"')
    return s.strip()


def is_select_only(sql: str) -> bool:
    s = re.sub(r"/\*.*?\*/", "", (sql or ""), flags=re.S).strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate "]
    return not any(x in s for x in forbidden)


def qualify_table(sql: str, table_fqn: str) -> str:
    """Rewrite FROM/JOIN clinical_trials -> FROM/JOIN {table_fqn}."""
    def repl(m: re.Match) -> str:
        kw = m.group(1)
        return f"{kw} {table_fqn}"
    return re.sub(r"(?i)\b(from|join)\s+clinical_trials\b", repl, sql or "")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_norm_map(cols: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for c in cols:
        m[norm(c)] = c
        m[norm(c.replace("_", ""))] = c
        m[norm(c.replace("_", " "))] = c
    return m


def rewrite_quoted_identifiers_to_actual(
    sql: str,
    cols_set: Set[str],
    norm_map: Dict[str, str],
    table_schema: str,
    table_name: str,
) -> str:
    if not sql:
        return sql

    def repl(m: re.Match) -> str:
        ident = m.group(1)
        if ident in {table_schema, table_name}:
            return f'"{ident}"'
        nm = norm(ident)
        if nm in norm_map and norm_map[nm] in cols_set:
            return f'"{norm_map[nm]}"'
        return f'"{ident}"'

    return re.sub(r'"([^"]+)"', repl, sql)


def stable_hash_df_order_insensitive(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return ""
    try:
        df2 = df.copy()
        df2.columns = [str(c) for c in df2.columns]
        df2 = df2.reindex(sorted(df2.columns), axis=1)
        for c in df2.columns:
            df2[c] = df2[c].astype(str)
        df2 = df2.sort_values(list(df2.columns), kind="mergesort").reset_index(drop=True)
        b = df2.to_csv(index=False).encode("utf-8")
        return hashlib.md5(b).hexdigest()
    except Exception:
        return f"shape:{getattr(df, 'shape', None)}|cols:{list(getattr(df, 'columns', []))}"


def column_overlap_ratio(df_pred: Optional[pd.DataFrame], df_gt: Optional[pd.DataFrame]) -> Optional[float]:
    if df_pred is None or df_gt is None:
        return None
    A = {str(c).lower() for c in df_pred.columns}
    B = {str(c).lower() for c in df_gt.columns}
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


# ----------------------------------------------------------------------------
# Relaxed EM: SQL token Jaccard
# ----------------------------------------------------------------------------
def normalize_sql_tokens(sql: str) -> List[str]:
    if not sql:
        return []
    s = sql.lower()
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"(?s)'.*?'", " <str> ", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", " <num> ", s)
    toks = re.split(r"[^a-z0-9_]+", s)
    toks = [t for t in toks if t]
    stop = {
        "select","from","where","and","or","as","on","join","left","right","inner","outer",
        "group","by","order","limit","offset","distinct","asc","desc",
        "case","when","then","else","end",
        "ilike","like","in","is","null","not","with","union","all",
        "public","clinical_trials","str","num",
    }
    return [t for t in toks if t not in stop]


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def relaxed_em_sql(pred_sql: str, gt_sql: str, threshold: float) -> Tuple[Optional[float], Optional[bool]]:
    if not (gt_sql or "").strip():
        return None, None
    score = jaccard(normalize_sql_tokens(pred_sql or ""), normalize_sql_tokens(gt_sql or ""))
    return score, (score >= float(threshold))


# ----------------------------------------------------------------------------
# Auto-fix undefined columns (same)
# ----------------------------------------------------------------------------
KNOWN_MAP = {
    "publication_type": "originial_publication_or_follow_up",
    "ici_name": "name_of_ici",
    "ici_class": "class_of_ici",
    "therapy_modality": "monotherapy_combination",
    "combination_type": "type_of_combination",
    "control_type": "type_of_control",
    "clinical_setting_in_relation_to_surgery": "clincal_setting_in_relation_to_surgery",
    "pdl1_inclusion": "is_pd_l1_positivity_inclusion_criteria",
    "other_biomarker_inclusion": "is_any_other_biomarker_used_for_inclusion",
    "follow_up_type": "type_of_follow_up_given",
    "follow_up_duration_overall_months": "follow_up_duration_for_primary_endpoint_s_in_months_overall",
    "follow_up_duration_rx_months": "follow_up_duration_for_primary_endpoint_s_in_months_rx",
    "follow_up_duration_control_months": "follow_up_duration_for_primary_endpoint_s_in_months_control",
    "id": "nct",
}


def extract_undefined_column(err: str) -> Optional[str]:
    if not err:
        return None
    m = re.search(r'column\s+"([^"]+)"\s+does not exist', err, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"column\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+does not exist", err, flags=re.I)
    if m:
        return m.group(1)
    return None


def replace_identifier(sql: str, old: str, new: str) -> str:
    if not sql or not old or not new:
        return sql
    sql2 = re.sub(rf'"{re.escape(old)}"', f'"{new}"', sql)
    sql2 = re.sub(rf"\b{re.escape(old)}\b", new, sql2)
    return sql2


def pick_replacement(missing_col: str, cols_set: Set[str], norm_map: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    if not missing_col:
        return None, None

    if missing_col in KNOWN_MAP and KNOWN_MAP[missing_col] in cols_set:
        return KNOWN_MAP[missing_col], "known_map"

    nm = norm(missing_col)
    if nm in norm_map:
        cand = norm_map[nm]
        if cand in cols_set:
            return cand, "norm_fuzzy"

    tokens = [t for t in re.split(r"[_\W]+", missing_col.lower()) if t]
    best, best_score = None, 0
    for c in cols_set:
        c_low = c.lower()
        score = sum(1 for t in tokens if t and t in c_low)
        if score > best_score:
            best_score, best = score, c
    if best and best_score >= max(2, len(tokens) // 2):
        return best, f"token_overlap(score={best_score})"

    return None, None


def execute_sql(engine, sql: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str], float]:
    if engine is None:
        return "skipped", None, "No DB engine available", 0.0
    t0 = time.perf_counter()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return "ok", df, None, (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return "error", None, str(e), (time.perf_counter() - t0) * 1000.0


def execute_with_autofix(
    engine,
    sql: str,
    cols_set: Set[str],
    norm_map: Dict[str, str],
    max_fixes: int = 6,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str], float, str, List[SQLFix]]:
    if engine is None:
        return "skipped", None, "No DB engine available", 0.0, sql, []

    fixes: List[SQLFix] = []
    final_sql = sql

    for _ in range(max_fixes + 1):
        status, df, err, ms = execute_sql(engine, final_sql)
        if status == "ok":
            return "ok", df, None, ms, final_sql, fixes

        missing = extract_undefined_column(err or "")
        if not missing:
            return "error", None, err, ms, final_sql, fixes

        repl, reason = pick_replacement(missing, cols_set, norm_map)
        if not repl:
            return "error", None, err, ms, final_sql, fixes

        before = final_sql
        final_sql = replace_identifier(final_sql, missing, repl)
        if final_sql == before:
            return "error", None, err, ms, final_sql, fixes

        fixes.append(SQLFix(missing=missing, replaced_with=repl, reason=reason))

    return "error", None, f"Exceeded max_fixes={max_fixes}", 0.0, final_sql, fixes


# ----------------------------------------------------------------------------
# Prompts loading
# ----------------------------------------------------------------------------
def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise FileNotFoundError(f"--prompts-dir not found or not a dir: {prompts_dir}")

    prompts: Dict[str, str] = {}
    for p in sorted(prompts_dir.glob("*.txt")):
        prompts[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
    if "system" not in prompts or "user" not in prompts:
        raise ValueError("prompts-dir must contain system.txt and user.txt")
    return prompts


# ----------------------------------------------------------------------------
# Schema from Excel + schema output as Excel table
# ----------------------------------------------------------------------------
def load_schema_from_excel(schema_xlsx: Path) -> List[str]:
    if not schema_xlsx.exists():
        raise FileNotFoundError(f"--schema-xlsx not found: {schema_xlsx}")

    df = pd.read_excel(schema_xlsx)
    if df.empty:
        # maybe header-only
        hdr = pd.read_excel(schema_xlsx, nrows=0)
        cols = [str(c).strip() for c in hdr.columns if str(c).strip()]
        if cols:
            return cols
        raise ValueError(f"No schema found in {schema_xlsx}")

    # If there's a column_name column, use it
    col_name_col = next((c for c in df.columns if str(c).strip().lower() in {"column_name", "column", "name"}), None)
    if col_name_col:
        cols = [str(x).strip() for x in df[col_name_col].dropna().tolist() if str(x).strip()]
        if cols:
            return cols

    # Otherwise, if the sheet looks like schema-in-header, use headers
    if len(df.columns) > 1:
        cols = [str(c).strip() for c in df.columns if str(c).strip()]
        if cols:
            return cols

    # Last resort: first column values
    first_col = df.columns[0]
    cols = [str(x).strip() for x in df[first_col].dropna().tolist() if str(x).strip()]
    if cols:
        return cols

    raise ValueError(f"No schema found in {schema_xlsx}")


def write_schema_excel(out_path: Path, columns: List[str]) -> None:
    if not HAVE_OPENPYXL:
        # fallback to pandas
        pd.DataFrame({"column_name": columns}).to_excel(out_path, index=False)
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "schema"
    ws.append(["column_name"])
    for c in columns:
        ws.append([c])

    # Create an Excel "Table"
    end_row = len(columns) + 1
    tab = Table(displayName="SchemaTable", ref=f"A1:A{end_row}")
    style = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True, showColumnStripes=False)
    tab.tableStyleInfo = style
    ws.add_table(tab)
    wb.save(out_path)


# ----------------------------------------------------------------------------
# vLLM batch generation
# ----------------------------------------------------------------------------
def build_chat_prompt(model_id_or_path: str, system_text: str, user_text: str) -> str:
    # Minimal template; you can also use tokenizer.apply_chat_template as you had before if desired.
    return f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"


def vllm_generate_batch(llm: "LLM", prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
    sp = SamplingParams(temperature=float(temperature), max_tokens=int(max_tokens))
    outputs = llm.generate(prompts, sp)
    # outputs aligned with input prompts
    out_texts: List[str] = []
    for o in outputs:
        out_texts.append(o.outputs[0].text if o.outputs else "")
    return out_texts


def force_json_suffix(prompts: Dict[str, str]) -> str:
    # Prefer file-based suffix; else default.
    return (prompts.get("final_json_instructions") or "").strip() or (
        "\n\nIMPORTANT:\n"
        "- Return JSON only.\n"
        "- Do NOT wrap in ``` fences.\n"
        "- Begin your response with { and end with }.\n"
        "- Keys must be exactly: sql, answer, assumptions.\n"
    )


# ----------------------------------------------------------------------------
# Progress helpers
# ----------------------------------------------------------------------------
def append_jsonl(progress_jsonl: Path, obj: Dict[str, Any]) -> None:
    with progress_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    settings = Settings()

    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--data-xlsx", required=True, help="Excel with questions (+ optional GT SQL column).")
    ap.add_argument("--schema-xlsx", required=True, help="Excel containing schema (column names).")
    ap.add_argument("--prompts-dir", required=True, help="Folder containing system.txt and user.txt.")
    ap.add_argument("--results-dir", required=True, help="Output directory for artifacts/results.")

    # Columns in data xlsx
    ap.add_argument("--question-col", default="", help="Question column name. If empty, auto-detect.")
    ap.add_argument("--gt-sql-col", default="", help="GT SQL column name. If empty, auto-detect or none.")

    # Table identifiers
    ap.add_argument("--table-schema", default="public")
    ap.add_argument("--table-name", default="clinical_trials")
    ap.add_argument("--table-fqn", default="", help='Optional override, e.g., public."clinical_trials"')

    # Model
    ap.add_argument("--model", default=settings.cat2_model)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dtype", default=settings.cat2_dtype)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=600)

    # SQL exec
    ap.add_argument("--run-sql", action="store_true")
    ap.add_argument("--engine-uri", default=settings.cat2_engine_uri, help="SQLAlchemy URI (or env CAT2_ENGINE_URI).")

    # Auto-fix
    ap.add_argument("--autofix-gt", action="store_true", default=True)
    ap.add_argument("--autofix-pred", action="store_true", default=False)
    ap.add_argument("--autofix-max", type=int, default=6)

    # Eval
    ap.add_argument("--relaxed-em-threshold", type=float, default=0.70)

    args = ap.parse_args()

    if not HAVE_VLLM:
        raise RuntimeError("vLLM not installed (pip install vllm).")
    if args.run_sql and not HAVE_SQLALCHEMY:
        raise RuntimeError("SQLAlchemy not installed but --run-sql set.")
    if args.run_sql and not args.engine_uri:
        raise RuntimeError("--run-sql set but --engine-uri empty (and CAT2_ENGINE_URI empty).")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sqls_root = results_dir / "sql_runs"
    sqls_root.mkdir(parents=True, exist_ok=True)
    progress_jsonl = results_dir / "progress.jsonl"
    final_csv = results_dir / "evaluation.csv"

    prompts_dir = Path(args.prompts_dir)
    prompts = load_prompts(prompts_dir)

    schema_xlsx = Path(args.schema_xlsx)
    actual_columns = load_schema_from_excel(schema_xlsx)
    cols_set = set(actual_columns)
    norm_map = build_norm_map(actual_columns)

    # Write schema table to Excel output
    write_schema_excel(results_dir / "schema.xlsx", actual_columns)

    table_fqn = args.table_fqn.strip() or f'{args.table_schema}."{args.table_name}"'

    # Render system prompt with schema
    columns_block = "\n".join([f'- "{c}"' for c in actual_columns])
    system_text = prompts["system"].format(table_fqn=table_fqn, columns_block=columns_block)

    suffix = force_json_suffix(prompts)

    # Load data
    df_in = pd.read_excel(Path(args.data_xlsx))

    qcol = args.question_col.strip()
    if not qcol:
        qcol = next((c for c in df_in.columns if "query" in c.lower() or "question" in c.lower()), df_in.columns[0])

    gcol = args.gt_sql_col.strip()
    if not gcol:
        gcol = next((c for c in df_in.columns if "sql" in c.lower() or "ground" in c.lower()), "")

    questions = [str(x).strip() for x in df_in[qcol].tolist()]

    # Build ALL prompts first (batched)
    chat_prompts: List[str] = []
    for q in questions:
        user_text = prompts["user"].format(question=q)
        p = build_chat_prompt(args.model, system_text, user_text) + suffix
        chat_prompts.append(p)

    print(f"[INFO] Generating {len(chat_prompts)} completions in batched mode...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=True,
        swap_space=0,
    )

    t0 = time.time()
    raw_outputs = vllm_generate_batch(llm, chat_prompts, temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"[INFO] LLM generation done in {time.time()-t0:.2f}s")

    # Parse all model outputs into sql/answer/assumptions
    gens: List[Dict[str, Any]] = []
    for text_out in raw_outputs:
        data = parse_json_safe(text_out)
        parsed = ModelJSON(**data) if isinstance(data, dict) else ModelJSON()
        sql = parsed.sql.strip() or extract_first_select(text_out).strip()
        gens.append(
            {
                "sql": sql.strip(),
                "answer": parsed.answer.strip(),
                "assumptions": parsed.assumptions.strip(),
                "raw_text": text_out,
            }
        )

    # Create DB engine (optional)
    engine = None
    if args.run_sql:
        engine = create_engine(args.engine_uri, pool_pre_ping=True)

    records: List[Dict[str, Any]] = []

    # Now: execute pred sql (then GT) and evaluate
    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Execute+Eval", unit="q"):
        question = str(row[qcol]).strip()
        gen = gens[int(idx)]
        pred_sql_raw = (gen.get("sql") or "").strip()

        if not pred_sql_raw:
            pred_sql_raw = f"/* empty model output for row {idx} */ SELECT 1;"
        if not is_select_only(pred_sql_raw):
            pred_sql_raw = "/* non-SELECT produced; forcing safe query */ SELECT 1;"

        pred_sql = qualify_table(pred_sql_raw, table_fqn)

        # GT pipeline
        gt_sql_raw = ""
        if gcol and gcol in df_in.columns and pd.notna(row[gcol]):
            gt_sql_raw = sanitize_excel_sql(row[gcol])

        gt_sql = qualify_table(gt_sql_raw, table_fqn) if gt_sql_raw else ""
        gt_sql = rewrite_quoted_identifiers_to_actual(gt_sql, cols_set, norm_map, args.table_schema, args.table_name)

        sql_jaccard, relaxed_em = relaxed_em_sql(pred_sql, gt_sql, args.relaxed_em_threshold)

        per_row_dir = sqls_root / f"row{int(idx):03d}"
        per_row_dir.mkdir(parents=True, exist_ok=True)

        (per_row_dir / "raw_text.txt").write_text(gen.get("raw_text") or "", encoding="utf-8")
        (per_row_dir / "pred_sql.sql").write_text(pred_sql, encoding="utf-8")
        (per_row_dir / "gt_sql.sql").write_text(gt_sql or "", encoding="utf-8")

        pred_status, pred_df, pred_err, pred_ms = ("skipped", None, None, 0.0)
        gt_status, gt_df, gt_err, gt_ms = ("skipped", None, None, 0.0)
        pred_sql_final = pred_sql
        gt_sql_final = gt_sql
        pred_fixes: List[SQLFix] = []
        gt_fixes: List[SQLFix] = []

        if args.run_sql and engine is not None:
            # Pred
            if pred_sql and is_select_only(pred_sql):
                if args.autofix_pred:
                    pred_status, pred_df, pred_err, pred_ms, pred_sql_final, pred_fixes = execute_with_autofix(
                        engine, pred_sql, cols_set, norm_map, max_fixes=args.autofix_max
                    )
                else:
                    pred_status, pred_df, pred_err, pred_ms = execute_sql(engine, pred_sql)
            else:
                pred_status, pred_err = "skipped", "Pred SQL not SELECT/WITH-safe"

            # GT
            if gt_sql and is_select_only(gt_sql):
                if args.autofix_gt:
                    gt_status, gt_df, gt_err, gt_ms, gt_sql_final, gt_fixes = execute_with_autofix(
                        engine, gt_sql, cols_set, norm_map, max_fixes=args.autofix_max
                    )
                else:
                    gt_status, gt_df, gt_err, gt_ms = execute_sql(engine, gt_sql)
            else:
                gt_status = "skipped"
                gt_err = "No GT SQL" if not gt_sql else "GT SQL not SELECT/WITH-safe"

        pred_hash = stable_hash_df_order_insensitive(pred_df)
        gt_hash = stable_hash_df_order_insensitive(gt_df)

        if pred_df is not None:
            pred_df.to_csv(per_row_dir / "pred_results.csv", index=False)
        if gt_df is not None:
            gt_df.to_csv(per_row_dir / "gt_results.csv", index=False)

        results_equal_exact = None
        results_equal_normalized = None
        rowcount_delta = None
        col_overlap = None

        if pred_df is not None and gt_df is not None:
            try:
                results_equal_exact = bool(pred_df.equals(gt_df))
            except Exception:
                results_equal_exact = False
            results_equal_normalized = (pred_hash == gt_hash)
            rowcount_delta = len(pred_df) - len(gt_df)
            col_overlap = column_overlap_ratio(pred_df, gt_df)

        manifest = RowManifest(
            row_index=int(idx),
            question=question,
            pred_sql=pred_sql,
            pred_sql_final=pred_sql_final,
            pred_sql_fixes=pred_fixes,
            pred_answer=gen.get("answer") or "",
            pred_assumptions=gen.get("assumptions") or "",
            ground_truth_sql=(gt_sql or gt_sql_raw),
            gt_sql_final=gt_sql_final,
            gt_sql_fixes=gt_fixes,
            sql_jaccard=sql_jaccard,
            relaxed_exact_match=relaxed_em,
            relaxed_em_threshold=float(args.relaxed_em_threshold),
            pred_exec_status=pred_status,
            pred_exec_ms=pred_ms,
            pred_exec_error=pred_err,
            pred_result_rows=(None if pred_df is None else int(len(pred_df))),
            pred_result_hash=pred_hash,
            gt_exec_status=gt_status,
            gt_exec_ms=gt_ms,
            gt_exec_error=gt_err,
            gt_result_rows=(None if gt_df is None else int(len(gt_df))),
            gt_result_hash=gt_hash,
            results_equal_exact=results_equal_exact,
            results_equal_normalized=results_equal_normalized,
            rowcount_delta_pred_minus_gt=rowcount_delta,
            column_overlap_ratio=col_overlap,
            row_artifacts_dir=str(per_row_dir),
        )

        (per_row_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        append_jsonl(progress_jsonl, manifest.model_dump())
        records.append(manifest.model_dump())

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(final_csv, index=False)

    # Summary
    print("\n================ RUN SUMMARY ================")
    print(f"Artifacts folder: {results_dir}")
    print(f"Schema excel:      {results_dir / 'schema.xlsx'}")
    print(f"Progress JSONL:    {progress_jsonl}")
    print(f"Evaluation CSV:    {final_csv}")
    print(f"Completed rows:    {len(df_out)}")

    if len(df_out) > 0 and "relaxed_exact_match" in df_out:
        rem = df_out["relaxed_exact_match"].dropna()
        if len(rem) > 0:
            print(f"Relaxed EM rate (SQL Jaccard >= {args.relaxed_em_threshold}): {float(rem.mean()):.4f}")

    if args.run_sql and len(df_out) > 0 and "pred_exec_status" in df_out:
        pred_ok = (df_out["pred_exec_status"] == "ok")
        pred_ok_count = int(pred_ok.sum())
        pred_total = int(pred_ok.shape[0])
        print("\n------------- PRED SQL EXECUTION -------------")
        print(f"Pred SQL executed successfully: {pred_ok_count} / {pred_total} ({pred_ok_count / max(1, pred_total):.4f})")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
