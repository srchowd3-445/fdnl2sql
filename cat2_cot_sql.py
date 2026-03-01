#!/usr/bin/env python3
"""
CATEGORY 2 QUESTIONS - CoT 2-pass (vLLM)

SIMPLE SQL RESPONSES ON QUESTIONS THAT REQUIRE AN SQL RESPONSE

Keeps:
- Same DB config/envs (CAT2_DB_NAME, CAT2_ENGINE_URI, CAT2_TABLE_*, CAT2_XLSX, etc.)
- Same evaluation metrics + outputs:
  - pred/gt execution (when --run-sql)
  - save per-row artifacts (sql + result CSVs + manifest)
  - compare results (exact df.equals + normalized hash)
  - relaxed EM metric on SQL text (token-jaccard >= threshold)
  - print count of predicted SQLs that executed successfully (pred_exec_status == "ok")
- Same auto-fix behavior for undefined columns (GT default on; pred optional)

Changes:
- "Chain-of-thought style" prompting via 2-pass generation:
  Pass 1: scratchpad + draft SQL (we extract first SELECT/WITH)
  Pass 2: strict JSON only (sql/answer/assumptions), using draft as hint
- Optional saving of pass-1 scratchpad text per row
"""

import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set

from tqdm.auto import tqdm
import pandas as pd

# Optional: SQLAlchemy/DB
try:
    from sqlalchemy import create_engine, text
    HAVE_SQLALCHEMY = True
except Exception:
    HAVE_SQLALCHEMY = False

# Optional: vLLM
try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except Exception:
    HAVE_VLLM = False

# Optional: dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ----------------------------------------------------------------------------
# Defaults / ENV (kept same)
# ----------------------------------------------------------------------------
DB_NAME = os.getenv("CAT2_DB_NAME", "trialsdb")
ENGINE_URI_ENV = os.getenv("CAT2_ENGINE_URI", "").strip()

TABLE_NAME = os.getenv("CAT2_TABLE_NAME", "clinical_trials")
TABLE_SCHEMA = os.getenv("CAT2_TABLE_SCHEMA", "public")
TABLE_FQN = os.getenv("CAT2_TABLE_FQN", f'{TABLE_SCHEMA}."{TABLE_NAME}"')

# vLLM defaults (override by env or CLI)
ENV_MODEL = os.getenv("CAT2_MODEL", "").strip()
ENV_TP = os.getenv("CAT2_TP", "").strip()
ENV_GPU_MEM = os.getenv("CAT2_GPU_MEMORY_UTILIZATION", "").strip()
ENV_MAX_LEN = os.getenv("CAT2_MAX_MODEL_LEN", "").strip()
ENV_DTYPE = os.getenv("CAT2_DTYPE", "bfloat16").strip()
ENV_TEMP = os.getenv("CAT2_TEMPERATURE", "0.1").strip()
ENV_MAX_TOKENS = os.getenv("CAT2_MAX_TOKENS", "600").strip()

REPO_RUNS_ABS = Path(
    os.getenv(
        "CAT2_RUNS_ABS",
        "/mnt/data1/srchowd3/Mayo-CoRAL/LLM_Extraction_and_CrossCritique/interactive-table/runs",
    )
)


def _candidate_xlsx_paths(cli_xlsx: Optional[str] = None) -> List[Path]:
    out: List[Path] = []
    if cli_xlsx:
        out.append(Path(cli_xlsx).expanduser())

    env_x = os.getenv("CAT2_XLSX", "").strip()
    if env_x:
        out.append(Path(env_x).expanduser())

    out.extend(
        [
            Path("/mnt/data/cat2_query_sql.xlsx"),
            Path("/mnt/data/runs/cat2_query_sql.xlsx"),
            Path("runs/cat2_query_sql.xlsx"),
            Path("../runs/cat2_query_sql.xlsx"),
            REPO_RUNS_ABS / "cat2_query_sql.xlsx",
        ]
    )

    seen = set()
    uniq = []
    for p in out:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        k = str(rp)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# ----------------------------------------------------------------------------
# Column reflection / mapping (kept same)
# ----------------------------------------------------------------------------
actual_columns: List[str] = []
normalized_to_actual: Dict[str, str] = {}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_column_maps(cols: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for c in cols:
        m[norm(c)] = c
        m[norm(c.replace("_", " ").replace(" ", ""))] = c
        m[norm(c.replace(" ", "_"))] = c
    if "nct" in cols and "id" not in cols:
        m["id"] = "nct"
    return m


# Fallback should match your actual DB schema as closely as possible
fallback_cols = [
    "nct",
    "pubmed_id",
    "trial_name",
    "author",
    "year",
    "trial_phase",
    "number_of_arms",
    "total_sample_size",
    "originial_publication_or_follow_up",
    "cancer_type",
    "treatment_regimen",
    "name_of_ici",
    "class_of_ici",
    "monotherapy_combination",
    "type_of_combination",
    "control_regimen",
    "type_of_control",
    "lines_of_treatment",
    "clincal_setting_in_relation_to_surgery",
    "primary_endpoint",
    "priamry_multiple_composite_or_co_primary_endpoints",
    "secondary_endpoint",
    "is_pd_l1_positivity_inclusion_criteria",
    "is_any_other_biomarker_used_for_inclusion",
    "type_of_follow_up_given",
    "follow_up_duration_for_primary_endpoint_s_in_months_overall",
    "follow_up_duration_for_primary_endpoint_s_in_months_rx",
    "follow_up_duration_for_primary_endpoint_s_in_months_control",
    "class_of_ici_1",
    "name_of_ici_1",
    "type_of_therapy",
    "type_of_combination_1",
    "control_arm",
    "clinical_setting",
    "primary_endpoint_1",
    "included_in_ma",
]
actual_columns = fallback_cols[:]
normalized_to_actual = build_column_maps(actual_columns)


# ----------------------------------------------------------------------------
# Optional definitions (best-effort) (kept same)
# ----------------------------------------------------------------------------
def_roots = [
    Path("/mnt/data/definitions_folder"),
    Path("runs/definitions_folder"),
    Path("../runs/definitions_folder"),
    Path("./definitions_folder"),
    Path("/mnt/data"),
    Path("runs"),
    Path(".."),
    REPO_RUNS_ABS / "definitions_folder",
    Path("/mnt/data1/runs/definitions_folder"),
]
def_names = [
    "definitions - aim2 - filter concise.txt",
    "definitions - aim2 - column concise.txt",
]
def_paths: Dict[str, Optional[Path]] = {}
for name in def_names:
    found = None
    for root in def_roots:
        candidate = root / name
        if candidate.exists():
            found = candidate
            break
    def_paths[name] = found


def read_text_or_empty(p: Optional[Path]) -> str:
    if p is None:
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


definitions_filter = read_text_or_empty(def_paths["definitions - aim2 - filter concise.txt"])
definitions_column = read_text_or_empty(def_paths["definitions - aim2 - column concise.txt"])


# ----------------------------------------------------------------------------
# GT SQL sanitization + quoted identifier rewrite (IMPORTED FROM self_consistency)
# ----------------------------------------------------------------------------
def sanitize_excel_sql(s: str) -> str:
    """
    Fix common Excel/CSV quoting for SQL strings:
    - Strips surrounding single/double quotes if the whole SQL is wrapped
    - Unescapes doubled double-quotes ("") -> (")
    """
    if s is None:
        return ""
    s = str(s).strip()

    # strip outer quotes if the whole thing is quoted
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    # Excel escaping: "" -> "
    s = s.replace('""', '"')

    return s.strip()


def build_norm_map(cols: List[str]) -> Dict[str, str]:
    m = {}
    for c in cols:
        m[norm(c)] = c
        m[norm(c.replace("_", ""))] = c
        m[norm(c.replace("_", " "))] = c
    return m


def rewrite_quoted_identifiers_to_actual(sql: str, cols_set: Set[str], norm_map: Dict[str, str]) -> str:
    """
    Rewrites any quoted identifiers like "NCT" or "Cancer type" to the real column names
    using the norm_map (built from actual columns).
    """
    if not sql:
        return sql

    def repl(m):
        ident = m.group(1)
        # don't touch schema/table quoting
        if ident in {TABLE_SCHEMA, TABLE_NAME}:
            return f'"{ident}"'
        nm = norm(ident)
        if nm in norm_map and norm_map[nm] in cols_set:
            return f'"{norm_map[nm]}"'
        return f'"{ident}"'

    return re.sub(r'"([^"]+)"', repl, sql)


# ----------------------------------------------------------------------------
# Prompt construction (kept same guardrails/output format)
# ----------------------------------------------------------------------------
def make_system_context() -> str:
    schema_block = (
        f"# DATABASE\n"
        f"- Engine: PostgreSQL\n"
        f'- Table: {TABLE_FQN}\n'
        f"- Columns (use EXACT spelling below and always double-quote):\n  - "
        + "\n  - ".join(actual_columns)
    )

    definitions_block = ""
    if definitions_filter.strip() or definitions_column.strip():
        definitions_block = (
            "\n\n# DEFINITIONS (concise)\n"
            "## Filters\n" + (definitions_filter.strip() or "[missing]") + "\n\n"
            "## Columns\n" + (definitions_column.strip() or "[missing]")
        )

    guardrails = r"""
    # REQUIREMENTS
    - Use ONLY the columns listed above and reference them with double quotes, e.g., "cancer_type".
    - Qualify the table as public."clinical_trials".
    - When counting trials, use COUNT(DISTINCT "nct").
    - If PD-1 is mentioned, map to "PD1" for "class_of_ici" (no dash).
    - Prefer exact equality for enumerated values; use ILIKE with %...% for free text.
    - SELECT queries only; no DDL/DML.

    # SELECT COLUMN POLICY (MANDATORY)
    You MUST include columns in three groups:

    (A) Always include these identifiers in every non-COUNT query:
    "nct", "author", "year", "pubmed_id"

    (B) Always include these context columns in every non-COUNT query:
    "cancer_type", "trial_phase", "originial_publication_or_follow_up", "name_of_ici", "class_of_ici",
    "treatment_regimen", "type_of_combination", "monotherapy_combination", "total_sample_size"

    (C) If the question asks about a specific attribute, you MUST include the corresponding column(s):
    - primary endpoint(s) -> "primary_endpoint", "priamry_multiple_composite_or_co_primary_endpoints"
    - PD-L1 positivity inclusion -> "is_pd_l1_positivity_inclusion_criteria"
    - other biomarker inclusion -> "is_any_other_biomarker_used_for_inclusion"
    - number of arms -> "number_of_arms"
    - control regimen/arm -> "control_arm", "control_regimen"
    - lines of treatment -> "lines_of_treatment"
    - trial name -> "trial_name"

    If a column in (B) or (C) is NULL-heavy, still include it.
    If the query is COUNT/DISTINCT-only, do not include (A)/(B); return only the count.

    # OUTPUT FORMAT (strict JSON)
    Return ONLY a JSON object with keys:
    - "sql": string (runnable PostgreSQL SELECT)
    - "answer": string (<= 1–2 sentences)
    - "assumptions": string (empty if none)

    # OUTPUT SQL SKELETON (follow exactly)
    For non-COUNT queries, format the query like:

    SELECT
    "nct",
    "author",
    "year",
    "pubmed_id",
    <include any required columns from policy B/C here>
    FROM public."clinical_trials"
    WHERE <filters>;
    """

    return f"{schema_block}{definitions_block}{guardrails}"


def make_user_prompt(question: str) -> str:
    return (
        f"You are an expert clinical-trials data analyst. "
        f"Write a PostgreSQL query on table {TABLE_FQN} to answer the question.\n\n"
        f"QUESTION:\n{question}\n"
    )


# ----------------------------------------------------------------------------
# Robust parsing helpers (kept same)
# ----------------------------------------------------------------------------
def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        else:
            return ""
    s = s.strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s.strip()


def _normalize_json_quotes(s: str) -> str:
    if not s:
        return s
    return (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )


def parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None

    raw = strip_code_fences(text)
    raw = _normalize_json_quotes(raw)

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
            pass

    return None


def extract_first_select(text: str) -> str:
    t = strip_code_fences(text or "")
    m = re.search(r"(?is)\b(with|select)\b.*?(;|\Z)", t)
    return m.group(0).strip() if m else ""


# ----------------------------------------------------------------------------
# SQL helpers + execution + auto-fix (kept same)
# ----------------------------------------------------------------------------
def qualify_table(sql: str) -> str:
    """Rewrite FROM/JOIN clinical_trials -> FROM/JOIN {TABLE_FQN}."""
    def repl(m):
        kw = m.group(1)
        return f'{kw} {TABLE_FQN}'
    return re.sub(r"(?i)\b(from|join)\s+clinical_trials\b", repl, sql or "")


def is_select_only(sql: str) -> bool:
    s = re.sub(r"/\*.*?\*/", "", (sql or ""), flags=re.S).strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate "]
    return not any(x in s for x in forbidden)


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
# Relaxed EM: SQL token Jaccard (kept same)
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
        "select", "from", "where", "and", "or", "as", "on", "join", "left", "right", "inner", "outer",
        "group", "by", "order", "limit", "offset", "distinct", "asc", "desc",
        "case", "when", "then", "else", "end",
        "ilike", "like", "in", "is", "null", "not", "with", "union", "all",
        "public", "clinical_trials",
        "str", "num",
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


# --- Auto-fix: undefined column rewrite + retry (kept same)
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
    best = None
    best_score = 0
    for c in cols_set:
        c_low = c.lower()
        score = sum(1 for t in tokens if t and t in c_low)
        if score > best_score:
            best_score = score
            best = c
    if best and best_score >= max(2, len(tokens) // 2):
        return best, f"token_overlap(score={best_score})"

    return None, None


def execute_with_autofix(
    engine,
    sql: str,
    cols_set: Set[str],
    norm_map: Dict[str, str],
    max_fixes: int = 6,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str], float, str, List[Dict[str, Any]]]:
    if engine is None:
        return "skipped", None, "No DB engine available", 0.0, sql, []

    fixes: List[Dict[str, Any]] = []
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

        fixes.append({"missing": missing, "replaced_with": repl, "reason": reason})

    return "error", None, f"Exceeded max_fixes={max_fixes}", 0.0, final_sql, fixes


# ----------------------------------------------------------------------------
# vLLM prompting helpers
# ----------------------------------------------------------------------------
def build_chat_prompt(model_id_or_path: str, system_text: str, user_text: str) -> str:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        msgs = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"


def _vllm_generate_text(llm: "LLM", prompt: str, temperature: float, max_tokens: int) -> str:
    sp = SamplingParams(temperature=float(temperature), max_tokens=int(max_tokens))
    out = llm.generate([prompt], sp)
    return out[0].outputs[0].text if out and out[0].outputs else ""


def generate_sql_json_cot_2pass(
    llm: "LLM",
    model_id_or_path: str,
    system_text: str,
    question: str,
    temperature: float,
    max_tokens: int,
    final_temperature: Optional[float] = None,
    final_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    # Pass 1
    user1 = (
        "You are an expert clinical-trials data analyst.\n"
        f"QUESTION:\n{question}\n\n"
        "Think step-by-step privately to derive the correct SQL.\n"
        "Then output ONLY a runnable PostgreSQL SELECT/WITH query.\n"
        "Do not output JSON in this step. Do not wrap in ```.\n"
    )
    prompt1 = build_chat_prompt(model_id_or_path, system_text, user1)
    cot_raw = _vllm_generate_text(llm, prompt1, temperature=temperature, max_tokens=max_tokens)
    draft_sql = extract_first_select(cot_raw).strip()

    # Pass 2
    user2 = (
        "You are an expert clinical-trials data analyst.\n"
        f"QUESTION:\n{question}\n\n"
        "Here is a draft SQL attempt (may be incomplete):\n"
        f"{draft_sql or '[EMPTY DRAFT]'}\n\n"
        "Now produce the final answer.\n"
    )
    prompt2 = build_chat_prompt(model_id_or_path, system_text, user2)
    prompt2_forced = (
        prompt2
        + "\n\nIMPORTANT:\n"
          "- Return JSON only.\n"
          "- Do NOT wrap in ``` fences.\n"
          "- Begin your response with { and end with }.\n"
          "- Keys must be exactly: sql, answer, assumptions.\n"
          "- You may revise the draft SQL if needed.\n"
    )
    t2 = float(final_temperature if final_temperature is not None else temperature)
    m2 = int(final_max_tokens if final_max_tokens is not None else max_tokens)
    text_out = _vllm_generate_text(llm, prompt2_forced, temperature=t2, max_tokens=m2)

    data = parse_json_safe(text_out)

    sql = ""
    answer = ""
    assumptions = ""
    if isinstance(data, dict):
        sql = str(data.get("sql", "")).strip()
        answer = str(data.get("answer", "")).strip()
        assumptions = str(data.get("assumptions", "")).strip()

    if not sql:
        sql = extract_first_select(text_out).strip() or draft_sql

    return {
        "sql": (sql or "").strip(),
        "answer": answer,
        "assumptions": assumptions,
        "raw_text": text_out,
        "cot_raw_text": cot_raw,
        "draft_sql": draft_sql,
    }


def generate_sql_json_singlepass_json(
    llm: "LLM",
    prompt_forced: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    text_out = _vllm_generate_text(llm, prompt_forced, temperature=temperature, max_tokens=max_tokens)
    data = parse_json_safe(text_out)

    sql = ""
    answer = ""
    assumptions = ""
    if isinstance(data, dict):
        sql = str(data.get("sql", "")).strip()
        answer = str(data.get("answer", "")).strip()
        assumptions = str(data.get("assumptions", "")).strip()

    if not sql:
        sql = extract_first_select(text_out)

    return {"sql": sql.strip(), "answer": answer, "assumptions": assumptions, "raw_text": text_out}


# ----------------------------------------------------------------------------
# Resume helpers (kept same)
# ----------------------------------------------------------------------------
def load_completed_indices(progress_jsonl: Path) -> Set[int]:
    done: Set[int] = set()
    if not progress_jsonl.exists():
        return done
    with progress_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "row_index" in obj:
                    done.add(int(obj["row_index"]))
            except Exception:
                continue
    return done


def append_jsonl(progress_jsonl: Path, obj: Dict[str, Any]) -> None:
    with progress_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_slug(s: str, max_len: int = 60) -> str:
    s = (s or "").strip()
    s = re.sub(r"^models--", "", s)
    s = s.replace("--", "/")
    s = re.sub(r"[^A-Za-z0-9_.-/]+", "_", s)
    s = s.strip("_")
    return s[:max_len] if len(s) > max_len else s


def _mean(series_like) -> Optional[float]:
    try:
        s = pd.to_numeric(series_like, errors="coerce").dropna()
        if len(s) == 0:
            return None
        return float(s.mean())
    except Exception:
        return None


def reflect_columns(engine, schema: str, table: str) -> List[str]:
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            q = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """
            )
            cols = [r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()]
        return cols
    except Exception:
        return []


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    global actual_columns, normalized_to_actual

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=(ENV_MODEL or "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--xlsx", type=str, default="")
    parser.add_argument("--limit", type=int, default=int(os.getenv("NUM_QUESTIONS", "0") or 0))

    parser.add_argument("--tp", type=int, default=int(ENV_TP) if ENV_TP.isdigit() else 1)
    parser.add_argument("--gpu-mem-util", type=float, default=float(ENV_GPU_MEM) if ENV_GPU_MEM else 0.90)
    parser.add_argument("--max-model-len", type=int, default=int(ENV_MAX_LEN) if ENV_MAX_LEN.isdigit() else 8192)
    parser.add_argument("--dtype", type=str, default=ENV_DTYPE)
    parser.add_argument("--temperature", type=float, default=float(ENV_TEMP))
    parser.add_argument("--max-tokens", type=int, default=int(ENV_MAX_TOKENS))

    parser.add_argument("--results-root", type=str, default="results")

    # SQL / DB
    parser.add_argument("--run-sql", action="store_true")
    parser.add_argument("--engine-uri", type=str, default=ENGINE_URI_ENV,
                        help="SQLAlchemy URI (or set CAT2_ENGINE_URI).")

    # Auto-fix controls
    parser.add_argument("--no-autofix-gt", action="store_true",
                        help="Disable auto-fix GT SQL on undefined-column.")
    parser.add_argument("--autofix-pred", action="store_true", default=False,
                        help="Also auto-fix predicted SQL on undefined-column (default False).")
    parser.add_argument("--autofix-max", type=int, default=int(os.getenv("AUTOFIX_MAX", "6")),
                        help="Max number of undefined-column rewrites per SQL execution.")

    # Relaxed EM threshold
    parser.add_argument("--relaxed-em-threshold", type=float,
                        default=float(os.getenv("RELAXED_EM_THRESHOLD", "0.70")),
                        help="Jaccard threshold over normalized SQL tokens for relaxed EM.")

    # CoT 2-pass controls
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable 2-pass CoT-style prompting (use single-pass strict JSON).")
    parser.add_argument("--cot-final-temperature", type=float, default=None,
                        help="Optional temperature for pass-2 final JSON (defaults to --temperature).")
    parser.add_argument("--cot-final-max-tokens", type=int, default=None,
                        help="Optional max tokens for pass-2 final JSON (defaults to --max-tokens).")
    parser.add_argument("--save-cot", action="store_true",
                        help="If set, save pass-1 scratchpad output to cot_raw_text.txt per row.")

    args = parser.parse_args()
    args.autofix_gt = (not bool(args.no_autofix_gt))

    if not HAVE_VLLM:
        raise RuntimeError("vLLM not installed (pip install vllm).")

    if args.run_sql and not HAVE_SQLALCHEMY:
        raise RuntimeError("SQLAlchemy not installed but --run-sql was set.")
    if args.run_sql and not args.engine_uri:
        raise RuntimeError("--run-sql set but --engine-uri is empty and CAT2_ENGINE_URI is empty.")

    # Locate xlsx
    candidates = _candidate_xlsx_paths(args.xlsx.strip() or None)
    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        print("\n[ERROR] Could not find query-cat2.xlsx.")
        print(f"  cwd: {Path.cwd()}")
        print("  searched:")
        for p in candidates:
            print(f"   - {p}")
        raise FileNotFoundError("Could not find query-cat2.xlsx (set --xlsx or CAT2_XLSX).")

    df_in = pd.read_excel(xlsx_path)
    qcol = next((c for c in df_in.columns if "query" in c.lower() or "question" in c.lower()), df_in.columns[0])
    gcol = next((c for c in df_in.columns if "sql" in c.lower() or "ground" in c.lower() or "postgres" in c.lower()), None)

    if args.limit and args.limit > 0:
        df_in = df_in.head(args.limit)

    model_name = args.model_name.strip() or (
        args.model if ("/" in args.model and not Path(args.model).exists()) else Path(args.model).name
    )
    model_name_slug = safe_slug(model_name, max_len=80).replace("/", "_")

    run_id = args.run_id.strip() or f"cat2_sql_{time.strftime('%Y%m%d_%H%M%S')}_{model_name_slug}"
    results_dir = Path(args.results_root) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    progress_jsonl = results_dir / "progress.jsonl"
    final_csv = results_dir / "evaluation.csv"
    sqls_root = results_dir / "sql_runs"
    sqls_root.mkdir(parents=True, exist_ok=True)

    completed = load_completed_indices(progress_jsonl)

    engine = None
    cols_set: Set[str] = set(actual_columns)
    norm_map: Dict[str, str] = build_norm_map(actual_columns)

    if HAVE_SQLALCHEMY and args.engine_uri:
        engine = create_engine(args.engine_uri, pool_pre_ping=True)
        cols = reflect_columns(engine, TABLE_SCHEMA, TABLE_NAME)
        if cols:
            actual_columns = cols
            normalized_to_actual = build_column_maps(actual_columns)
            cols_set = set(actual_columns)
            norm_map = build_norm_map(actual_columns)

    if args.run_sql and engine is None:
        raise RuntimeError("--run-sql was set but engine could not be created (invalid engine uri?).")

    print("========================================")
    print("CAT2 SQL (vLLM) - GEN + (optional) EXEC + SAVE + COMPARE")
    print(f"Model (load): {args.model}")
    print(f"Model (name): {model_name}")
    print(f"Run ID: {run_id}")
    print(f"Results dir: {results_dir}")
    print(f"XLSX: {xlsx_path}")
    print(f"qcol: {qcol} | gt col: {gcol}")
    print(f"engine_uri_set: {bool(args.engine_uri)} | run_sql: {args.run_sql}")
    print(f"Auto-fix GT: {args.autofix_gt} | Auto-fix pred: {args.autofix_pred} | max fixes: {args.autofix_max}")
    print(f"Relaxed EM threshold: {args.relaxed_em_threshold}")
    print(f"CoT 2-pass enabled: {not bool(args.no_cot)} | save-cot: {bool(args.save_cot)}")
    print(f"Resume: {len(completed)} rows already completed")
    print("========================================")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=True,
        swap_space=0,
    )

    system_text = make_system_context()

    records: List[Dict[str, Any]] = []
    if progress_jsonl.exists():
        with progress_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "row_index" in obj and "question" in obj:
                        records.append(obj)
                except Exception:
                    continue

    start = time.time()

    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Questions", unit="q"):
        if int(idx) in completed:
            continue

        question = str(row[qcol]).strip()

        # -----------------------------
        # GT: sanitize + qualify + rewrite quoted identifiers  (FIX)
        # -----------------------------
        gt_sql_raw = ""
        if gcol and gcol in df_in.columns and pd.notna(row[gcol]):
            gt_sql_raw = sanitize_excel_sql(row[gcol])

        user_text = make_user_prompt(question)
        prompt = build_chat_prompt(args.model, system_text, user_text)
        prompt_forced = (
            prompt
            + "\n\nIMPORTANT:\n"
              "- Return JSON only.\n"
              "- Do NOT wrap in ``` fences.\n"
              "- Begin your response with { and end with }.\n"
              "- Keys must be exactly: sql, answer, assumptions.\n"
        )

        if not bool(args.no_cot):
            gen = generate_sql_json_cot_2pass(
                llm=llm,
                model_id_or_path=args.model,
                system_text=system_text,
                question=question,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                final_temperature=args.cot_final_temperature,
                final_max_tokens=args.cot_final_max_tokens,
            )
        else:
            gen = generate_sql_json_singlepass_json(
                llm=llm,
                prompt_forced=prompt_forced,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        pred_sql_raw = (gen.get("sql") or "").strip()

        if not pred_sql_raw:
            pred_sql_raw = f"/* empty model output for row {idx} */ SELECT 1;"
        if not is_select_only(pred_sql_raw):
            pred_sql_raw = f"/* non-SELECT produced; forcing safe query */ SELECT 1;"

        pred_sql = qualify_table(pred_sql_raw)

        # Build GT SQL (FIXED PIPELINE)
        gt_sql = qualify_table(gt_sql_raw) if gt_sql_raw else ""
        gt_sql = rewrite_quoted_identifiers_to_actual(gt_sql, cols_set, norm_map)

        # ---- Relaxed EM metric (SQL text) ----
        sql_jaccard, relaxed_em = relaxed_em_sql(pred_sql, gt_sql, args.relaxed_em_threshold)

        per_row_dir = sqls_root / f"row{int(idx):03d}"
        per_row_dir.mkdir(parents=True, exist_ok=True)

        (per_row_dir / "raw_text.txt").write_text(gen.get("raw_text") or "", encoding="utf-8")
        (per_row_dir / "pred_sql.sql").write_text(pred_sql, encoding="utf-8")
        (per_row_dir / "gt_sql.sql").write_text(gt_sql or "", encoding="utf-8")

        if bool(args.save_cot) and gen.get("cot_raw_text") is not None:
            (per_row_dir / "cot_raw_text.txt").write_text(gen.get("cot_raw_text") or "", encoding="utf-8")
        if gen.get("draft_sql") is not None:
            (per_row_dir / "draft_sql.sql").write_text(gen.get("draft_sql") or "", encoding="utf-8")

        pred_status, pred_df, pred_err, pred_ms = ("skipped", None, None, 0.0)
        gt_status, gt_df, gt_err, gt_ms = ("skipped", None, None, 0.0)
        pred_sql_final = pred_sql
        gt_sql_final = gt_sql
        pred_fixes = []
        gt_fixes = []

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

            # GT (now sanitized + rewritten before exec)
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

        rec = {
            "row_index": int(idx),
            "question": question,
            "model_name": model_name,
            "model_load_path_or_id": args.model,

            "pred_sql": pred_sql,
            "pred_sql_final": pred_sql_final,
            "pred_sql_fixes": pred_fixes,
            "pred_answer": gen.get("answer"),
            "pred_assumptions": gen.get("assumptions"),
            "raw_text": gen.get("raw_text"),

            "ground_truth_sql": gt_sql or gt_sql_raw,
            "gt_sql_final": gt_sql_final,
            "gt_sql_fixes": gt_fixes,

            "sql_jaccard": sql_jaccard,
            "relaxed_exact_match": relaxed_em,
            "relaxed_em_threshold": float(args.relaxed_em_threshold),

            "pred_exec_status": pred_status,
            "pred_exec_ms": pred_ms,
            "pred_exec_error": pred_err,
            "pred_result_rows": (None if pred_df is None else int(len(pred_df))),
            "pred_result_hash": pred_hash,

            "gt_exec_status": gt_status,
            "gt_exec_ms": gt_ms,
            "gt_exec_error": gt_err,
            "gt_result_rows": (None if gt_df is None else int(len(gt_df))),
            "gt_result_hash": gt_hash,

            "results_equal_exact": results_equal_exact,
            "results_equal_normalized": results_equal_normalized,
            "rowcount_delta_pred_minus_gt": rowcount_delta,
            "column_overlap_ratio": col_overlap,

            "row_artifacts_dir": str(per_row_dir),
        }

        (per_row_dir / "manifest.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        append_jsonl(progress_jsonl, rec)

        records.append(rec)
        completed.add(int(idx))

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(final_csv, index=False)

    elapsed = time.time() - start

    print("\n================ RUN SUMMARY ================")
    print(f"Artifacts folder: {results_dir}")
    print(f"Progress JSONL:   {progress_jsonl}")
    print(f"Evaluation CSV:   {final_csv}")
    print(f"Completed rows:   {len(df_out)} / {len(df_in)}")
    print(f"Runtime:          {elapsed:.2f}s")

    if len(df_out) > 0 and "relaxed_exact_match" in df_out:
        rem = df_out["relaxed_exact_match"].dropna()
        if len(rem) > 0:
            print(f"Relaxed EM rate (SQL Jaccard >= {args.relaxed_em_threshold}): {float(rem.mean()):.4f}")

    if args.run_sql and len(df_out) > 0:
        pred_ok = (df_out.get("pred_exec_status") == "ok") if "pred_exec_status" in df_out else None
        if pred_ok is not None:
            pred_ok_count = int(pred_ok.sum())
            pred_total = int(pred_ok.shape[0])
            print("\n------------- PRED SQL EXECUTION -------------")
            print(f"Pred SQL executed successfully: {pred_ok_count} / {pred_total} ({pred_ok_count / max(1, pred_total):.4f})")

        ok_rate_pred = float((df_out["pred_exec_status"] == "ok").mean()) if "pred_exec_status" in df_out else None
        ok_rate_gt = float((df_out["gt_exec_status"] == "ok").mean()) if "gt_exec_status" in df_out else None
        eq_exact = float(df_out["results_equal_exact"].fillna(False).astype(bool).mean()) if "results_equal_exact" in df_out else None
        eq_norm = float(df_out["results_equal_normalized"].fillna(False).astype(bool).mean()) if "results_equal_normalized" in df_out else None

        print("\n------------- EXEC + COMPARE METRICS -------------")
        print(f"Pred exec ok rate:        {ok_rate_pred:.4f}" if ok_rate_pred is not None else "Pred exec ok rate: NA")
        print(f"GT exec ok rate:          {ok_rate_gt:.4f}" if ok_rate_gt is not None else "GT exec ok rate: NA")
        print(f"Results equal (exact):    {eq_exact:.4f}" if eq_exact is not None else "Results equal (exact): NA")
        print(f"Results equal (normhash): {eq_norm:.4f}" if eq_norm is not None else "Results equal (normhash): NA")
        print(f"Avg pred ms:              {_mean(df_out.get('pred_exec_ms')):.2f}" if _mean(df_out.get("pred_exec_ms")) is not None else "Avg pred ms: NA")
        print(f"Avg gt ms:                {_mean(df_out.get('gt_exec_ms')):.2f}" if _mean(df_out.get("gt_exec_ms")) is not None else "Avg gt ms: NA")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
