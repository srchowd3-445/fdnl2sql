#!/usr/bin/env python3
"""
Decompose natural clinical-trial questions into atomic sub-questions with SQL,
with both single-row test mode and OpenAI Batch API mode.

Modes:
- single:
    Run direct Chat Completions requests for a sliced subset.
    If --limit is omitted (-1), single mode defaults to 1 row for safe testing.

- prepare_batch:
    Build batch request JSONL + metadata mapping (no API submission).

- batch_run:
    Prepare batch JSONL, submit to Batch API, poll until terminal state,
    download output/error JSONL, parse, and write final merged output JSON.

- batch_collect:
    Collect/parse a previously submitted batch using --batch_id or existing
    --batch_output_jsonl_path + --batch_meta_path.

Environment:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from pydantic import BaseModel, ConfigDict, Field

try:
    # Optional runtime schema sanity check.
    from openai.types import ResponseFormatJSONSchema
except Exception:
    ResponseFormatJSONSchema = None


API_KEY_ENV_CANDIDATES = [
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "GPT_KEY",
    "gpt_key",
]


SYSTEM_PROMPT = """You are a clinical-trials SQL decomposition assistant.
Decompose each original question into smaller, atomic data-point questions and produce one SQL query per atomic question.

Requirements:
- Use ONLY table clinical_trials.
- Use ONLY provided columns.
- Preserve literal values, entities, and thresholds from the original question/GT SQL.
- Prefer short atomic checks over one giant query.
- SQL must be a single SELECT statement per decomposed item.
- Avoid markdown and explanations; output only the required JSON schema.
"""


class DecomposedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., description="Atomic decomposed sub-question")
    sql: str = Field(..., description="SQL query answering the atomic sub-question")


class DecompositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decomposed_queries: List[DecomposedQuery] = Field(
        ..., min_length=1, max_length=12, description="Atomic query decomposition"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decompose natural questions into atomic sub-queries with SQL.")

    # Execution mode
    ap.add_argument(
        "--mode",
        choices=["single", "prepare_batch", "batch_run", "batch_collect"],
        default="single",
    )

    # Core I/O
    ap.add_argument(
        "--input_json",
        default="data/natural_question_additional_200_naturalized.json",
        help="Input JSON list with natural_question + gt_sql",
    )
    ap.add_argument(
        "--output_json",
        default="data/natural_question_additional_200_decomposed.json",
        help="Final parsed decomposition output JSON",
    )
    ap.add_argument(
        "--schema_json",
        default="data/schema.json",
        help="JSON list of allowed column names",
    )
    ap.add_argument("--table_name", default="clinical_trials")

    # Row slicing
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1, help="-1 means no limit")

    # Input key mapping
    ap.add_argument(
        "--question_keys",
        default="natural_question,question,original_question,new_question",
        help="Comma-separated candidate keys for original question",
    )
    ap.add_argument(
        "--sql_keys",
        default="gt_sql,new_sql,new_gt_sql,sql",
        help="Comma-separated candidate keys for ground-truth SQL",
    )
    ap.add_argument("--id_key", default="item_id")

    # Model/provider
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--api_key", default="", help="API key override; otherwise resolved from env/.env")
    ap.add_argument("--env_file", default=".env", help="Path to .env file containing OPENAI_API_KEY")
    ap.add_argument("--api_base", default="", help="Optional base URL for OpenAI-compatible endpoint")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1200, help="Max completion tokens value")
    ap.add_argument(
        "--token_param",
        choices=["auto", "max_completion_tokens", "max_tokens"],
        default="auto",
        help="Token parameter name to send to Chat Completions",
    )

    # Batch artifacts
    ap.add_argument("--batch_jsonl_path", default="data/decompose_batch_input.jsonl")
    ap.add_argument("--batch_meta_path", default="data/decompose_batch_meta.json")
    ap.add_argument("--batch_output_jsonl_path", default="data/decompose_batch_output.jsonl")
    ap.add_argument("--batch_error_jsonl_path", default="data/decompose_batch_error.jsonl")
    ap.add_argument(
        "--raw_output_jsonl",
        default="",
        help="Optional path for raw model outputs; default is <output_json>.raw.jsonl",
    )
    ap.add_argument("--batch_id", default="", help="Existing batch ID (used by batch_collect)")
    ap.add_argument("--poll_every_s", type=float, default=10.0)
    ap.add_argument("--completion_window", default="24h")

    return ap.parse_args()


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            return str(obj)
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_raw_output_path(args: argparse.Namespace) -> str:
    if (args.raw_output_jsonl or "").strip():
        return args.raw_output_jsonl
    return args.output_json + ".raw.jsonl"


def load_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("input_json must contain a JSON list.")
    out = [r for r in obj if isinstance(r, dict)]
    return out


def load_schema_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("schema_json must be a JSON list of column names")
    cols = [str(x).strip() for x in obj if str(x).strip()]
    if not cols:
        raise ValueError("schema_json has no usable columns")
    return cols


def parse_comma_keys(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_dotenv_value(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def read_key_from_env_file(env_path: str, key_name: str = "OPENAI_API_KEY") -> str:
    if not env_path or not os.path.exists(env_path):
        return ""
    target = (key_name or "").strip().lower()
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                if t.startswith("export "):
                    t = t[len("export ") :].strip()
                if "=" not in t:
                    continue
                k, v = t.split("=", 1)
                if k.strip().lower() != target:
                    continue
                return parse_dotenv_value(v)
    except Exception:
        return ""
    return ""


def resolve_api_key(cli_api_key: str, env_file: str) -> str:
    if (cli_api_key or "").strip():
        return cli_api_key.strip()

    for env_name in API_KEY_ENV_CANDIDATES:
        env_key = (os.getenv(env_name) or "").strip()
        if env_key:
            return env_key

    candidates: List[str] = []
    if env_file:
        if os.path.isabs(env_file):
            candidates.append(env_file)
        else:
            candidates.append(str(Path.cwd() / env_file))
            candidates.append(str(Path(__file__).resolve().parent / env_file))

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        for key_name in API_KEY_ENV_CANDIDATES:
            k = read_key_from_env_file(c, key_name=key_name)
            if k:
                return k
    return ""


def pick_first_nonempty(row: Dict[str, Any], keys: List[str]) -> Tuple[str, Optional[str]]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            return sv, k
    return "", None


def choose_token_param_name(model: str, token_param: str) -> str:
    forced = (token_param or "").strip()
    if forced in {"max_completion_tokens", "max_tokens"}:
        return forced
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def should_send_temperature(model: str) -> bool:
    m = (model or "").strip().lower()
    # GPT-5 chat completions currently enforce default temperature behavior.
    if m.startswith("gpt-5"):
        return False
    return True


def slice_rows(rows: List[Dict[str, Any]], start: int, limit: int, mode: str) -> List[Dict[str, Any]]:
    s = max(0, int(start))
    part = rows[s:]
    eff_limit = int(limit)
    # Default-safe behavior for initial testing.
    if mode == "single" and eff_limit == -1:
        eff_limit = 1
    if eff_limit > -1:
        part = part[:eff_limit]
    return part


def pydantic_to_response_format(model: type[BaseModel]) -> Dict[str, Any]:
    payload = {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }
    if ResponseFormatJSONSchema is not None:
        ResponseFormatJSONSchema.model_validate(payload)
    return payload


def build_user_prompt(
    *,
    table_name: str,
    schema_cols: List[str],
    original_question: str,
    gt_sql: str,
) -> str:
    col_list = ", ".join([f'"{c}"' for c in schema_cols])
    return f"""Table:
{table_name}

Allowed columns:
{col_list}

Original question:
{original_question}

Ground-truth SQL:
{gt_sql}

Task:
Decompose the original question into atomic data-point questions.

How to decompose:
- Cover selected output concepts and filter constraints from the original query.
- Typically include one sub-question per key filter/entity/condition.
- Keep each sub-question focused (atomic), not full end-to-end.

SQL rules:
- Use only table \"{table_name}\".
- Use only listed columns.
- One SELECT statement per decomposed query.
- Keep literals/values consistent with the GT SQL.
- End each SQL with semicolon.

Return JSON that matches the required schema only.
"""


def normalize_chat_content(content: Any) -> str:
    def extract_text_one(part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            for k in ("text", "content", "value"):
                v = part.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return ""
        for attr in ("text", "content", "value"):
            v = getattr(part, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            txt = extract_text_one(p)
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    return extract_text_one(content).strip()


def normalize_sql(sql: str) -> str:
    s = (sql or "").strip()
    if not s:
        return ""
    if not s.endswith(";"):
        s += ";"
    return s


def normalize_question(q: str) -> str:
    s = (q or "").strip().strip('"').strip("'").strip()
    return s


def payload_to_query_map(payload: DecompositionPayload) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for i, dq in enumerate(payload.decomposed_queries, start=1):
        out[f"query_{i}"] = {
            "question": normalize_question(dq.question),
            "sql": normalize_sql(dq.sql),
        }
    return out


def make_output_row(
    *,
    idx: int,
    item_id: Any,
    original_question: str,
    gt_sql: str,
    q_key: Optional[str],
    sql_key: Optional[str],
    model: str,
    payload: Optional[DecompositionPayload],
    status: str,
    error: Optional[str] = None,
    custom_id: Optional[str] = None,
) -> Dict[str, Any]:
    query_map = payload_to_query_map(payload) if payload is not None else {}
    return {
        "idx": idx,
        "item_id": item_id,
        "original_question": original_question,
        "gt_sql": gt_sql,
        "decomposed_query": query_map,
        "status": status,
        "error": error,
        "source_question_key": q_key,
        "source_sql_key": sql_key,
        "model": model,
        "custom_id": custom_id,
    }


def get_client(api_key: str, api_base: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai is not installed/importable. Install with: pip install openai")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api_key")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def call_decompose_once(
    *,
    client: Any,
    model: str,
    response_format: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    token_param: str,
    prompt: str,
) -> Tuple[Optional[DecompositionPayload], Dict[str, Any], Optional[str]]:
    req: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": response_format,
    }
    if should_send_temperature(model):
        req["temperature"] = float(temperature)
    if int(max_tokens) > 0:
        req[choose_token_param_name(model, token_param)] = int(max_tokens)

    raw: Dict[str, Any] = {
        "request": to_jsonable(req),
    }

    try:
        completion = client.chat.completions.create(**req)
    except Exception as exc:
        return None, raw, str(exc).splitlines()[0]

    raw["completion"] = to_jsonable(completion)
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    message = getattr(choice, "message", None) if choice is not None else None
    raw["choice"] = to_jsonable(choice)
    raw["message"] = to_jsonable(message)

    content = normalize_chat_content(getattr(message, "content", None) if message is not None else "")
    raw["raw_content"] = content

    # Some SDK/model combinations can populate a parsed object instead of text content.
    if not content and message is not None:
        parsed = getattr(message, "parsed", None)
        raw["parsed"] = to_jsonable(parsed)
        if parsed is not None:
            try:
                if isinstance(parsed, dict):
                    return DecompositionPayload.model_validate(parsed), raw, None
                if hasattr(parsed, "model_dump"):
                    return DecompositionPayload.model_validate(parsed.model_dump()), raw, None
            except Exception as exc:
                return None, raw, f"parse_error: {str(exc).splitlines()[0]}"

    if not content:
        refusal = getattr(message, "refusal", None) if message is not None else None
        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        raw["finish_reason"] = finish_reason
        raw["refusal"] = to_jsonable(refusal)
        return None, raw, f"Empty model response content (finish_reason={finish_reason}, refusal={refusal})"

    try:
        return DecompositionPayload.model_validate_json(content), raw, None
    except Exception as exc:
        return None, raw, f"parse_error: {str(exc).splitlines()[0]}"


def process_rows_single(args: argparse.Namespace) -> None:
    rows = load_json_rows(args.input_json)
    cols = load_schema_columns(args.schema_json)
    q_keys = parse_comma_keys(args.question_keys)
    sql_keys = parse_comma_keys(args.sql_keys)

    sliced = slice_rows(rows, args.start, args.limit, args.mode)
    response_format = pydantic_to_response_format(DecompositionPayload)
    client = get_client(args.api_key, args.api_base)
    raw_output_path = resolve_raw_output_path(args)

    out_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    base_idx = max(0, int(args.start))

    for i, row in enumerate(sliced):
        idx = base_idx + i + 1
        item_id = row.get(args.id_key)
        q_text, q_key = pick_first_nonempty(row, q_keys)
        sql_text, sql_key = pick_first_nonempty(row, sql_keys)

        if not q_text or not sql_text:
            out_rows.append(
                make_output_row(
                    idx=idx,
                    item_id=item_id,
                    original_question=q_text,
                    gt_sql=sql_text,
                    q_key=q_key,
                    sql_key=sql_key,
                    model=args.model,
                    payload=None,
                    status="missing_input",
                    error="Missing question or SQL",
                )
            )
            raw_rows.append(
                {
                    "idx": idx,
                    "item_id": item_id,
                    "status": "missing_input",
                    "error": "Missing question or SQL",
                    "raw": {},
                }
            )
            continue

        prompt = build_user_prompt(
            table_name=args.table_name,
            schema_cols=cols,
            original_question=q_text,
            gt_sql=sql_text,
        )

        try:
            payload, raw, err = call_decompose_once(
                client=client,
                model=args.model,
                response_format=response_format,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                token_param=args.token_param,
                prompt=prompt,
            )
            if err is not None or payload is None:
                final_err = err or "Empty parsed payload"
                out_rows.append(
                    make_output_row(
                        idx=idx,
                        item_id=item_id,
                        original_question=q_text,
                        gt_sql=sql_text,
                        q_key=q_key,
                        sql_key=sql_key,
                        model=args.model,
                        payload=None,
                        status="error",
                        error=final_err,
                    )
                )
            else:
                out_rows.append(
                    make_output_row(
                        idx=idx,
                        item_id=item_id,
                        original_question=q_text,
                        gt_sql=sql_text,
                        q_key=q_key,
                        sql_key=sql_key,
                        model=args.model,
                        payload=payload,
                        status="ok",
                    )
                )
            raw_rows.append(
                {
                    "idx": idx,
                    "item_id": item_id,
                    "status": "error" if (err is not None or payload is None) else "ok",
                    "error": (err or "Empty parsed payload") if payload is None else err,
                    "raw": raw,
                }
            )
        except Exception as exc:
            out_rows.append(
                make_output_row(
                    idx=idx,
                    item_id=item_id,
                    original_question=q_text,
                    gt_sql=sql_text,
                    q_key=q_key,
                    sql_key=sql_key,
                    model=args.model,
                    payload=None,
                    status="error",
                    error=str(exc).splitlines()[0],
                )
            )
            raw_rows.append(
                {
                    "idx": idx,
                    "item_id": item_id,
                    "status": "error",
                    "error": str(exc).splitlines()[0],
                    "raw": {},
                }
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(sliced):
            print(f"single mode processed {i + 1}/{len(sliced)}")

    ensure_parent(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    write_jsonl(raw_output_path, raw_rows)

    print("\n=== single mode complete ===")
    print(f"rows_in_input:  {len(rows)}")
    print(f"rows_processed: {len(sliced)}")
    print(f"output_json:    {args.output_json}")
    print(f"raw_output:     {raw_output_path}")


def prepare_batch_files(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_json_rows(args.input_json)
    cols = load_schema_columns(args.schema_json)
    q_keys = parse_comma_keys(args.question_keys)
    sql_keys = parse_comma_keys(args.sql_keys)

    sliced = slice_rows(rows, args.start, args.limit, args.mode)
    response_format = pydantic_to_response_format(DecompositionPayload)

    ensure_parent(args.batch_jsonl_path)
    ensure_parent(args.batch_meta_path)

    requests_meta: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    base_idx = max(0, int(args.start))

    with open(args.batch_jsonl_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(sliced):
            idx = base_idx + i + 1
            item_id = row.get(args.id_key)
            q_text, q_key = pick_first_nonempty(row, q_keys)
            sql_text, sql_key = pick_first_nonempty(row, sql_keys)

            if not q_text or not sql_text:
                skipped_rows.append(
                    make_output_row(
                        idx=idx,
                        item_id=item_id,
                        original_question=q_text,
                        gt_sql=sql_text,
                        q_key=q_key,
                        sql_key=sql_key,
                        model=args.model,
                        payload=None,
                        status="missing_input",
                        error="Missing question or SQL",
                    )
                )
                continue

            prompt = build_user_prompt(
                table_name=args.table_name,
                schema_cols=cols,
                original_question=q_text,
                gt_sql=sql_text,
            )

            custom_id = f"decomp-{idx}"
            body: Dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "response_format": response_format,
            }
            if should_send_temperature(args.model):
                body["temperature"] = float(args.temperature)
            if int(args.max_tokens) > 0:
                body[choose_token_param_name(args.model, args.token_param)] = int(args.max_tokens)

            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

            requests_meta.append(
                {
                    "custom_id": custom_id,
                    "idx": idx,
                    "item_id": item_id,
                    "original_question": q_text,
                    "gt_sql": sql_text,
                    "source_question_key": q_key,
                    "source_sql_key": sql_key,
                }
            )

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "input_json": args.input_json,
        "schema_json": args.schema_json,
        "table_name": args.table_name,
        "model": args.model,
        "start": int(args.start),
        "limit": int(args.limit),
        "rows_in_input": len(rows),
        "rows_in_slice": len(sliced),
        "request_count": len(requests_meta),
        "batch_jsonl_path": args.batch_jsonl_path,
        "requests": requests_meta,
        "skipped_rows": skipped_rows,
    }

    with open(args.batch_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== batch prepare complete ===")
    print(f"batch_jsonl_path: {args.batch_jsonl_path}")
    print(f"batch_meta_path:  {args.batch_meta_path}")
    print(f"request_count:    {len(requests_meta)}")
    print(f"skipped_rows:     {len(skipped_rows)}")

    return meta


def poll_batch_until_terminal(client: Any, batch_id: str, poll_every_s: float) -> Any:
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        status = getattr(b, "status", None)
        print(f"[batch {batch_id}] status={status}")
        if status in terminal:
            return b
        time.sleep(float(max(0.5, poll_every_s)))


def download_file_to_path(client: Any, file_id: str, out_path: str) -> str:
    content = client.files.content(file_id).read()
    ensure_parent(out_path)
    Path(out_path).write_bytes(content)
    return out_path


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def extract_message_content_from_batch_line(line_obj: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    if line_obj.get("error"):
        return "", f"batch_error: {line_obj.get('error')}"

    response = line_obj.get("response")
    if not isinstance(response, dict):
        return "", "missing_response"

    status_code = response.get("status_code")
    if status_code != 200:
        return "", f"http_status_{status_code}"

    body = response.get("body")
    if not isinstance(body, dict):
        return "", "missing_response_body"

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", "missing_choices"

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return "", "missing_message"

    content = normalize_chat_content(msg.get("content"))
    if not content:
        return "", "empty_message_content"
    return content, None


def build_output_from_batch_files(
    *,
    batch_meta_path: str,
    batch_output_jsonl_path: str,
    output_json_path: str,
    raw_output_jsonl_path: str = "",
) -> None:
    with open(batch_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    requests = meta.get("requests") or []
    skipped_rows = meta.get("skipped_rows") or []

    req_map: Dict[str, Dict[str, Any]] = {}
    for r in requests:
        if isinstance(r, dict) and r.get("custom_id"):
            req_map[str(r["custom_id"])] = r

    output_lines = read_jsonl(batch_output_jsonl_path)
    raw_rows: List[Dict[str, Any]] = []

    parsed_by_custom_id: Dict[str, Dict[str, Any]] = {}
    for line_obj in output_lines:
        custom_id = str(line_obj.get("custom_id") or "").strip()
        if not custom_id or custom_id not in req_map:
            continue

        req_meta = req_map[custom_id]
        idx = int(req_meta.get("idx"))
        item_id = req_meta.get("item_id")
        q_text = str(req_meta.get("original_question") or "")
        sql_text = str(req_meta.get("gt_sql") or "")
        q_key = req_meta.get("source_question_key")
        sql_key = req_meta.get("source_sql_key")
        model = str(meta.get("model") or "")

        content, err = extract_message_content_from_batch_line(line_obj)
        raw_rows.append(
            {
                "custom_id": custom_id,
                "idx": idx,
                "item_id": item_id,
                "extract_error": err,
                "extracted_content": content,
                "raw_line": line_obj,
            }
        )
        if err:
            parsed_by_custom_id[custom_id] = make_output_row(
                idx=idx,
                item_id=item_id,
                original_question=q_text,
                gt_sql=sql_text,
                q_key=q_key,
                sql_key=sql_key,
                model=model,
                payload=None,
                status="error",
                error=err,
                custom_id=custom_id,
            )
            continue

        try:
            payload = DecompositionPayload.model_validate_json(content)
            parsed_by_custom_id[custom_id] = make_output_row(
                idx=idx,
                item_id=item_id,
                original_question=q_text,
                gt_sql=sql_text,
                q_key=q_key,
                sql_key=sql_key,
                model=model,
                payload=payload,
                status="ok",
                custom_id=custom_id,
            )
        except Exception as exc:
            parsed_by_custom_id[custom_id] = make_output_row(
                idx=idx,
                item_id=item_id,
                original_question=q_text,
                gt_sql=sql_text,
                q_key=q_key,
                sql_key=sql_key,
                model=model,
                payload=None,
                status="error",
                error=f"parse_error: {str(exc).splitlines()[0]}",
                custom_id=custom_id,
            )

    final_rows: List[Dict[str, Any]] = []

    # Add skipped rows first (already in final schema format).
    for r in skipped_rows:
        if isinstance(r, dict):
            final_rows.append(r)

    # Add requested rows in deterministic idx order; fill missing outputs.
    for req in sorted(requests, key=lambda x: int(x.get("idx", 0))):
        custom_id = str(req.get("custom_id") or "")
        if not custom_id:
            continue
        if custom_id in parsed_by_custom_id:
            final_rows.append(parsed_by_custom_id[custom_id])
            continue

        # Request existed but no output line found.
        final_rows.append(
            make_output_row(
                idx=int(req.get("idx", 0)),
                item_id=req.get("item_id"),
                original_question=str(req.get("original_question") or ""),
                gt_sql=str(req.get("gt_sql") or ""),
                q_key=req.get("source_question_key"),
                sql_key=req.get("source_sql_key"),
                model=str(meta.get("model") or ""),
                payload=None,
                status="error",
                error="missing_batch_output_line",
                custom_id=custom_id,
            )
        )

    final_rows.sort(key=lambda r: int(r.get("idx", 0)))

    ensure_parent(output_json_path)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)
    if raw_output_jsonl_path:
        write_jsonl(raw_output_jsonl_path, raw_rows)

    ok_count = sum(1 for r in final_rows if r.get("status") == "ok")
    err_count = sum(1 for r in final_rows if r.get("status") == "error")
    miss_count = sum(1 for r in final_rows if r.get("status") == "missing_input")

    print("\n=== batch collect complete ===")
    print(f"rows_total:      {len(final_rows)}")
    print(f"rows_ok:         {ok_count}")
    print(f"rows_error:      {err_count}")
    print(f"rows_missing:    {miss_count}")
    print(f"output_json:     {output_json_path}")
    if raw_output_jsonl_path:
        print(f"raw_output:      {raw_output_jsonl_path}")


def run_prepare_batch(args: argparse.Namespace) -> None:
    prepare_batch_files(args)


def run_batch_run(args: argparse.Namespace) -> None:
    meta = prepare_batch_files(args)
    if int(meta.get("request_count", 0)) <= 0:
        print("No batch requests were created; nothing to submit.")
        return

    client = get_client(args.api_key, args.api_base)

    with open(args.batch_jsonl_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"task": "decompose_clinical_query"},
    )

    batch_id = batch.id
    print(f"submitted batch_id={batch_id}")

    # Update meta with submission data.
    with open(args.batch_meta_path, "r", encoding="utf-8") as f:
        meta2 = json.load(f)
    meta2["batch_id"] = batch_id
    meta2["batch_input_file_id"] = batch_input_file.id
    with open(args.batch_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta2, f, ensure_ascii=False, indent=2)

    terminal = poll_batch_until_terminal(client, batch_id, args.poll_every_s)
    status = getattr(terminal, "status", None)

    # Persist terminal metadata.
    with open(args.batch_meta_path, "r", encoding="utf-8") as f:
        meta3 = json.load(f)
    meta3["batch_terminal_status"] = status
    meta3["batch_output_file_id"] = getattr(terminal, "output_file_id", None)
    meta3["batch_error_file_id"] = getattr(terminal, "error_file_id", None)
    with open(args.batch_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta3, f, ensure_ascii=False, indent=2)

    if status != "completed":
        print(f"Batch finished with non-completed status: {status}")
        print(f"See metadata: {args.batch_meta_path}")
        return

    output_file_id = getattr(terminal, "output_file_id", None)
    error_file_id = getattr(terminal, "error_file_id", None)

    if not output_file_id:
        raise RuntimeError("Batch completed but output_file_id is missing")

    download_file_to_path(client, output_file_id, args.batch_output_jsonl_path)
    print(f"downloaded batch output: {args.batch_output_jsonl_path}")

    if error_file_id:
        download_file_to_path(client, error_file_id, args.batch_error_jsonl_path)
        print(f"downloaded batch errors: {args.batch_error_jsonl_path}")

    build_output_from_batch_files(
        batch_meta_path=args.batch_meta_path,
        batch_output_jsonl_path=args.batch_output_jsonl_path,
        output_json_path=args.output_json,
        raw_output_jsonl_path=resolve_raw_output_path(args),
    )


def run_batch_collect(args: argparse.Namespace) -> None:
    client: Optional[Any] = None

    if args.batch_id:
        client = get_client(args.api_key, args.api_base)
        b = client.batches.retrieve(args.batch_id)
        status = getattr(b, "status", None)
        print(f"batch_id={args.batch_id} status={status}")

        output_file_id = getattr(b, "output_file_id", None)
        error_file_id = getattr(b, "error_file_id", None)

        if output_file_id:
            download_file_to_path(client, output_file_id, args.batch_output_jsonl_path)
            print(f"downloaded batch output: {args.batch_output_jsonl_path}")
        else:
            raise RuntimeError("No output_file_id found for provided batch_id")

        if error_file_id:
            download_file_to_path(client, error_file_id, args.batch_error_jsonl_path)
            print(f"downloaded batch errors: {args.batch_error_jsonl_path}")

    if not os.path.exists(args.batch_meta_path):
        raise FileNotFoundError(f"batch meta file not found: {args.batch_meta_path}")
    if not os.path.exists(args.batch_output_jsonl_path):
        raise FileNotFoundError(f"batch output JSONL not found: {args.batch_output_jsonl_path}")

    build_output_from_batch_files(
        batch_meta_path=args.batch_meta_path,
        batch_output_jsonl_path=args.batch_output_jsonl_path,
        output_json_path=args.output_json,
        raw_output_jsonl_path=resolve_raw_output_path(args),
    )


def main() -> None:
    args = parse_args()
    args.api_key = resolve_api_key(args.api_key, args.env_file)

    if args.mode == "single":
        process_rows_single(args)
        return

    if args.mode == "prepare_batch":
        run_prepare_batch(args)
        return

    if args.mode == "batch_run":
        run_batch_run(args)
        return

    if args.mode == "batch_collect":
        run_batch_collect(args)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
