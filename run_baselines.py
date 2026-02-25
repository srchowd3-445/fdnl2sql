#!/usr/bin/env python3
"""OpenAI-compatible baseline runner for SQL generation."""

import argparse
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    from openai.types import ResponseFormatJSONSchema
except Exception:
    ResponseFormatJSONSchema = None
try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None

from utils import (
    append_jsonl,
    build_checkpoint_meta,
    build_prompt,
    chunked_indices,
    extract_sql,
    fetch_schema_columns_from_db,
    fetch_schema_columns_from_json,
    fetch_schema_value_hints,
    is_retryable_provider_error,
    load_checkpoint,
    load_input,
    load_text,
    make_row_result,
    openai_completion_to_meta,
    parse_openai_text,
    parse_question_keys,
    persist_outputs,
    pick_question,
    openai_token_logprob_payload,
    render_schema_hints,
    setup_logger,
    structured_logprob_payload,
)

try:
    from structured_logprobs import add_logprobs
except Exception:
    add_logprobs = None


DEFAULT_COT_SUFFIX = (
    "Think step-by-step internally to ensure correctness, but output ONLY the final SQL query."
    
)


class SQLResponse(BaseModel if BaseModel is not None else object):
    sql: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SQL baselines via OpenAI-compatible endpoint.")

    # Core I/O
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--input_format", choices=["auto", "jsonl", "json"], default="auto")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--error_path", default="")
    ap.add_argument("--log_dir", default="results/logs")
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=10)

    # Data mapping
    ap.add_argument("--question_keys", default="natural_question")
    ap.add_argument("--id_key", default="item_id")
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--schema_file", default="data/schema.json")
    ap.add_argument("--db_path", default="data/database.db")

    # Prompting
    ap.add_argument("--prompt_style", choices=["zero_shot", "few_shot", "cot"], required=True)
    ap.add_argument("--zs_prompt_file", default="prompting/zero_shot_sql_expert.txt")
    ap.add_argument("--few_shot_file", default="prompting/few_shot_sql.txt")
    ap.add_argument("--cot_suffix", default=DEFAULT_COT_SUFFIX)
    ap.add_argument("--use_schema_hints", type=int, default=1)

    # Model/provider
    ap.add_argument("--api_base", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="dummy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--batch_concurrency", type=int, default=8, help="Parallel requests per batch")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument("--use_pydantic_schema", type=int, default=1)

    # Logprobs
    ap.add_argument("--logprob_mode", choices=["structured", "none"], default="structured")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed/importable. Install it before running this script.")

    if not args.error_path:
        args.error_path = args.output_path + ".errors.jsonl"
    if not args.checkpoint_path:
        args.checkpoint_path = args.output_path + ".checkpoint.json"

    for path in [args.input_path, args.schema_file, args.zs_prompt_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
    if args.prompt_style == "few_shot" and not os.path.exists(args.few_shot_file):
        raise FileNotFoundError(f"few_shot file not found: {args.few_shot_file}")

    if args.logprob_mode == "structured" and add_logprobs is None:
        raise RuntimeError("structured_logprobs is not installed/importable but logprob_mode=structured was requested.")
    if bool(args.use_pydantic_schema):
        if BaseModel is None:
            raise RuntimeError("pydantic is required when --use_pydantic_schema=1")
        if ResponseFormatJSONSchema is None:
            raise RuntimeError("openai.types.ResponseFormatJSONSchema is required when --use_pydantic_schema=1")


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_response_schema(enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    json_schema = SQLResponse.model_json_schema()
    response_schema = ResponseFormatJSONSchema.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "SQLResponse",
                "schema": json_schema,
            },
        }
    )
    return response_schema.model_dump(by_alias=True)


def _strip_prefix(text: str, prefix: str) -> str:
    t = (text or "").strip()
    p = prefix.strip()
    if t.lower().startswith(p.lower()):
        return t[len(p):].strip()
    return t


# def parse_few_shot_examples(few_shot_text: str) -> List[Tuple[str, str]]:
#     text = (few_shot_text or "").strip()
#     if not text:
#         return []

#     # Normalize optional XML-like wrappers used in the first example.
#     text = re.sub(r"</?\s*user\s*>", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"</?\s*assistant\s*>", "", text, flags=re.IGNORECASE)

#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     examples: List[Tuple[str, str]] = []
#     i = 0
#     while i < len(lines):
#         if lines[i].lower().startswith("question:"):
#             q = _strip_prefix(lines[i], "Question:")
#             if i + 1 < len(lines) and lines[i + 1].lower().startswith("sql:"):
#                 s = _strip_prefix(lines[i + 1], "SQL:")
#                 if q and s:
#                     examples.append((q, s))
#                 i += 2
#                 continue
#         i += 1
#     return examples


def parse_few_shot_examples(few_shot_text: str) -> List[Tuple[str, str]]:
    text = (few_shot_text or "").strip()
    if not text:
        return []

    # remove wrappers like <USER>...</USER>, <ASSISTANT>...</ASSISTANT>
    text = re.sub(r"</?\s*user\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*assistant\s*>", "", text, flags=re.IGNORECASE)

    # Normalize bracketed tags
    text = text.replace("[RESPONSE]", "RESPONSE:")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    examples: List[Tuple[str, str]] = []

    q: Optional[str] = None
    sql_lines: List[str] = []
    mode: Optional[str] = None  # "q" or "sql"

    def flush():
        nonlocal q, sql_lines
        if q and sql_lines:
            s = " ".join(sql_lines).strip()
            if s:
                examples.append((q.strip(), s))
        q = None
        sql_lines = []

    for ln in lines:
        low = ln.lower()

        if low.startswith("question:"):
            flush()
            q = _strip_prefix(ln, "Question:")
            mode = "q"
            continue

        # Accept "SQL:" or "RESPONSE:" as SQL header
        if low.startswith("sql:") or low.startswith("response:"):
            sql_lines = [_strip_prefix(ln, "SQL:") if low.startswith("sql:") else _strip_prefix(ln, "RESPONSE:")]
            mode = "sql"
            continue

        # If we're currently collecting SQL, allow multi-line continuation
        if mode == "sql":
            # stop if a new question begins
            if low.startswith("question:"):
                flush()
                q = _strip_prefix(ln, "Question:")
                mode = "q"
            else:
                sql_lines.append(ln)
            continue

    flush()
    return examples

# def build_messages_for_row(
#     *,
#     prompt_style: str,
#     question: str,
#     prompt: str,
#     zs_template: str,
#     schema_cols: List[str],
#     schema_hints_text: str,
#     few_shot_examples: List[Tuple[str, str]],
# ) -> List[Dict[str, str]]:
#     if prompt_style != "few_shot" or not few_shot_examples:
#         return build_messages(prompt)

#     # Keep few-shot aligned with chat roles:
#     # Question -> user, SQL -> assistant, then final user question.
#     base_zero_prompt = build_prompt(
#         question="",
#         prompt_style="zero_shot",
#         zs_template=zs_template,
#         few_shot_text="",
#         schema_cols=schema_cols,
#         schema_hints_text=schema_hints_text,
#         cot_suffix="",
#     )
#     cut = base_zero_prompt.rfind("Question:")
#     system_prompt = base_zero_prompt[:cut].strip() if cut >= 0 else base_zero_prompt.strip()

#     messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
#     for ex_q, ex_sql in few_shot_examples:
#         messages.append({"role": "user", "content": f"Question: {ex_q}\nGenerate SQL:"})
#         messages.append({"role": "assistant", "content": f"SQL: {ex_sql}"})
#     messages.append({"role": "user", "content": f"Question: {question}\nGenerate SQL:"})
#     return messages

def build_messages_for_row(
    *,
    prompt_style: str,
    question: str,
    prompt: str,
    zs_template: str,
    schema_cols: List[str],
    schema_hints_text: str,
    few_shot_examples: List[Tuple[str, str]],
    use_pydantic_schema: bool,
) -> List[Dict[str, str]]:
    if prompt_style != "few_shot" or not few_shot_examples:
        return build_messages(prompt)

    base_zero_prompt = build_prompt(
        question="",
        prompt_style="zero_shot",
        zs_template=zs_template,
        few_shot_text="",
        schema_cols=schema_cols,
        schema_hints_text=schema_hints_text,
        cot_suffix="",
    )
    cut = base_zero_prompt.rfind("Question:")
    system_prompt = base_zero_prompt[:cut].strip() if cut >= 0 else base_zero_prompt.strip()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for ex_q, ex_sql in few_shot_examples:
        messages.append({"role": "user", "content": ex_q})
        if use_pydantic_schema:
            messages.append({"role": "assistant", "content": json.dumps({"sql": ex_sql})})
        else:
            messages.append({"role": "assistant", "content": ex_sql})

    messages.append({"role": "user", "content": question})
    return messages


def _run_one_openai_call(
    client: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    logger: Any,
    response_schema: Optional[Dict[str, Any]],
) -> Any:
    completion: Any = None
    attempts = max(1, int(args.num_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            req: Dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "logprobs": (args.logprob_mode == "structured"),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
            }
            if response_schema is not None:
                req["response_format"] = response_schema
            completion = client.chat.completions.create(
                **req
            )
            break
        except Exception as e:
            retryable = is_retryable_provider_error(e)
            if attempt >= attempts or not retryable:
                completion = e
                break
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retry openai call attempt=%d/%d sleep=%.1fs err=%s",
                attempt,
                attempts,
                backoff,
                str(e).splitlines()[0] if str(e) else "unknown",
            )
            time.sleep(backoff)
    return completion


def run_batch_openai(
    client: Any,
    messages_batch: List[List[Dict[str, str]]],
    args: argparse.Namespace,
    logger: Any,
    response_schema: Optional[Dict[str, Any]],
) -> List[Any]:
    outputs: List[Any] = [None] * len(messages_batch)
    max_workers = max(1, min(int(args.batch_concurrency), len(messages_batch) if messages_batch else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_openai_call, client, messages, args, logger, response_schema): idx
            for idx, messages in enumerate(messages_batch)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                outputs[idx] = future.result()
            except Exception as e:
                outputs[idx] = e
    return outputs


def run() -> None:
    args = parse_args()
    validate_args(args)
    logger = setup_logger(args.log_dir, args.output_path)

    question_keys = parse_question_keys(args.question_keys)
    if not question_keys:
        raise ValueError("--question_keys resolved to empty list")

    payload = load_input(args.input_path, args.input_format)
    rows = payload.records
    input_format_used = payload.format_used
    if rows:
        has_any_q = False
        for r in rows:
            if pick_question(r, question_keys):
                has_any_q = True
                break
        if not has_any_q:
            sample_keys = sorted(list(rows[0].keys()))
            raise RuntimeError(
                f"No rows contain any of question_keys={question_keys} in input={args.input_path}. "
                f"Sample available keys: {sample_keys}"
            )

    schema_cols = fetch_schema_columns_from_json(args.schema_file)
    schema_hints: Dict[str, Dict[str, Any]] = {}
    if bool(args.use_schema_hints) and args.db_path and os.path.exists(args.db_path):
        try:
            conn = sqlite3.connect(args.db_path)
            db_schema_cols = fetch_schema_columns_from_db(conn, args.table_name)
            if db_schema_cols:
                schema_hints = fetch_schema_value_hints(conn, args.table_name, db_schema_cols)
            conn.close()
        except Exception as e:
            logger.warning("Schema hints unavailable from db_path=%s: %s", args.db_path, e)
    schema_hints_text = render_schema_hints(schema_cols, schema_hints) if bool(args.use_schema_hints) else ""

    zs_template = load_text(args.zs_prompt_file)
    few_shot_text = load_text(args.few_shot_file) if args.prompt_style == "few_shot" else ""
    few_shot_examples = parse_few_shot_examples(few_shot_text) if args.prompt_style == "few_shot" else []

    ckpt = load_checkpoint(args.checkpoint_path) if bool(args.resume) else {"completed": {}, "meta": {}}
    completed: Dict[str, Dict[str, Any]] = ckpt.get("completed", {}) if isinstance(ckpt, dict) else {}
    selected_ids = {str(r.get(args.id_key) or f"idx_{i}") for i, r in enumerate(rows)}
    completed = {k: v for k, v in completed.items() if k in selected_ids}

    logger.info("=========== RUN START ===========")
    logger.info("Input: %s", args.input_path)
    logger.info("Input format used: %s", input_format_used)
    logger.info("Output: %s", args.output_path)
    logger.info("Error log: %s", args.error_path)
    logger.info("Checkpoint: %s", args.checkpoint_path)
    logger.info("Rows selected: %d", len(rows))
    logger.info("Resuming: %s", bool(args.resume))
    logger.info("Already completed: %d", len(completed))
    logger.info("Prompt style: %s", args.prompt_style)
    logger.info("Backend: openai_compat")
    logger.info("Model: %s", args.model)
    logger.info("Logprob mode: %s", args.logprob_mode)
    logger.info("Batch size: %d", max(1, int(args.batch_size)))
    logger.info("Batch concurrency: %d", max(1, int(args.batch_concurrency)))
    logger.info("Use pydantic response schema: %s", bool(args.use_pydantic_schema))
    logger.info("Schema columns: %d", len(schema_cols))
    logger.info("Schema hints columns: %d", len(schema_hints))

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    response_schema = build_response_schema(bool(args.use_pydantic_schema))

    pending: List[Tuple[str, Dict[str, Any], List[Dict[str, str]]]] = []
    for i, row in enumerate(rows):
        item_id = str(row.get(args.id_key) or f"idx_{i}")
        if item_id in completed:
            continue
        question = pick_question(row, question_keys)
        prompt = build_prompt(
            question=question,
            prompt_style=args.prompt_style,
            zs_template=zs_template,
            few_shot_text=few_shot_text,
            schema_cols=schema_cols,
            schema_hints_text=schema_hints_text,
            cot_suffix=args.cot_suffix,
        )
        # messages = build_messages_for_row(
        #     prompt_style=args.prompt_style,
        #     question=question,
        #     prompt=prompt,
        #     zs_template=zs_template,
        #     schema_cols=schema_cols,
        #     schema_hints_text=schema_hints_text,
        #     few_shot_examples=few_shot_examples,
        # )

        messages = build_messages_for_row(
            prompt_style=args.prompt_style,
            question=question,
            prompt=prompt,
            zs_template=zs_template,
            schema_cols=schema_cols,
            schema_hints_text=schema_hints_text,
            few_shot_examples=few_shot_examples,
            use_pydantic_schema=bool(args.use_pydantic_schema),
            )
        pending.append((item_id, row, messages))

    run_start = time.time()
    done_in_run = 0
    processed_since_save = 0
    success_count = 0
    error_count = 0
    conf_values: List[float] = []
    saved_error_ids = set()

    batches = chunked_indices(len(pending), args.batch_size)
    for batch_index, (lo, hi) in enumerate(batches):
        batch = pending[lo:hi]
        messages_batch = [x[2] for x in batch]
        batch_start = time.time()
        outputs = run_batch_openai(client, messages_batch, args, logger, response_schema)

        for idx_in_batch, (item_id, row_obj, _) in enumerate(batch):
            question = pick_question(row_obj, question_keys)
            raw_output = ""
            pred_sql = ""
            error: Optional[str] = None
            field_logprobs: Dict[str, Any] = {}
            field_confidence: Dict[str, Any] = {}
            conf_overall: Optional[float] = None
            model_meta: Dict[str, Any] = {"batch_index": batch_index, "batch_item_index": idx_in_batch}

            try:
                out_obj = outputs[idx_in_batch] if idx_in_batch < len(outputs) else None
                if isinstance(out_obj, Exception):
                    raise out_obj

                raw_output = parse_openai_text(out_obj)
                model_meta.update(openai_completion_to_meta(out_obj))
                if bool(args.use_pydantic_schema):
                    parsed_json = json.loads(raw_output)
                    pred_sql = (parsed_json.get("sql") or "").strip()
                    model_meta["response_schema_used"] = True
                else:
                    pred_sql = extract_sql(raw_output)
                if not pred_sql:
                    pred_sql = extract_sql(raw_output)
                if not question:
                    error = "Empty question from configured question_keys"
                elif not pred_sql:
                    error = "Failed to extract SQL from model output"
                elif args.logprob_mode == "structured":
                    try:
                        field_logprobs, field_confidence, conf_overall = structured_logprob_payload(
                            out_obj, add_logprobs
                        )
                    except Exception as lp_err:
                        # structured_logprobs can fail for non-JSON outputs; fallback to token logprobs.
                        model_meta["structured_logprob_error"] = str(lp_err).splitlines()[0] if str(lp_err) else str(lp_err)
                        field_logprobs, field_confidence, conf_overall = openai_token_logprob_payload(out_obj)
                        model_meta["logprob_fallback"] = "openai_token_logprobs"
            except Exception as e:
                error = str(e)
                model_meta.setdefault("provider_call_ok", False)

            result = make_row_result(
                item_id=item_id,
                question=question,
                prompt_style=args.prompt_style,
                backend="openai_compat",
                model=args.model,
                raw_model_output=raw_output,
                pred_sql=pred_sql,
                logprob_mode=args.logprob_mode,
                field_logprobs=field_logprobs,
                field_confidence=field_confidence,
                confidence_overall=conf_overall,
                model_meta=model_meta,
                error=error,
            )
            completed[item_id] = result

            done_in_run += 1
            processed_since_save += 1
            if error:
                error_count += 1
                if item_id not in saved_error_ids:
                    append_jsonl(args.error_path, result)
                    saved_error_ids.add(item_id)
            else:
                success_count += 1
            if isinstance(conf_overall, (int, float)):
                conf_values.append(float(conf_overall))

        batch_elapsed = max(1e-9, time.time() - batch_start)
        logger.info(
            "Batch %d/%d processed=%d elapsed=%.2fs throughput=%.2f rec/s completed=%d/%d success=%d error=%d",
            batch_index + 1,
            len(batches),
            len(batch),
            batch_elapsed,
            len(batch) / batch_elapsed,
            len(completed),
            len(rows),
            success_count,
            error_count,
        )

        if processed_since_save >= max(1, int(args.save_every)):
            persist_outputs(
                output_path=args.output_path,
                output_format=input_format_used,
                input_rows=rows,
                completed=completed,
                id_key=args.id_key,
                checkpoint_path=args.checkpoint_path,
                checkpoint_meta=build_checkpoint_meta(args, len(rows), input_format_used),
            )
            processed_since_save = 0

    persist_outputs(
        output_path=args.output_path,
        output_format=input_format_used,
        input_rows=rows,
        completed=completed,
        id_key=args.id_key,
        checkpoint_path=args.checkpoint_path,
        checkpoint_meta=build_checkpoint_meta(args, len(rows), input_format_used),
    )

    total_elapsed = max(1e-9, time.time() - run_start)
    non_empty_sql = sum(1 for v in completed.values() if (v.get("pred_sql") or "").strip())
    avg_conf = mean(conf_values) if conf_values else None

    logger.info("=========== RUN SUMMARY ===========")
    logger.info("Rows selected: %d", len(rows))
    logger.info("Rows completed in checkpoint: %d", len(completed))
    logger.info("Processed this run: %d", done_in_run)
    logger.info("Non-empty predicted SQL: %d", non_empty_sql)
    logger.info("Success count (this run): %d", success_count)
    logger.info("Error count (this run): %d", error_count)
    logger.info("Avg confidence_overall (this run): %s", f"{avg_conf:.6f}" if avg_conf is not None else "NA")
    logger.info("Total runtime: %.2fs", total_elapsed)
    logger.info("===================================")


if __name__ == "__main__":
    run()
