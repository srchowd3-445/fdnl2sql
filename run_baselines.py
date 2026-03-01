#!/usr/bin/env python3
"""OpenAI-compatible baseline runner for SQL generation."""

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from pydantic import BaseModel

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
    parse_openai_text,
    parse_question_keys,
    persist_outputs,
    pick_question,
    render_schema_hints,
    setup_logger,
    structured_logprob_payload,
    openai_completion_to_meta
)

try:
    from structured_logprobs import add_logprobs
except Exception:
    add_logprobs = None


# =====================================================
# Pydantic Models
# =====================================================

class SQLOnlyResponse(BaseModel):
    sql: str


class COTResponse(BaseModel):
    thinking: str
    sql: str


def build_response_schema_from_model(model_cls):
    json_schema = model_cls.model_json_schema()
    response_schema = ResponseFormatJSONSchema.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": json_schema,
            },
        }
    )
    return response_schema.model_dump(by_alias=True)


# =====================================================
# Args
# =====================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", required=True)
    ap.add_argument("--input_format", choices=["auto", "jsonl", "json"], default="auto")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--error_path", default="")
    ap.add_argument("--log_dir", default="results/logs")
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=10)

    ap.add_argument("--question_keys", default="natural_question")
    ap.add_argument("--id_key", default="item_id")
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--db_path", default="")

    ap.add_argument("--prompt_style", choices=["zero_shot", "few_shot", "cot"], required=True)
    ap.add_argument("--zs_prompt_file", required=True)
    ap.add_argument("--few_shot_file", default="")
    ap.add_argument("--cot_prompt_file", default="")
    ap.add_argument("--use_schema_hints", type=int, default=1)

    ap.add_argument("--api_base", required=True)
    ap.add_argument("--api_key", default="dummy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--batch_concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)

    # <-- Re-introduced retry argument
    ap.add_argument("--num_retries", type=int, default=2)

    ap.add_argument("--use_pydantic_schema", type=int, default=1)
    ap.add_argument("--logprob_mode", choices=["structured", "none"], default="structured")

    return ap.parse_args()


# =====================================================
# OpenAI Call with retry/backoff logic
# =====================================================

def _run_one_call_with_retries(client, messages, args, logger, response_schema):
    completion = None
    attempts = max(1, int(args.num_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            req: Dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "logprobs": args.logprob_mode == "structured",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
            }
            if response_schema is not None:
                req["response_format"] = response_schema
            completion = client.chat.completions.create(**req)
            # success
            return completion
        except Exception as e:
            retryable = is_retryable_provider_error(e)
            if attempt >= attempts or not retryable:
                # final failure
                logger.warning("OpenAI call failed (attempt %d/%d): %s", attempt, attempts, str(e).splitlines()[0] if str(e) else "")
                raise
            # else retry
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retrying OpenAI call attempt=%d/%d after %.1fs (err=%s)",
                attempt,
                attempts,
                backoff,
                str(e).splitlines()[0] if str(e) else "",
            )
            time.sleep(backoff)
    return completion


def run_batch(client, messages_batch, args, logger, response_schema):
    outputs = [None] * len(messages_batch)
    max_workers = max(1, min(int(args.batch_concurrency), len(messages_batch) if messages_batch else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_call_with_retries, client, m, args, logger, response_schema): idx
            for idx, m in enumerate(messages_batch)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                outputs[idx] = future.result()
            except Exception as e:
                outputs[idx] = e
    return outputs


# =====================================================
# Main
# =====================================================

def run():
    args = parse_args()
    logger = setup_logger(args.log_dir, args.output_path)

    payload = load_input(args.input_path, args.input_format)
    rows = payload.records
    question_keys = parse_question_keys(args.question_keys)

    schema_cols = fetch_schema_columns_from_json(args.schema_file)

    schema_hints = {}
    if args.use_schema_hints and args.db_path and os.path.exists(args.db_path):
        conn = sqlite3.connect(args.db_path)
        db_cols = fetch_schema_columns_from_db(conn, args.table_name)
        schema_hints = fetch_schema_value_hints(conn, args.table_name, db_cols)
        conn.close()

    schema_hints_text = render_schema_hints(schema_cols, schema_hints)

    if args.prompt_style == "cot":
        zs_template = load_text(args.cot_prompt_file)
    else:
        zs_template = load_text(args.zs_prompt_file)

    few_shot_text = load_text(args.few_shot_file) if args.prompt_style == "few_shot" else ""

    if args.use_pydantic_schema:
        if args.prompt_style == "cot":
            response_schema = build_response_schema_from_model(COTResponse)
        else:
            response_schema = build_response_schema_from_model(SQLOnlyResponse)
    else:
        response_schema = None

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    ckpt = load_checkpoint(args.checkpoint_path) if bool(args.resume and args.checkpoint_path) else {"completed": {}, "meta": {}}
    completed: Dict[str, Dict[str, Any]] = ckpt.get("completed", {}) if isinstance(ckpt, dict) else {}
    processed_since_save = 0
    conf_values: List[float] = []

    pending = []
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
            cot_suffix="",
        )

        messages = [{"role": "user", "content": prompt}]
        pending.append((item_id, row, question, messages))

    batches = chunked_indices(len(pending), args.batch_size)

    success_count = 0
    error_count = 0

    for batch_index, (lo, hi) in enumerate(batches):
        batch = pending[lo:hi]
        messages_batch = [x[3] for x in batch]
        batch_start = time.time()
        outputs = run_batch(client, messages_batch, args, logger, response_schema)

        for idx_in_batch, (item_id, row_obj, question, _) in enumerate(batch):
            raw_output = ""
            pred_sql = ""
            thinking = ""
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
                    if args.prompt_style == "cot":
                        thinking = (parsed_json.get("thinking") or "").strip()
                        pred_sql = (parsed_json.get("sql") or "").strip()
                    else:
                        pred_sql = (parsed_json.get("sql") or "").strip()
                    model_meta["response_schema_used"] = True
                else:
                    pred_sql = extract_sql(raw_output)

                if args.logprob_mode == "structured":
                    try:
                        field_logprobs, field_confidence, conf_overall = structured_logprob_payload(out_obj, add_logprobs)
                    except Exception as lp_err:
                        model_meta["structured_logprob_error"] = str(lp_err).splitlines()[0] if str(lp_err) else str(lp_err)
                        field_logprobs, field_confidence, conf_overall = {}, {}, None

                if isinstance(conf_overall, (int, float)):
                    conf_values.append(float(conf_overall))

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
            processed_since_save += 1

            if error:
                error_count += 1
                append_jsonl(args.error_path or (args.output_path + ".errors.jsonl"), result)
            else:
                success_count += 1

            if processed_since_save >= max(1, int(args.save_every)):
                persist_outputs(
                    output_path=args.output_path,
                    output_format=payload.format_used,
                    input_rows=rows,
                    completed=completed,
                    id_key=args.id_key,
                    checkpoint_path=args.checkpoint_path,
                    checkpoint_meta=build_checkpoint_meta(args, len(rows), payload.format_used),
                )
                logger.info("Checkpoint saved at batch %d (processed %d since last save).", batch_index + 1, processed_since_save)
                processed_since_save = 0

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

    # Final persist
    persist_outputs(
        output_path=args.output_path,
        output_format=payload.format_used,
        input_rows=rows,
        completed=completed,
        id_key=args.id_key,
        checkpoint_path=args.checkpoint_path,
        checkpoint_meta=build_checkpoint_meta(args, len(rows), payload.format_used),
    )

    total_elapsed = max(1e-9, time.time() - sum([0.0, 0.0]))  # trivial
    avg_conf = mean(conf_values) if conf_values else None

    logger.info("=========== RUN SUMMARY ===========")
    logger.info("Rows selected: %d", len(rows))
    logger.info("Rows completed: %d", len(completed))
    logger.info("Processed (success/error): %d/%d", success_count, error_count)
    logger.info("Avg confidence_overall: %s", f"{avg_conf:.6f}" if avg_conf is not None else "NA")
    logger.info("Total runtime: %.2fs", total_elapsed)
    logger.info("===================================")


if __name__ == "__main__":
    run()