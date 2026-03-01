#!/usr/bin/env python3
"""
Run one-question NL2SQL inference with MPX0222forHF/SQL-R1-7B.

Supports two backends:
- vllm (recommended for speed/throughput)
- transformers (fallback)

Examples:
  python run_sqlr1_single.py \
    --model_path /mnt/shared/shared_hf_home/hub/models--MPX0222forHF--SQL-R1-7B/snapshots/db409e8372ca5e463126b07e905b5245caf14ea6 \
    --backend vllm \
    --input_json data/natural_question_1500.json \
    --row_index 0 \
    --gpu 0

  python run_sqlr1_single.py \
    --model_path /mnt/shared/shared_hf_home/hub/models--MPX0222forHF--SQL-R1-7B/snapshots/db409e8372ca5e463126b07e905b5245caf14ea6 \
    --question "Which Colorectal trials had 3 or more arms and used a multikinase inhibitor as a control?" \
    --backend transformers
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    openai_completion_to_meta,
    openai_token_logprob_payload,
    parse_openai_text,
    structured_logprob_payload,
)

try:
    from structured_logprobs import add_logprobs
except Exception:
    add_logprobs = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one SQL-R1-7B inference for NL2SQL.")

    ap.add_argument(
        "--model_path",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--MPX0222forHF--SQL-R1-7B/"
            "snapshots/db409e8372ca5e463126b07e905b5245caf14ea6"
        ),
    )
    ap.add_argument("--backend", choices=["vllm", "transformers", "openai_compat"], default="vllm")

    # Input question options
    ap.add_argument("--question", default="", help="Direct natural-language question. If set, input_json/row_index is ignored.")
    ap.add_argument("--input_json", default="data/natural_question_1500.json")
    ap.add_argument("--row_index", type=int, default=0, help="0-based index into input_json list")
    ap.add_argument("--question_keys", default="natural_question,question,original_question,new_question")

    # Prompt/schema
    ap.add_argument("--schema_json", default="data/schema.json")
    ap.add_argument("--table_name", default="clinical_trials")

    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Runtime
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--trust_remote_code", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)

    # OpenAI-compatible runtime (for structured logprob + pydantic schema)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="dummy")
    ap.add_argument("--model_name", default="sql-r1-7b-local")
    ap.add_argument("--use_pydantic_schema", type=int, default=1)
    ap.add_argument("--logprob_mode", choices=["structured", "none"], default="structured")

    # Optional output artifact
    ap.add_argument("--output_json", default="", help="Optional output path for saving result JSON")

    return ap.parse_args()


def parse_comma_keys(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def pick_first_nonempty(row: Dict[str, Any], keys: List[str]) -> Tuple[str, Optional[str]]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            return sv, k
    return "", None


def load_schema_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("schema_json must be a JSON list of column names")
    cols = [str(x).strip() for x in obj if str(x).strip()]
    if not cols:
        raise ValueError("No columns found in schema_json")
    return cols


def resolve_question(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    if (args.question or "").strip():
        return args.question.strip(), {"source": "cli", "row_index": None, "question_key": "question"}

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("input_json must contain a JSON list")
    if args.row_index < 0 or args.row_index >= len(rows):
        raise IndexError(f"row_index out of range: {args.row_index} (rows={len(rows)})")

    row = rows[args.row_index]
    if not isinstance(row, dict):
        raise ValueError(f"Row at index {args.row_index} is not an object")

    q, qk = pick_first_nonempty(row, parse_comma_keys(args.question_keys))
    if not q:
        raise ValueError(f"No non-empty question found using keys: {args.question_keys}")

    meta = {
        "source": args.input_json,
        "row_index": args.row_index,
        "question_key": qk,
        "item_id": row.get("item_id"),
    }
    return q, meta


def build_prompt(table_name: str, schema_cols: List[str], question: str) -> List[Dict[str, str]]:
    col_list = ", ".join([f'"{c}"' for c in schema_cols])

    system = (
        "You are an expert NL2SQL assistant. "
        "Generate exactly one valid SQL query that answers the user question. "
        "Return only SQL, no explanation."
    )

    user = f"""Table name:
{table_name}

Available columns:
{col_list}

Rules:
- Use only table \"{table_name}\".
- Use only listed columns.
- Keep all filters and numeric thresholds from the question.
- Output one SQL query only.

Question:
{question}
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_sql(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # Remove markdown fences if present.
    t = re.sub(r"^```[a-zA-Z0-9_-]*\\s*", "", t).strip()
    t = re.sub(r"\\s*```$", "", t).strip()

    # Prefer first SELECT/WITH statement block.
    m = re.search(r"\\b(SELECT|WITH)\\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start() :].strip()

    # Keep only first statement.
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"

    return t


def build_response_schema_from_model(model_cls):
    from openai.types import ResponseFormatJSONSchema

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


def generate_with_vllm(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vllm backend requires transformers + vllm in this environment"
        ) from exc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype=args.dtype,
        seed=int(args.seed),
    )
    sampling = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    out = llm.generate([prompt], sampling)
    text = (out[0].outputs[0].text or "").strip() if out and out[0].outputs else ""
    return text, extract_sql(text)


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name)


def generate_with_transformers(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers backend requires torch + transformers in this environment"
        ) from exc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=_torch_dtype(args.dtype) if args.dtype != "auto" else None,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
    }
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out = model.generate(inputs, **gen_kwargs)

    new_tokens = out[0, inputs.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, extract_sql(text)


def generate_with_openai_compat(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        from openai import OpenAI
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError(
            "openai_compat backend requires openai + pydantic in this environment"
        ) from exc

    class SQLOnlyResponse(BaseModel):
        sql: str

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    req: Dict[str, Any] = {
        "model": args.model_name,
        "messages": messages,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_new_tokens),
        "timeout": float(args.timeout),
        "logprobs": args.logprob_mode == "structured",
    }

    if bool(int(args.use_pydantic_schema)):
        req["response_format"] = build_response_schema_from_model(SQLOnlyResponse)

    completion = client.chat.completions.create(**req)
    raw_text = parse_openai_text(completion)
    model_meta: Dict[str, Any] = openai_completion_to_meta(completion)

    pred_sql = ""
    if bool(int(args.use_pydantic_schema)):
        parsed_json = json.loads(raw_text)
        pred_sql = str(parsed_json.get("sql") or "").strip()
        model_meta["response_schema_used"] = True
    else:
        pred_sql = extract_sql(raw_text)

    field_logprobs: Dict[str, Any] = {}
    field_confidence: Dict[str, Any] = {}
    conf_overall: Optional[float] = None
    if args.logprob_mode == "structured":
        try:
            field_logprobs, field_confidence, conf_overall = structured_logprob_payload(completion, add_logprobs)
            if conf_overall is None:
                fb_lp, fb_conf, fb_overall = openai_token_logprob_payload(completion)
                if isinstance(fb_overall, (int, float)):
                    field_logprobs, field_confidence, conf_overall = fb_lp, fb_conf, fb_overall
                    model_meta["logprob_fallback"] = "openai_token_logprobs"
        except Exception as lp_err:
            model_meta["structured_logprob_error"] = str(lp_err).splitlines()[0] if str(lp_err) else str(lp_err)
            fb_lp, fb_conf, fb_overall = openai_token_logprob_payload(completion)
            if isinstance(fb_overall, (int, float)):
                field_logprobs, field_confidence, conf_overall = fb_lp, fb_conf, fb_overall
                model_meta["logprob_fallback"] = "openai_token_logprobs"

    return {
        "raw_text": raw_text,
        "pred_sql": pred_sql,
        "field_logprobs": field_logprobs,
        "field_confidence": field_confidence,
        "confidence_overall": conf_overall,
        "model_meta": model_meta,
    }


def main() -> None:
    args = parse_args()

    question, q_meta = resolve_question(args)
    schema_cols = load_schema_columns(args.schema_json)
    messages = build_prompt(args.table_name, schema_cols, question)

    field_logprobs: Dict[str, Any] = {}
    field_confidence: Dict[str, Any] = {}
    confidence_overall: Optional[float] = None
    model_meta: Dict[str, Any] = {}

    if args.backend == "vllm":
        raw_text, pred_sql = generate_with_vllm(args, messages)
    elif args.backend == "transformers":
        raw_text, pred_sql = generate_with_transformers(args, messages)
    else:
        out = generate_with_openai_compat(args, messages)
        raw_text = out["raw_text"]
        pred_sql = out["pred_sql"]
        field_logprobs = out["field_logprobs"]
        field_confidence = out["field_confidence"]
        confidence_overall = out["confidence_overall"]
        model_meta = out["model_meta"]

    result = {
        "model_path": args.model_path,
        "backend": args.backend,
        "runtime": {
            "gpu": args.gpu,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
        },
        "input": {
            "question": question,
            **q_meta,
        },
        "pred_sql": pred_sql,
        "raw_text": raw_text,
        "logprob_mode": args.logprob_mode if args.backend == "openai_compat" else "none",
        "field_logprobs": field_logprobs,
        "field_confidence": field_confidence,
        "confidence_overall": confidence_overall,
        "model_meta": model_meta,
    }

    print("\n=== INPUT QUESTION ===")
    print(question)
    print("\n=== PREDICTED SQL ===")
    print(pred_sql or "<empty>")
    if args.backend == "openai_compat":
        print("\n=== CONFIDENCE OVERALL ===")
        print(confidence_overall if confidence_overall is not None else "NA")

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
