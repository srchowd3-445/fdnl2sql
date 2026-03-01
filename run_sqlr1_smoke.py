#!/usr/bin/env python3
"""
Run SQL-R1-7B on multiple questions (smoke/default first 10) in one process.

This script uses vLLM and loads the model once for efficiency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SQL-R1-7B on a question slice and save predictions.")

    ap.add_argument(
        "--model_path",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--MPX0222forHF--SQL-R1-7B/"
            "snapshots/db409e8372ca5e463126b07e905b5245caf14ea6"
        ),
    )
    ap.add_argument("--input_json", default="data/natural_question_1500.json")
    ap.add_argument("--schema_json", default="data/schema.json")
    ap.add_argument("--output_json", default="results/nl-sql-r1-smoke.json")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10, help="Use -1 for all remaining rows")
    ap.add_argument("--question_keys", default="natural_question,question,original_question,new_question")
    ap.add_argument("--table_name", default="clinical_trials")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--trust_remote_code", type=int, default=1)

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


def build_messages(table_name: str, schema_cols: List[str], question: str) -> List[Dict[str, str]]:
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
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\b(SELECT|WITH)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start() :].strip()

    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"

    return t


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("This script requires transformers + vllm in the active Python env") from exc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("input_json must be a JSON list")

    start = max(0, int(args.start))
    picked = rows[start:]
    if int(args.limit) > -1:
        picked = picked[: int(args.limit)]

    schema_cols = load_schema_columns(args.schema_json)
    q_keys = parse_comma_keys(args.question_keys)

    # Build prompt entries.
    entries: List[Dict[str, Any]] = []
    for off, row in enumerate(picked):
        gidx = start + off
        if not isinstance(row, dict):
            entries.append(
                {
                    "row_index": gidx,
                    "item_id": None,
                    "original_question": "",
                    "gt_sql": "",
                    "pred_sql": "",
                    "raw_text": "",
                    "status": "error",
                    "error": "ROW_NOT_OBJECT",
                }
            )
            continue

        q, qk = pick_first_nonempty(row, q_keys)
        if not q:
            entries.append(
                {
                    "row_index": gidx,
                    "item_id": row.get("item_id"),
                    "original_question": "",
                    "gt_sql": str(row.get("gt_sql") or ""),
                    "pred_sql": "",
                    "raw_text": "",
                    "status": "error",
                    "error": f"MISSING_QUESTION_KEYS:{args.question_keys}",
                }
            )
            continue

        entries.append(
            {
                "row_index": gidx,
                "item_id": row.get("item_id"),
                "question_key": qk,
                "original_question": q,
                "gt_sql": str(row.get("gt_sql") or ""),
                "messages": build_messages(args.table_name, schema_cols, q),
                "pred_sql": "",
                "raw_text": "",
                "status": "pending",
                "error": None,
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
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

    pending_idx = [i for i, e in enumerate(entries) if e.get("status") == "pending"]
    bs = max(1, int(args.batch_size))

    for b0 in range(0, len(pending_idx), bs):
        batch_ids = pending_idx[b0 : b0 + bs]
        prompts: List[str] = []
        for i in batch_ids:
            prompt = tokenizer.apply_chat_template(entries[i]["messages"], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        outs = llm.generate(prompts, sampling)
        for i, out in zip(batch_ids, outs):
            text = (out.outputs[0].text or "").strip() if out.outputs else ""
            entries[i]["raw_text"] = text
            entries[i]["pred_sql"] = extract_sql(text)
            entries[i]["status"] = "ok" if entries[i]["pred_sql"] else "empty"
            entries[i]["error"] = None if entries[i]["pred_sql"] else "EMPTY_SQL"
            entries[i].pop("messages", None)

        done = min(b0 + len(batch_ids), len(pending_idx))
        print(f"Processed {done}/{len(pending_idx)}")

    # Remove prompt payload from non-generated error rows if any.
    for e in entries:
        e.pop("messages", None)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    ok = sum(1 for e in entries if e.get("status") == "ok")
    empty = sum(1 for e in entries if e.get("status") == "empty")
    err = sum(1 for e in entries if e.get("status") == "error")
    print("\n=== SQL-R1 SMOKE RUN ===")
    print(f"Input slice rows: {len(entries)}")
    print(f"OK:               {ok}")
    print(f"EMPTY:            {empty}")
    print(f"ERROR:            {err}")
    print(f"Output JSON:      {args.output_json}")


if __name__ == "__main__":
    main()
