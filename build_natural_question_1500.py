#!/usr/bin/env python3
"""
Generate natural questions from SQL + original question using Gemma 3 (vLLM).

Default flow:
- Input JSON list:  data/empty_gt_replaced_final.json
- Output JSON list: data/natural_question_1500.json

Each output row keeps the original fields and adds:
- natural_question
- original_question_used_for_naturalization
- sql_used_for_naturalization
- natural_question_model_meta
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


def pick_first_nonempty(row: Dict[str, Any], keys: List[str]) -> Tuple[str, Optional[str]]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s, k
    return "", None


def build_prompt(original_question: str, sql_text: str, bad_output_hint: str = "") -> str:
    hint_block = ""
    if bad_output_hint:
        hint_block = f"""
Previous invalid output (do NOT repeat):
{bad_output_hint}
"""

    return f"""Rewrite the question to sound natural while keeping intent exactly the same as the SQL.

Rules:
- Output ONLY one natural-language question on one line.
- Do not mention SQL, table names, databases, or column names.
- Preserve every filter/constraint implied by the SQL (entities, values, thresholds, years, phase, etc.).
- Do not add new constraints and do not remove any constraints.
- Keep clinical-trial wording natural and concise.
- Do NOT use markdown, code fences, labels, or backticks.
- Keep explicit numbers exactly as digits (example: 3, 2018).
- Prefer explicit disease and treatment names rather than abbreviations.

Original question:
{original_question}

Ground-truth SQL:
{sql_text}
{hint_block}
"""


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    return t


def strip_known_prefix(line: str) -> str:
    s = (line or "").strip()
    s = re.sub(
        r"^(?:revised question|correct output|correct answer|natural language question|rewritten question|question|output|answer)\s*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def looks_like_junk_line(line: str) -> bool:
    lo = (line or "").strip().lower()
    if not lo:
        return True
    if lo in {"```", "text", "sql", "json"}:
        return True
    if lo.endswith(":") and len(lo.split()) <= 4:
        return True
    if "failed with" in lo or "raw output" in lo:
        return True
    return False


def extract_question_candidate(text: str) -> str:
    if not text:
        return ""
    t = strip_code_fences(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # First pass: line-level extraction (skip labels and helper text)
    for ln in lines:
        ln2 = strip_known_prefix(ln)
        if looks_like_junk_line(ln2):
            continue
        if len(re.findall(r"[A-Za-z0-9]+", ln2)) >= 4:
            return ln2

    # Second pass: find first question-like sentence in full text
    t2 = strip_known_prefix(t)
    m = re.search(r"([A-Z][^?\n]{10,}\?)", t2)
    if m:
        cand = m.group(1).strip()
        if not looks_like_junk_line(cand):
            return cand

    # Last fallback: first non-empty non-junk line
    for ln in lines:
        ln2 = strip_known_prefix(ln)
        if ln2 and not looks_like_junk_line(ln2):
            return ln2
    return ""


def sanitize_question(q: str) -> str:
    s = (q or "").strip()
    s = s.replace("`", "").strip()
    s = strip_known_prefix(s)
    s = re.sub(r"^(question\s*:)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.strip('"').strip("'").strip()
    if s in {"", "-", "---"}:
        return ""
    if not s.endswith("?"):
        s = s.rstrip(".").strip() + "?"
    return s


def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def extract_sql_literals(sql_text: str) -> List[str]:
    vals = re.findall(r"'([^']+)'", sql_text or "")
    out: List[str] = []
    for v in vals:
        vv = v.strip()
        if vv:
            out.append(vv)
    return out


def extract_sql_numbers(sql_text: str) -> List[str]:
    return re.findall(r'"\s*[^"]+\s*"\s*(?:>=|>|<=|<|=)\s*(-?\d+(?:\.\d+)?)', sql_text or "")


def is_weak_literal(lit: str) -> bool:
    weak = {
        "yes",
        "no",
        "combination",
        "monotherapy",
        "original publication",
        "follow-up",
        "original",
    }
    return (lit or "").strip().lower() in weak


def literal_coverage(question: str, literals: List[str]) -> float:
    strong_literals = [x for x in literals if not is_weak_literal(x)]
    if not strong_literals:
        return 1.0
    q_tokens = set(tokenize_simple(question))
    covered = 0
    total = 0
    for lit in strong_literals:
        ltoks = [t for t in tokenize_simple(lit) if t]
        if not ltoks:
            continue
        total += 1
        if all(t in q_tokens for t in ltoks):
            covered += 1
    if total == 0:
        return 1.0
    return covered / total


def has_required_numbers(question: str, required_numbers: List[str]) -> bool:
    if not required_numbers:
        return True
    q = question or ""
    return all(n in q for n in required_numbers)


def validate_natural_question(
    q: str,
    sql_text: str,
    min_question_tokens: int,
    min_literal_coverage: float,
) -> Tuple[bool, Optional[str], float]:
    if not q:
        return False, "EMPTY", 0.0

    q0 = q.strip()
    if "`" in q0:
        return False, "HAS_BACKTICKS", 0.0
    if q0.lower().startswith(("select ", "sql", "```", "text")):
        return False, "SQLISH_OR_MARKDOWN", 0.0

    tok_count = len(tokenize_simple(q0))
    if tok_count < max(1, int(min_question_tokens)):
        return False, f"TOO_SHORT_{tok_count}", 0.0

    lits = extract_sql_literals(sql_text)
    cov = literal_coverage(q0, lits)
    if cov < float(min_literal_coverage):
        return False, f"LITERAL_COVERAGE_{cov:.2f}", cov

    nums = extract_sql_numbers(sql_text)
    if not has_required_numbers(q0, nums):
        return False, "MISSING_SQL_NUMBER", cov

    return True, None, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default="data/empty_gt_replaced_final.json")
    ap.add_argument("--output_json", default="data/natural_question_1500.json")
    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--retry_attempts", type=int, default=4, help="Retry invalid/questionable generations")
    ap.add_argument("--min_question_tokens", type=int, default=4)
    ap.add_argument("--min_literal_coverage", type=float, default=0.0)
    args = ap.parse_args()

    if LLM is None or SamplingParams is None:
        raise RuntimeError("vllm is not installed/importable. Install vllm first.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("input_json must contain a JSON list.")

    rows = rows[args.start:]
    if args.limit is not None and args.limit > -1:
        rows = rows[: args.limit]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_rows: List[Dict[str, Any]] = []
    missing_sql = 0
    generated = 0

    for b0 in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[b0 : b0 + max(1, args.batch_size)]
        batch_prompts: List[str] = []
        batch_indices: List[int] = []
        batch_ctx: List[Tuple[str, str, Optional[str], Optional[str]]] = []

        # Prepare placeholders for batch outputs
        batch_out: List[Dict[str, Any]] = [dict(r) for r in batch]

        for i, r in enumerate(batch):
            sql_text, sql_key = pick_first_nonempty(r, ["new_sql", "gt_sql", "new_gt_sql", "sql"])
            q_text, q_key = pick_first_nonempty(r, ["original_question", "question", "generated_question", "new_question"])

            if not sql_text:
                missing_sql += 1
                fallback_q = q_text if q_text else "Which clinical trials match the criteria?"
                batch_out[i]["natural_question"] = fallback_q if fallback_q.endswith("?") else (fallback_q.rstrip(".") + "?")
                batch_out[i]["original_question_used_for_naturalization"] = q_text
                batch_out[i]["sql_used_for_naturalization"] = ""
                batch_out[i]["natural_question_model_meta"] = {
                    "error": "MISSING_SQL",
                    "question_key_used": q_key,
                    "sql_key_used": sql_key,
                }
                continue

            prompt = build_prompt(q_text or "Which clinical trials match the criteria?", sql_text)
            batch_prompts.append(prompt)
            batch_indices.append(i)
            batch_ctx.append((q_text, sql_text, q_key, sql_key))

        # Generate only for rows that have SQL
        if batch_prompts:
            pending: List[Dict[str, Any]] = []
            for j in range(len(batch_prompts)):
                pending.append({
                    "batch_i": batch_indices[j],
                    "q_text": batch_ctx[j][0],
                    "sql_text": batch_ctx[j][1],
                    "q_key": batch_ctx[j][2],
                    "sql_key": batch_ctx[j][3],
                    "bad_hint": "",
                    "last_raw": "",
                    "last_reason": None,
                    "best_q": "",
                    "best_cov": -1.0,
                })

            retry_attempts = max(1, int(args.retry_attempts))
            for attempt in range(1, retry_attempts + 1):
                if not pending:
                    break

                prompts = [
                    build_prompt(
                        p["q_text"] or "Which clinical trials match the criteria?",
                        p["sql_text"],
                        bad_output_hint=p["bad_hint"],
                    )
                    for p in pending
                ]
                outs = llm.generate(prompts, sampling)

                next_pending: List[Dict[str, Any]] = []
                for p, out in zip(pending, outs):
                    i = p["batch_i"]
                    q_text = p["q_text"]
                    sql_text = p["sql_text"]
                    q_key = p["q_key"]
                    sql_key = p["sql_key"]

                    raw = (out.outputs[0].text or "").strip() if out.outputs else ""
                    cand = sanitize_question(extract_question_candidate(raw))
                    ok, reason, cov = validate_natural_question(
                        cand,
                        sql_text,
                        min_question_tokens=args.min_question_tokens,
                        min_literal_coverage=args.min_literal_coverage,
                    )

                    if cov > p["best_cov"] and cand:
                        p["best_cov"] = cov
                        p["best_q"] = cand

                    if ok:
                        meta = {
                            "finish_reason": getattr(out.outputs[0], "finish_reason", None) if out.outputs else None,
                            "stop_reason": getattr(out.outputs[0], "stop_reason", None) if out.outputs else None,
                            "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []) if out.outputs else None,
                            "question_key_used": q_key,
                            "sql_key_used": sql_key,
                            "attempt_used": attempt,
                            "retry_attempts_configured": retry_attempts,
                            "validation_error": None,
                            "literal_coverage": round(cov, 4),
                        }

                        batch_out[i]["natural_question"] = cand
                        batch_out[i]["original_question_used_for_naturalization"] = q_text
                        batch_out[i]["sql_used_for_naturalization"] = sql_text
                        batch_out[i]["natural_question_model_meta"] = meta
                        generated += 1
                    else:
                        p["last_raw"] = raw
                        p["last_reason"] = reason
                        p["bad_hint"] = f"Previous attempt invalid due to: {reason}. Return only one clean question."
                        next_pending.append(p)

                pending = next_pending

            # Final fallback for unresolved rows
            for p in pending:
                i = p["batch_i"]
                q_text = p["q_text"]
                sql_text = p["sql_text"]
                q_key = p["q_key"]
                sql_key = p["sql_key"]

                fallback_q = p["best_q"] or q_text or "Which clinical trials match the criteria?"
                fallback_q = sanitize_question(fallback_q)
                if not fallback_q:
                    fallback_q = "Which clinical trials match the criteria?"

                batch_out[i]["natural_question"] = fallback_q
                batch_out[i]["original_question_used_for_naturalization"] = q_text
                batch_out[i]["sql_used_for_naturalization"] = sql_text
                batch_out[i]["natural_question_model_meta"] = {
                    "error": "FAILED_VALIDATION_AFTER_RETRIES",
                    "validation_error": p["last_reason"],
                    "last_raw_output": p["last_raw"][:500],
                    "question_key_used": q_key,
                    "sql_key_used": sql_key,
                    "retry_attempts_configured": retry_attempts,
                    "literal_coverage_best": round(max(0.0, p["best_cov"]), 4),
                }
                generated += 1

        output_rows.extend(batch_out)

        done = min(b0 + len(batch), len(rows))
        if done % 100 == 0 or done == len(rows):
            print(f"Processed {done}/{len(rows)}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    print("\n================ NATURAL QUESTION BUILD ================")
    print(f"Input rows:                     {len(rows)}")
    print(f"Generated with model:           {generated}")
    print(f"Rows missing SQL (fallback):    {missing_sql}")
    print(f"Output JSON:                    {args.output_json}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
