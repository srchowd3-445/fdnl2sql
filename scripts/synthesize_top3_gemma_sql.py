#!/usr/bin/env python3
"""Retrieve top-k similar queries, synthesize SQL with Gemma, execute preview, and save.

Modes:
- Single mode (default): one question via --question or --question-json + --question-index.
- Batch mode: --batch-mode 1 with --question-json + --start-index + --limit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import retrieve_similar_queries as rsq


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(
        description="Use top-k retrieved candidate SQLs as context, synthesize final SQL with Gemma, then execute preview."
    )

    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    source.add_argument("--candidate-json")
    source.add_argument("--candidate-sqlite")

    ap.add_argument("--candidate-table", default="query_library")
    ap.add_argument("--candidate-question-col", default="question")
    ap.add_argument("--candidate-sql-col", default="sql")
    ap.add_argument("--candidate-id-col", default="id")

    qsrc = ap.add_mutually_exclusive_group(required=True)
    qsrc.add_argument("--question", help="Ad-hoc natural-language question")
    qsrc.add_argument("--question-json", help="JSON list containing questions")

    ap.add_argument("--question-index", type=int, default=0, help="Single-mode index when using --question-json")
    ap.add_argument("--question-key", default="natural_question")

    ap.add_argument("--batch-mode", type=int, default=0, help="Set 1 to run a range from --question-json")
    ap.add_argument("--start-index", type=int, default=0, help="Batch start index (0-based)")
    ap.add_argument("--limit", type=int, default=1, help="Batch size; use -1 for all remaining")

    ap.add_argument("--top-k", type=int, default=3)

    ap.add_argument(
        "--model-path",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/"
            "snapshots/005ad3404e59d6023443cb575daa05336842228a"
        ),
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)
    ap.add_argument("--gen-batch-size", type=int, default=8, help="Prompt batch size per vLLM generate() call")

    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=0, help="1 to skip SQL execution preview")

    ap.add_argument(
        "--output-json",
        default=str(root / "results" / "retrieval_top3_gemma_synth.json"),
    )

    return ap.parse_args()


def load_candidates(args: argparse.Namespace) -> List[rsq.Candidate]:
    if args.candidate_sqlite:
        return rsq.load_candidates_from_sqlite(
            db_path=Path(args.candidate_sqlite),
            table=args.candidate_table,
            question_col=args.candidate_question_col,
            sql_col=args.candidate_sql_col,
            id_col=args.candidate_id_col or None,
        )

    src = Path(args.candidate_json) if args.candidate_json else Path(args.seed_json)
    return rsq.load_candidates_from_seed_json(src)


def extract_question_from_row(row: Dict[str, Any], key: str) -> Tuple[str, Optional[str]]:
    q = row.get(key)
    if isinstance(q, str) and q.strip():
        return q.strip(), None

    for alt in ("natural_question", "question", "original_question"):
        alt_q = row.get(alt)
        if isinstance(alt_q, str) and alt_q.strip():
            return alt_q.strip(), None

    return "", f"MISSING_QUESTION_KEY:{key}"


def load_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.question:
        return [
            {
                "task_index": 0,
                "question_index": None,
                "item_id": None,
                "question": args.question.strip(),
                "error": None,
            }
        ]

    # question-json path
    obj = json.loads(Path(args.question_json).read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list at {args.question_json}")

    tasks: List[Dict[str, Any]] = []

    if int(args.batch_mode) == 1:
        start = max(0, int(args.start_index))
        end = len(obj) if int(args.limit) == -1 else min(len(obj), start + max(0, int(args.limit)))
        indices = list(range(start, end))
    else:
        idx = int(args.question_index)
        if idx < 0 or idx >= len(obj):
            raise IndexError(f"question-index {idx} out of range for {len(obj)} rows")
        indices = [idx]

    for t_idx, q_idx in enumerate(indices):
        row = obj[q_idx]
        if not isinstance(row, dict):
            tasks.append(
                {
                    "task_index": t_idx,
                    "question_index": q_idx,
                    "item_id": None,
                    "question": "",
                    "error": "ROW_NOT_OBJECT",
                }
            )
            continue

        q_text, q_err = extract_question_from_row(row, args.question_key)
        tasks.append(
            {
                "task_index": t_idx,
                "question_index": q_idx,
                "item_id": row.get("item_id"),
                "question": q_text,
                "error": q_err,
            }
        )

    return tasks


def build_messages(question: str, ranked: Sequence[rsq.MatchResult]) -> List[Dict[str, str]]:
    candidate_blocks: List[str] = []
    for m in ranked:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate Rank {m.rank}",
                    f"Score: {m.total_score:.4f}",
                    f"Candidate Question: {m.candidate.question}",
                    "Candidate SQL:",
                    m.candidate.sql,
                ]
            )
        )

    system = (
        "You are an expert SQLite NL2SQL assistant. "
        "Given a user question and retrieved candidate SQL queries, synthesize one best final SQL query. "
        "Return only SQL, no explanation."
    )

    user = (
        "User Question:\n"
        f"{question}\n\n"
        "Top Retrieved Candidates:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        "Rules:\n"
        "- Output exactly one SQL statement ending with a semicolon.\n"
        "- Prefer preserving explicit constraints from the user question.\n"
        "- Prefer syntactically valid SQLite SQL.\n"
        "- Do not include markdown fences or any commentary.\n"
    )

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


def init_generator(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("This script requires transformers + vllm in the active Python env") from exc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    return tokenizer, llm, sampling


def generate_sql_batch(
    tokenizer: Any,
    llm: Any,
    sampling: Any,
    messages_list: Sequence[List[Dict[str, str]]],
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    n = len(messages_list)
    raw_texts = [""] * n
    sqls = [""] * n

    bs = max(1, int(batch_size))
    for b0 in range(0, n, bs):
        b1 = min(n, b0 + bs)
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list[b0:b1]
        ]

        outs = llm.generate(prompts, sampling)
        for j, out in enumerate(outs):
            text = (out.outputs[0].text or "").strip() if out.outputs else ""
            idx = b0 + j
            raw_texts[idx] = text
            sqls[idx] = extract_sql(text)

        print(f"Generated {b1}/{n}")

    return raw_texts, sqls


def run_execution_preview(args: argparse.Namespace, final_sql: str) -> Dict[str, Any]:
    if int(args.skip_exec) == 1:
        return {
            "columns": [],
            "rows": [],
            "error": None,
        }

    if not final_sql:
        return {
            "columns": [],
            "rows": [],
            "error": "EMPTY_SQL_FROM_MODEL",
        }

    cols, rows, err = rsq.execute_sql_preview(Path(args.db_path), final_sql, max_rows=int(args.preview_rows))
    return {
        "columns": cols,
        "rows": rows,
        "error": err,
    }


def result_from_error(task: Dict[str, Any], err: str) -> Dict[str, Any]:
    return {
        "task_index": task.get("task_index"),
        "question_index": task.get("question_index"),
        "item_id": task.get("item_id"),
        "input_question": task.get("question", ""),
        "error": err,
        "ranked_results": [],
        "synthesized": {
            "raw_text": "",
            "final_sql": "",
        },
        "execution_preview": {
            "columns": [],
            "rows": [],
            "error": err,
        },
    }


def main() -> None:
    args = parse_args()

    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No question tasks were loaded.")

    candidates = load_candidates(args)
    if not candidates:
        raise SystemExit("No candidates loaded. Check candidate source path/format.")

    prepared: List[Dict[str, Any]] = []
    results_by_task_index: Dict[int, Dict[str, Any]] = {}

    for task in tasks:
        task_idx = int(task["task_index"])
        if task.get("error"):
            results_by_task_index[task_idx] = result_from_error(task, str(task["error"]))
            continue

        question = str(task.get("question") or "").strip()
        if not question:
            results_by_task_index[task_idx] = result_from_error(task, "EMPTY_QUESTION")
            continue

        ranked = rsq.rank_candidates(question, candidates, top_k=max(1, int(args.top_k)))
        messages = build_messages(question, ranked)
        prepared.append(
            {
                "task": task,
                "ranked": ranked,
                "messages": messages,
            }
        )

    if prepared:
        tokenizer, llm, sampling = init_generator(args)
        raw_texts, final_sqls = generate_sql_batch(
            tokenizer,
            llm,
            sampling,
            [x["messages"] for x in prepared],
            batch_size=int(args.gen_batch_size),
        )

        for packed, raw_text, final_sql in zip(prepared, raw_texts, final_sqls):
            task = packed["task"]
            ranked = packed["ranked"]
            task_idx = int(task["task_index"])

            execution = run_execution_preview(args, final_sql)

            results_by_task_index[task_idx] = {
                "task_index": task_idx,
                "question_index": task.get("question_index"),
                "item_id": task.get("item_id"),
                "input_question": task.get("question"),
                "error": None,
                "ranked_results": [
                    {
                        **asdict(m),
                        "candidate": asdict(m.candidate),
                    }
                    for m in ranked
                ],
                "synthesized": {
                    "raw_text": raw_text,
                    "final_sql": final_sql,
                },
                "execution_preview": execution,
            }

    results = [results_by_task_index[int(t["task_index"])] for t in tasks]

    counts = {
        "total_tasks": len(results),
        "error_tasks": sum(1 for r in results if r.get("error")),
        "generated_nonempty_sql": sum(1 for r in results if (r.get("synthesized", {}).get("final_sql") or "").strip()),
        "exec_ok": sum(1 for r in results if not (r.get("execution_preview", {}).get("error"))),
    }

    meta: Dict[str, Any] = {
        "model_path": args.model_path,
        "top_k": int(args.top_k),
        "db_path": args.db_path,
        "question_source": "question" if args.question else "question_json",
        "question_index": args.question_index if (args.question_json and int(args.batch_mode) != 1) else None,
        "question_key": args.question_key if args.question_json else None,
        "batch_mode": bool(int(args.batch_mode)),
        "start_index": int(args.start_index) if int(args.batch_mode) == 1 else None,
        "limit": int(args.limit) if int(args.batch_mode) == 1 else None,
        "seed_json": args.seed_json,
        "candidate_json": args.candidate_json,
        "candidate_sqlite": args.candidate_sqlite,
        "gen_batch_size": int(args.gen_batch_size),
    }

    # Backward-compatible single-output shape when not using batch mode.
    if len(results) == 1 and int(args.batch_mode) != 1:
        one = results[0]
        payload: Dict[str, Any] = {
            "meta": meta,
            "input_question": one.get("input_question", ""),
            "ranked_results": one.get("ranked_results", []),
            "synthesized": one.get("synthesized", {"raw_text": "", "final_sql": ""}),
            "execution_preview": one.get("execution_preview", {"columns": [], "rows": [], "error": None}),
            "question_index": one.get("question_index"),
            "item_id": one.get("item_id"),
            "error": one.get("error"),
            "counts": counts,
        }
    else:
        payload = {
            "meta": meta,
            "counts": counts,
            "results": results,
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Tasks:", counts["total_tasks"])
    print("Errors:", counts["error_tasks"])
    print("Generated SQL:", counts["generated_nonempty_sql"])
    if int(args.skip_exec) != 1:
        print("Exec ok:", counts["exec_ok"])


if __name__ == "__main__":
    main()
