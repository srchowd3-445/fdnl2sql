#!/usr/bin/env python3
"""Single-question orchestrator for chat-style NL2SQL.

This script runs the existing 2-stage method pipeline for exactly one question:
1) decompose_retrieve_top3_gemma_sql.py
2) synthesize_from_decompose_retrieval_gemma_sql.py

It is designed for chatbot integration: one input question in, one structured
response out (SQL + confidence + supporting traces).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Run one-question decompose->retrieve->synthesize pipeline for chat usage.")

    # Question (single only)
    ap.add_argument("--question", required=True, help="Single user question text.")
    ap.add_argument("--item-id", default="", help="Optional external id from chat frontend.")

    # Candidate source for retrieval
    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    source.add_argument("--candidate-json", default="")
    source.add_argument("--candidate-sqlite", default="")

    ap.add_argument("--candidate-table", default="query_library")
    ap.add_argument("--candidate-question-col", default="question")
    ap.add_argument("--candidate-sql-col", default="sql")
    ap.add_argument("--candidate-id-col", default="id")

    # Prompt/context
    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--decompose-prompt-file", default=str(root / "method" / "prompt" / "decompose_prompt.txt"))
    ap.add_argument(
        "--decompose-examples-file",
        default=str(root / "method" / "prompt" / "decompose_examples.txt"),
    )
    ap.add_argument(
        "--synthesis-prompt-file",
        default=str(root / "method" / "prompt" / "synthesis_prompt.txt"),
    )

    # Retrieval params
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--retrieval-per-decomp", type=int, default=3)
    ap.add_argument("--max-decomposed-queries", type=int, default=5)
    ap.add_argument(
        "--sbert-model",
        default="/mnt/shared/shared_hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    )
    ap.add_argument("--sbert-device", default="")
    ap.add_argument("--sbert-batch-size", type=int, default=64)

    # Backend/model
    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default="gemma-3-27b-local")
    ap.add_argument(
        "--model-path",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/"
            "snapshots/005ad3404e59d6023443cb575daa05336842228a"
        ),
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=2)

    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=8192)

    # Generation params
    ap.add_argument("--decompose-max-new-tokens", type=int, default=384)
    ap.add_argument("--synth-max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", type=int, default=1)

    ap.add_argument("--decompose-gen-batch-size", type=int, default=1)
    ap.add_argument("--decompose-batch-concurrency", type=int, default=1)
    ap.add_argument("--synth-gen-batch-size", type=int, default=1)
    ap.add_argument("--synth-batch-concurrency", type=int, default=1)
    ap.add_argument("--synth-evidence-per-decomp", type=int, default=2)
    ap.add_argument("--synth-include-global-merged", type=int, default=0)
    ap.add_argument("--synth-where-only-evidence", type=int, default=1)
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")
    ap.add_argument("--logprobs", type=int, default=0)

    # Exec/eval
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=1)
    ap.add_argument("--run-eval", type=int, default=0)
    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--eval-script", default=str(root / "eval_run_baselines_v2.py"))
    ap.add_argument("--compute-bertscore", type=int, default=0)
    ap.add_argument("--ast-weight-select", type=float, default=0.5)
    ap.add_argument("--ast-weight-where", type=float, default=0.4)
    ap.add_argument("--ast-weight-from", type=float, default=0.1)
    ap.add_argument(
        "--acceptance-config",
        default=str(root / "method" / "decompose_method" / "seed_acceptance_rules.config"),
    )

    # Output
    ap.add_argument("--output-dir", default=str(root / "results" / "chat_single"))
    ap.add_argument("--run-tag", default="", help="Optional run tag; default is timestamp.")
    ap.add_argument("--output-json", default="", help="Optional final response JSON path.")
    ap.add_argument("--keep-artifacts", type=int, default=1, help="1=keep stage artifacts under output-dir/run-tag.")
    ap.add_argument("--dry-run", type=int, default=0)

    return ap.parse_args()


def _append(cmd: List[str], flag: str, val: Optional[str]) -> None:
    if val is None:
        return
    s = str(val).strip()
    if s:
        cmd.extend([flag, s])


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def derive_paths(args: argparse.Namespace, root: Path) -> Dict[str, Path]:
    out_root = Path(args.output_dir)
    tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / tag

    return {
        "out_root": out_root,
        "run_dir": run_dir,
        "decompose_json": run_dir / "decompose_retrieve.json",
        "decompose_jsonl": run_dir / "decompose_retrieve.jsonl",
        "synth_json": run_dir / "synth.json",
        "synth_jsonl": run_dir / "synth.sql.jsonl",
        "eval_ready_json": run_dir / "eval_ready.json",
        "eval_ready_jsonl": run_dir / "eval_ready.jsonl",
        "eval_output_json": run_dir / "eval.json",
        "acceptance_output_json": run_dir / "acceptance.json",
        "response_json": (Path(args.output_json) if args.output_json.strip() else run_dir / "chat_response.json"),
        "decompose_script": root / "method" / "decompose_retrieve_top3_gemma_sql.py",
        "synth_script": root / "method" / "synthesize_from_decompose_retrieval_gemma_sql.py",
    }


def build_decompose_cmd(args: argparse.Namespace, p: Dict[str, Path]) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(p["decompose_script"]),
        "--question",
        str(args.question),
        "--schema-json",
        str(args.schema_json),
        "--prompt-file",
        str(args.decompose_prompt_file),
        "--decompose-examples-file",
        str(args.decompose_examples_file),
        "--top-k",
        str(int(args.top_k)),
        "--retrieval-per-decomp",
        str(int(args.retrieval_per_decomp)),
        "--max-decomposed-queries",
        str(int(args.max_decomposed_queries)),
        "--sbert-model",
        str(args.sbert_model),
        "--sbert-batch-size",
        str(int(args.sbert_batch_size)),
        "--backend",
        str(args.backend),
        "--api-base",
        str(args.api_base),
        f"--api-key={args.api_key}",
        "--model-name",
        str(args.model_name),
        "--timeout",
        str(float(args.timeout)),
        "--num-retries",
        str(int(args.num_retries)),
        "--model-path",
        str(args.model_path),
        "--gpu",
        str(args.gpu),
        "--dtype",
        str(args.dtype),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--max-new-tokens",
        str(int(args.decompose_max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--seed",
        str(int(args.seed)),
        "--trust-remote-code",
        str(int(args.trust_remote_code)),
        "--gen-batch-size",
        str(int(args.decompose_gen_batch_size)),
        "--batch-concurrency",
        str(int(args.decompose_batch_concurrency)),
        "--output-json",
        str(p["decompose_json"]),
        "--output-jsonl",
        str(p["decompose_jsonl"]),
    ]

    if str(args.candidate_sqlite).strip():
        cmd.extend(["--candidate-sqlite", str(args.candidate_sqlite)])
        cmd.extend(["--candidate-table", str(args.candidate_table)])
        cmd.extend(["--candidate-question-col", str(args.candidate_question_col)])
        cmd.extend(["--candidate-sql-col", str(args.candidate_sql_col)])
        cmd.extend(["--candidate-id-col", str(args.candidate_id_col)])
    elif str(args.candidate_json).strip():
        cmd.extend(["--candidate-json", str(args.candidate_json)])
    else:
        cmd.extend(["--seed-json", str(args.seed_json)])

    _append(cmd, "--sbert-device", args.sbert_device)
    return cmd


def build_synth_cmd(args: argparse.Namespace, p: Dict[str, Path]) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(p["synth_script"]),
        "--retrieval-json",
        str(p["decompose_json"]),
        "--prompt-file",
        str(args.synthesis_prompt_file),
        "--schema-json",
        str(args.schema_json),
        "--backend",
        str(args.backend),
        "--api-base",
        str(args.api_base),
        f"--api-key={args.api_key}",
        "--model-name",
        str(args.model_name),
        "--timeout",
        str(float(args.timeout)),
        "--num-retries",
        str(int(args.num_retries)),
        "--use-pydantic-schema",
        str(int(args.use_pydantic_schema)),
        "--logprob-mode",
        str(args.logprob_mode),
        "--logprobs",
        str(int(args.logprobs)),
        "--model-path",
        str(args.model_path),
        "--gpu",
        str(args.gpu),
        "--dtype",
        str(args.dtype),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--max-new-tokens",
        str(int(args.synth_max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--seed",
        str(int(args.seed)),
        "--trust-remote-code",
        str(int(args.trust_remote_code)),
        "--gen-batch-size",
        str(int(args.synth_gen_batch_size)),
        "--batch-concurrency",
        str(int(args.synth_batch_concurrency)),
        "--synth-evidence-per-decomp",
        str(int(args.synth_evidence_per_decomp)),
        "--synth-include-global-merged",
        str(int(args.synth_include_global_merged)),
        "--synth-where-only-evidence",
        str(int(args.synth_where_only_evidence)),
        "--db-path",
        str(args.db_path),
        "--preview-rows",
        str(int(args.preview_rows)),
        "--skip-exec",
        str(int(args.skip_exec)),
        "--gt-json",
        str(args.gt_json),
        "--run-eval",
        str(int(args.run_eval)),
        "--eval-script",
        str(args.eval_script),
        "--eval-ready-json",
        str(p["eval_ready_json"]),
        "--eval-ready-jsonl",
        str(p["eval_ready_jsonl"]),
        "--eval-output-json",
        str(p["eval_output_json"]),
        "--compute-bertscore",
        str(int(args.compute_bertscore)),
        "--ast-weight-select",
        str(float(args.ast_weight_select)),
        "--ast-weight-where",
        str(float(args.ast_weight_where)),
        "--ast-weight-from",
        str(float(args.ast_weight_from)),
        "--acceptance-config",
        str(args.acceptance_config),
        "--acceptance-output-json",
        str(p["acceptance_output_json"]),
        "--output-json",
        str(p["synth_json"]),
        "--output-jsonl",
        str(p["synth_jsonl"]),
    ]
    return cmd


def _extract_top_retrievals(row: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    ranked = row.get("ranked_results")
    if not isinstance(ranked, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in ranked[: max(1, int(top_n))]:
        if not isinstance(item, dict):
            continue
        c = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        out.append(
            {
                "rank": item.get("rank"),
                "score": item.get("sbert_score", item.get("total_score")),
                "candidate_id": c.get("candidate_id"),
                "candidate_question": c.get("question"),
                "candidate_sql": c.get("sql"),
            }
        )
    return out


def build_chat_response(
    *,
    args: argparse.Namespace,
    p: Dict[str, Path],
) -> Dict[str, Any]:
    synth_obj = json.loads(p["synth_json"].read_text(encoding="utf-8"))
    results = synth_obj.get("results") if isinstance(synth_obj.get("results"), list) else []
    row = results[0] if results and isinstance(results[0], dict) else {}

    synth = row.get("synthesized") if isinstance(row.get("synthesized"), dict) else {}
    decomp = row.get("decomposition") if isinstance(row.get("decomposition"), dict) else {}
    exec_prev = row.get("execution_preview") if isinstance(row.get("execution_preview"), dict) else {}

    final_sql = str(synth.get("final_sql") or "").strip()
    error = row.get("error")
    status = "ok" if (final_sql and not error) else "error"

    response = {
        "status": status,
        "item_id": args.item_id or row.get("item_id"),
        "question": args.question,
        "final_sql": final_sql,
        "confidence_overall": synth.get("confidence_overall"),
        "error": error,
        "decomposed_queries": decomp.get("decomposed_queries") if isinstance(decomp.get("decomposed_queries"), list) else [],
        "retrieval_top": _extract_top_retrievals(row, top_n=min(3, int(args.top_k))),
        "execution_preview": {
            "error": exec_prev.get("error"),
            "columns": exec_prev.get("columns"),
            "rows": exec_prev.get("rows"),
        },
        "artifacts": {
            "decompose_json": str(p["decompose_json"]),
            "synth_json": str(p["synth_json"]),
            "eval_ready_json": str(p["eval_ready_json"]),
            "eval_ready_jsonl": str(p["eval_ready_jsonl"]),
            "eval_output_json": str(p["eval_output_json"]) if int(args.run_eval) == 1 else None,
        },
        "meta": {
            "backend": args.backend,
            "model_name": args.model_name,
            "use_pydantic_schema": bool(int(args.use_pydantic_schema)),
            "logprob_mode": args.logprob_mode,
            "run_eval": int(args.run_eval),
            "skip_exec": int(args.skip_exec),
            "run_tag": p["run_dir"].name,
        },
    }
    return response


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    p = derive_paths(args, root)

    if not bool(int(args.dry_run)):
        p["run_dir"].mkdir(parents=True, exist_ok=True)

    decompose_cmd = build_decompose_cmd(args, p)
    synth_cmd = build_synth_cmd(args, p)

    _run(decompose_cmd, dry_run=bool(int(args.dry_run)))
    _run(synth_cmd, dry_run=bool(int(args.dry_run)))

    if bool(int(args.dry_run)):
        return

    response = build_chat_response(args=args, p=p)
    response_path = p["response_json"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(int(args.keep_artifacts)):
        # Keep only the final response file.
        for k in (
            "decompose_json",
            "decompose_jsonl",
            "synth_json",
            "synth_jsonl",
            "eval_ready_json",
            "eval_ready_jsonl",
            "eval_output_json",
            "acceptance_output_json",
        ):
            fp = p[k]
            if fp.exists() and fp.is_file():
                fp.unlink()

    print(json.dumps(response, ensure_ascii=False, indent=2))
    print(f"\nSaved chat response: {response_path}")


if __name__ == "__main__":
    main()
