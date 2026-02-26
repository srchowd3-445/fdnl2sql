#!/usr/bin/env python3
"""Run the 3-step pipeline in one command.

Pipeline:
1) Decompose + SBERT retrieval
2) SQL synthesis from decomposition/retrieval artifacts
3) Post-hoc eval + seed acceptance (driven by synthesis script flags)
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Orchestrate decompose->retrieve->synthesize->eval pipeline.")

    # Input questions
    qsrc = ap.add_mutually_exclusive_group(required=False)
    qsrc.add_argument("--question", help="Single ad-hoc question text")
    qsrc.add_argument("--question-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--question-index", type=int, default=0)
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--id-key", default="item_id")

    ap.add_argument("--batch-mode", type=int, default=0, help="0=single index, 1=range mode")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1)

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
        default=str(root / "prompting" / "zero_shot_sql_expert.txt"),
        help="Base synthesis context prompt; retrieval/decomposition evidence is appended automatically.",
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

    ap.add_argument("--decompose-gen-batch-size", type=int, default=32)
    ap.add_argument("--decompose-batch-concurrency", type=int, default=8)
    ap.add_argument("--synth-gen-batch-size", type=int, default=32)
    ap.add_argument("--synth-batch-concurrency", type=int, default=8)
    ap.add_argument("--synth-evidence-per-decomp", type=int, default=2)
    ap.add_argument("--synth-include-global-merged", type=int, default=0)
    ap.add_argument("--synth-where-only-evidence", type=int, default=1)
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")
    ap.add_argument("--logprobs", type=int, default=0)

    # Exec/eval/acceptance
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--preview-rows", type=int, default=20)
    ap.add_argument("--skip-exec", type=int, default=1)

    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--run-eval", type=int, default=1)
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
    ap.add_argument("--output-dir", default=str(root / "results" / "decompose_method"))
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--decompose-output-json", default="")
    ap.add_argument("--decompose-output-jsonl", default="")
    ap.add_argument("--synth-output-json", default="")
    ap.add_argument("--synth-output-jsonl", default="")
    ap.add_argument("--eval-ready-json", default="")
    ap.add_argument("--eval-ready-jsonl", default="")
    ap.add_argument("--eval-output-json", default="")
    ap.add_argument("--acceptance-output-json", default="")

    # Flow control
    ap.add_argument("--skip-decompose", type=int, default=0)
    ap.add_argument("--retrieval-json", default="", help="Use existing decompose+retrieve JSON when --skip-decompose=1")
    ap.add_argument("--dry-run", type=int, default=0)

    return ap.parse_args()


def _append(cmd: List[str], flag: str, val: Optional[str]) -> None:
    if val is None:
        return
    s = str(val).strip()
    if not s:
        return
    cmd.extend([flag, s])


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def derive_paths(args: argparse.Namespace, root: Path) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_tag.strip():
        tag = args.run_tag.strip()
    elif int(args.batch_mode) == 1:
        lim = int(args.limit)
        if lim == -1:
            tag = f"batch_from_{int(args.start_index)}"
        else:
            end = int(args.start_index) + max(0, lim) - 1
            tag = f"batch_{int(args.start_index)}_{end}"
    else:
        tag = f"oneq_idx_{int(args.question_index)}"

    p = {
        "out_dir": out_dir,
        "tag": tag,
        "decompose_json": Path(args.decompose_output_json) if args.decompose_output_json.strip() else out_dir / f"{tag}.decompose_retrieve.json",
        "decompose_jsonl": Path(args.decompose_output_jsonl) if args.decompose_output_jsonl.strip() else out_dir / f"{tag}.decompose_retrieve.jsonl",
        "synth_json": Path(args.synth_output_json) if args.synth_output_json.strip() else out_dir / f"{tag}.synth.json",
        "synth_jsonl": Path(args.synth_output_jsonl) if args.synth_output_jsonl.strip() else out_dir / f"{tag}.synth.sql.jsonl",
        "eval_ready_json": Path(args.eval_ready_json) if args.eval_ready_json.strip() else out_dir / f"{tag}.eval_ready.json",
        "eval_ready_jsonl": Path(args.eval_ready_jsonl) if args.eval_ready_jsonl.strip() else out_dir / f"{tag}.eval_ready.jsonl",
        "eval_output_json": Path(args.eval_output_json) if args.eval_output_json.strip() else out_dir / f"{tag}.eval.json",
        "acceptance_output_json": (
            Path(args.acceptance_output_json)
            if args.acceptance_output_json.strip()
            else out_dir / f"{tag}.acceptance.json"
        ),
        "decompose_script": root / "method" / "decompose_retrieve_top3_gemma_sql.py",
        "synth_script": root / "method" / "synthesize_from_decompose_retrieval_gemma_sql.py",
    }
    return p


def build_decompose_cmd(args: argparse.Namespace, p: dict) -> List[str]:
    cmd: List[str] = [sys.executable, str(p["decompose_script"])]

    if args.question and args.question.strip():
        _append(cmd, "--question", args.question)
    else:
        cmd.extend(["--question-json", str(args.question_json)])
        cmd.extend(["--question-key", str(args.question_key)])
        cmd.extend(["--id-key", str(args.id_key)])
        cmd.extend(["--batch-mode", str(int(args.batch_mode))])
        if int(args.batch_mode) == 1:
            cmd.extend(["--start-index", str(int(args.start_index))])
            cmd.extend(["--limit", str(int(args.limit))])
        else:
            cmd.extend(["--question-index", str(int(args.question_index))])

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

    cmd.extend(["--schema-json", str(args.schema_json)])
    cmd.extend(["--prompt-file", str(args.decompose_prompt_file)])
    cmd.extend(["--decompose-examples-file", str(args.decompose_examples_file)])

    cmd.extend(["--top-k", str(int(args.top_k))])
    cmd.extend(["--retrieval-per-decomp", str(int(args.retrieval_per_decomp))])
    cmd.extend(["--max-decomposed-queries", str(int(args.max_decomposed_queries))])

    cmd.extend(["--sbert-model", str(args.sbert_model)])
    _append(cmd, "--sbert-device", args.sbert_device)
    cmd.extend(["--sbert-batch-size", str(int(args.sbert_batch_size))])

    cmd.extend(["--backend", str(args.backend)])
    cmd.extend(["--api-base", str(args.api_base)])
    cmd.extend(["--api-key", str(args.api_key)])
    cmd.extend(["--model-name", str(args.model_name)])
    cmd.extend(["--timeout", str(float(args.timeout))])
    cmd.extend(["--num-retries", str(int(args.num_retries))])

    cmd.extend(["--model-path", str(args.model_path)])
    cmd.extend(["--gpu", str(args.gpu)])
    cmd.extend(["--dtype", str(args.dtype)])
    cmd.extend(["--tensor-parallel-size", str(int(args.tensor_parallel_size))])
    cmd.extend(["--gpu-memory-utilization", str(float(args.gpu_memory_utilization))])
    cmd.extend(["--max-model-len", str(int(args.max_model_len))])

    cmd.extend(["--max-new-tokens", str(int(args.decompose_max_new_tokens))])
    cmd.extend(["--temperature", str(float(args.temperature))])
    cmd.extend(["--top-p", str(float(args.top_p))])
    cmd.extend(["--seed", str(int(args.seed))])
    cmd.extend(["--trust-remote-code", str(int(args.trust_remote_code))])

    cmd.extend(["--gen-batch-size", str(int(args.decompose_gen_batch_size))])
    cmd.extend(["--batch-concurrency", str(int(args.decompose_batch_concurrency))])

    cmd.extend(["--output-json", str(p["decompose_json"])])
    _append(cmd, "--output-jsonl", str(p["decompose_jsonl"]))

    return cmd


def build_synth_cmd(args: argparse.Namespace, p: dict, retrieval_json: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(p["synth_script"]),
        "--retrieval-json",
        str(retrieval_json),
        "--prompt-file",
        str(args.synthesis_prompt_file),
        "--schema-json",
        str(args.schema_json),
        "--backend",
        str(args.backend),
        "--api-base",
        str(args.api_base),
        "--api-key",
        str(args.api_key),
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
    ]

    _append(cmd, "--output-jsonl", str(p["synth_jsonl"]))
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    p = derive_paths(args, root)
    print("Output dir:", p["out_dir"])
    print("Run tag:", p["tag"])

    if int(args.skip_decompose) == 1:
        if not str(args.retrieval_json).strip():
            raise SystemExit("--skip-decompose=1 requires --retrieval-json")
        retrieval_json = Path(args.retrieval_json)
    else:
        decompose_cmd = build_decompose_cmd(args, p)
        _run(decompose_cmd, dry_run=bool(int(args.dry_run)))
        retrieval_json = Path(p["decompose_json"])

    synth_cmd = build_synth_cmd(args, p, retrieval_json)
    _run(synth_cmd, dry_run=bool(int(args.dry_run)))

    print("\nPipeline complete")
    print("Decompose+Retrieve JSON:", retrieval_json)
    print("Synthesis JSON:", p["synth_json"])
    print("Eval output JSON:", p["eval_output_json"])
    print("Acceptance output JSON:", p["acceptance_output_json"])


if __name__ == "__main__":
    main()
