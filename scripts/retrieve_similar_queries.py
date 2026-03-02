#!/usr/bin/env python3
"""SBERT retrieval over NL->SQL seed candidates.

This script ranks candidate (question, sql) pairs against an input natural-language
question using sentence-transformer cosine similarity only.

Default sources:
- questions: data/natural_question_1500.json (via --question-index)
- candidates: data/seed_questions.json (decomposed_query blocks)

Optional sources:
- generic candidate JSON list
- sqlite table containing question/sql columns
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SBERT_BATCH_SIZE = 64
DEFAULT_EMBED_BACKEND = "sbert"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"

_SBERT_BACKEND: Optional[Tuple[Any, Any]] = None
_SBERT_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_SBERT_CANDIDATE_EMBED_CACHE: Dict[Tuple[str, str, str], Any] = {}
_OPENAI_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_OPENAI_CANDIDATE_EMBED_CACHE: Dict[Tuple[str, str, str], Any] = {}


@dataclass
class Candidate:
    candidate_id: str
    question: str
    sql: str
    parent_question: Optional[str] = None
    source: str = "seed_json"


@dataclass
class MatchResult:
    rank: int
    total_score: float
    sbert_score: float
    candidate: Candidate


def _get_sbert_backend() -> Tuple[Any, Any]:
    global _SBERT_BACKEND
    if _SBERT_BACKEND is not None:
        return _SBERT_BACKEND

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sentence-transformers is required for retrieval. "
            "Install with: pip install sentence-transformers"
        ) from e

    _SBERT_BACKEND = (SentenceTransformer, util)
    return _SBERT_BACKEND


def _get_sbert_model(model_name: str, device: Optional[str]) -> Any:
    key = (model_name, device or "")
    cached = _SBERT_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    SentenceTransformer, _ = _get_sbert_backend()
    kwargs: Dict[str, Any] = {}
    if device:
        kwargs["device"] = device

    model = SentenceTransformer(model_name, **kwargs)
    _SBERT_MODEL_CACHE[key] = model
    return model


def _get_openai_client(api_base: Optional[str], api_key: Optional[str]) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is required for embed-backend=openai")

    base = (api_base or "https://api.openai.com/v1").strip()
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY (or --embed-api-key) is required for embed-backend=openai")

    cache_key = (base, key)
    cached = _OPENAI_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = OpenAI(base_url=base, api_key=key)
    _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def _normalize_vec(v: Sequence[float]) -> List[float]:
    arr = [float(x) for x in v]
    norm = math.sqrt(sum(x * x for x in arr))
    if norm <= 0.0:
        return arr
    return [x / norm for x in arr]


def _embed_texts_openai(
    *,
    client: Any,
    model: str,
    texts: Sequence[str],
    batch_size: int,
) -> List[List[float]]:
    out: List[List[float]] = []
    bsz = max(1, int(batch_size))
    for i in range(0, len(texts), bsz):
        chunk = list(texts[i : i + bsz])
        if not chunk:
            continue
        resp = client.embeddings.create(model=model, input=chunk)
        data = sorted(list(resp.data), key=lambda x: int(x.index))
        for item in data:
            out.append(_normalize_vec(item.embedding))
    return out


def _get_candidate_embeddings_openai(
    *,
    candidates: Sequence[Candidate],
    api_base: Optional[str],
    api_key: Optional[str],
    model: str,
    batch_size: int,
) -> List[List[float]]:
    key = ((api_base or "").strip(), model, _candidate_signature(candidates))
    cached = _OPENAI_CANDIDATE_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    client = _get_openai_client(api_base, api_key)
    embeddings = _embed_texts_openai(
        client=client,
        model=model,
        texts=[c.question for c in candidates],
        batch_size=batch_size,
    )
    _OPENAI_CANDIDATE_EMBED_CACHE[key] = embeddings
    return embeddings


def _candidate_signature(candidates: Sequence[Candidate]) -> str:
    h = hashlib.sha1()
    for c in candidates:
        h.update(c.candidate_id.encode("utf-8", errors="ignore"))
        h.update(b"\t")
        h.update(c.question.strip().encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _get_candidate_embeddings(
    candidates: Sequence[Candidate],
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> Any:
    key = (model_name, device or "", _candidate_signature(candidates))
    cached = _SBERT_CANDIDATE_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    model = _get_sbert_model(model_name, device)
    embeddings = model.encode(
        [c.question for c in candidates],
        batch_size=max(1, int(batch_size)),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _SBERT_CANDIDATE_EMBED_CACHE[key] = embeddings
    return embeddings


def rank_candidates(
    question: str,
    candidates: Sequence[Candidate],
    top_k: int,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    sbert_device: Optional[str] = None,
    sbert_batch_size: int = DEFAULT_SBERT_BATCH_SIZE,
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_api_base: Optional[str] = None,
    embed_api_key: Optional[str] = None,
    embed_model: str = DEFAULT_OPENAI_EMBED_MODEL,
    embed_batch_size: int = DEFAULT_SBERT_BATCH_SIZE,
) -> List[MatchResult]:
    q = (question or "").strip()
    if not q:
        raise ValueError("Input question is empty")
    if not candidates:
        return []

    backend = (embed_backend or DEFAULT_EMBED_BACKEND).strip().lower()
    if backend not in {"sbert", "openai"}:
        raise ValueError(f"Unsupported embed_backend={embed_backend!r}; expected sbert|openai")

    scored: List[Tuple[float, Candidate]] = []

    if backend == "openai":
        client = _get_openai_client(embed_api_base, embed_api_key)
        q_vecs = _embed_texts_openai(
            client=client,
            model=embed_model,
            texts=[q],
            batch_size=1,
        )
        if not q_vecs:
            return []
        q_vec = q_vecs[0]
        cand_vecs = _get_candidate_embeddings_openai(
            candidates=candidates,
            api_base=embed_api_base,
            api_key=embed_api_key,
            model=embed_model,
            batch_size=embed_batch_size,
        )

        for i, cand in enumerate(candidates):
            c_vec = cand_vecs[i]
            raw = float(sum(a * b for a, b in zip(q_vec, c_vec)))
            sbert_score = max(0.0, min(1.0, (raw + 1.0) / 2.0))
            scored.append((sbert_score, cand))
    else:
        model = _get_sbert_model(sbert_model, sbert_device)
        _, util = _get_sbert_backend()

        query_embedding = model.encode(
            [q],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        candidate_embeddings = _get_candidate_embeddings(candidates, sbert_model, sbert_device, sbert_batch_size)

        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        for i, cand in enumerate(candidates):
            raw = float(cosine_scores[i])
            sbert_score = max(0.0, min(1.0, (raw + 1.0) / 2.0))
            scored.append((sbert_score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[MatchResult] = []
    for idx, (sbert_score, cand) in enumerate(scored[: max(1, int(top_k))], start=1):
        results.append(
            MatchResult(
                rank=idx,
                total_score=float(sbert_score),
                sbert_score=float(sbert_score),
                candidate=cand,
            )
        )
    return results


def load_candidates_from_seed_json(path: Path) -> List[Candidate]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[Candidate] = []

    def add_candidate(cand_id: str, q: str, sql: str, parent: Optional[str], source: str) -> None:
        q = (q or "").strip()
        sql = (sql or "").strip()
        if not q or not sql:
            return
        out.append(
            Candidate(
                candidate_id=cand_id,
                question=q,
                sql=sql,
                parent_question=parent,
                source=source,
            )
        )

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            parent = item.get("original_question") or item.get("question") or item.get("natural_question")

            dq = item.get("decomposed_query")
            if isinstance(dq, dict):
                for q_key, payload in dq.items():
                    if not isinstance(payload, dict):
                        continue
                    add_candidate(
                        cand_id=f"seed[{i}].{q_key}",
                        q=payload.get("question", ""),
                        sql=payload.get("sql", ""),
                        parent=parent,
                        source="seed_json/decomposed",
                    )
                continue

            add_candidate(
                cand_id=f"seed[{i}]",
                q=item.get("question", ""),
                sql=item.get("sql") or item.get("gt_sql") or "",
                parent=parent,
                source="seed_json/flat",
            )

    elif isinstance(obj, dict):
        for q_key, payload in obj.items():
            if not isinstance(payload, dict):
                continue
            add_candidate(
                cand_id=f"seed.{q_key}",
                q=payload.get("question", ""),
                sql=payload.get("sql", ""),
                parent=None,
                source="seed_json/dict",
            )

    return out


def quote_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def load_candidates_from_sqlite(
    db_path: Path,
    table: str,
    question_col: str,
    sql_col: str,
    id_col: Optional[str],
) -> List[Candidate]:
    out: List[Candidate] = []
    conn = sqlite3.connect(str(db_path))
    try:
        if id_col:
            q = (
                f"SELECT {quote_ident(id_col)}, {quote_ident(question_col)}, {quote_ident(sql_col)} "
                f"FROM {quote_ident(table)}"
            )
        else:
            q = (
                f"SELECT rowid AS _rowid_, {quote_ident(question_col)}, {quote_ident(sql_col)} "
                f"FROM {quote_ident(table)}"
            )
        rows = conn.execute(q).fetchall()
        for row in rows:
            cand_id = str(row[0])
            question = str(row[1] or "")
            sql = str(row[2] or "")
            if question.strip() and sql.strip():
                out.append(
                    Candidate(
                        candidate_id=cand_id,
                        question=question,
                        sql=sql,
                        source=f"sqlite:{table}",
                    )
                )
    finally:
        conn.close()
    return out


def load_question_from_dataset(path: Path, index: int, key: str) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if index < 0 or index >= len(obj):
        raise IndexError(f"question-index {index} is out of range for {len(obj)} rows")

    row = obj[index]
    if not isinstance(row, dict):
        raise ValueError(f"Row {index} is not an object")

    q = row.get(key)
    if not isinstance(q, str) or not q.strip():
        for alt in ("natural_question", "question", "original_question"):
            if isinstance(row.get(alt), str) and row[alt].strip():
                return row[alt].strip()
        raise KeyError(f"Could not find non-empty question text using key={key!r}")
    return q.strip()


def is_safe_readonly_sql(sql: str) -> Tuple[bool, str]:
    s = (sql or "").strip()
    if not s:
        return False, "empty"
    if ";" in s.rstrip(";"):
        return False, "multiple_statements"

    head = s.lstrip().split(None, 1)[0].lower()
    if head not in {"select", "with"}:
        return False, f"not_select_or_with:{head}"

    banned = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "replace",
    }
    toks = set(re.findall(r"[a-z_]+", s.lower()))
    bad = sorted(toks & banned)
    if bad:
        return False, f"contains_banned:{bad}"
    return True, ""


def execute_sql_preview(db_path: Path, sql: str, max_rows: int) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    ok, why = is_safe_readonly_sql(sql)
    if not ok:
        return [], [], f"UNSAFE_SQL:{why}"

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(sql.strip().rstrip(";"))
        cols = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in cur.fetchmany(max_rows)]
        return cols, rows, None
    except Exception as e:  # noqa: BLE001
        return [], [], f"SQL_EXEC_ERROR:{e}"
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Retrieve best matching seed SQL queries for a natural-language question.")

    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"), help="Seed JSON path")
    source.add_argument("--candidate-json", help="Generic candidate JSON path")
    source.add_argument("--candidate-sqlite", help="SQLite DB path for candidate table")

    ap.add_argument("--candidate-table", default="query_library", help="SQLite table name when using --candidate-sqlite")
    ap.add_argument("--candidate-question-col", default="question", help="Question column in SQLite candidate table")
    ap.add_argument("--candidate-sql-col", default="sql", help="SQL column in SQLite candidate table")
    ap.add_argument("--candidate-id-col", default="id", help="ID column in SQLite candidate table")

    qsrc = ap.add_mutually_exclusive_group(required=False)
    qsrc.add_argument("--question", help="Input natural-language question")
    qsrc.add_argument(
        "--question-json",
        default=str(root / "data" / "natural_question_1500.json"),
        help="JSON dataset containing questions",
    )

    ap.add_argument("--question-index", type=int, default=0, help="Question index when using --question-json")
    ap.add_argument(
        "--question-key",
        default="natural_question",
        help="Question key in --question-json rows (fallbacks: natural_question/question/original_question)",
    )

    ap.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL, help="Sentence-Transformer model name/path")
    ap.add_argument("--sbert-device", default=None, help="Optional device override, e.g. cpu/cuda")
    ap.add_argument("--sbert-batch-size", type=int, default=DEFAULT_SBERT_BATCH_SIZE, help="Embedding batch size")
    ap.add_argument("--embed-backend", choices=["sbert", "openai"], default=DEFAULT_EMBED_BACKEND)
    ap.add_argument("--embed-model", default=DEFAULT_OPENAI_EMBED_MODEL, help="OpenAI embedding model when --embed-backend=openai")
    ap.add_argument("--embed-api-base", default="", help="OpenAI-compatible embedding API base; default https://api.openai.com/v1")
    ap.add_argument("--embed-api-key", default="", help="Embedding API key; default OPENAI_API_KEY env")
    ap.add_argument("--embed-batch-size", type=int, default=128, help="Batch size for OpenAI embedding calls")

    ap.add_argument("--top-k", type=int, default=5, help="Top-k candidates to return")
    ap.add_argument("--db-path", help="Optional SQLite DB path to execute top-ranked SQL for preview")
    ap.add_argument("--preview-rows", type=int, default=5, help="Max preview rows when --db-path is provided")
    ap.add_argument("--output-json", help="Optional output JSON path with ranked results")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.question:
        query_text = args.question.strip()
    else:
        query_text = load_question_from_dataset(Path(args.question_json), args.question_index, args.question_key)

    if not query_text:
        raise SystemExit("Input question is empty.")

    if args.candidate_sqlite:
        candidates = load_candidates_from_sqlite(
            db_path=Path(args.candidate_sqlite),
            table=args.candidate_table,
            question_col=args.candidate_question_col,
            sql_col=args.candidate_sql_col,
            id_col=args.candidate_id_col or None,
        )
    else:
        source_path = Path(args.candidate_json) if args.candidate_json else Path(args.seed_json)
        candidates = load_candidates_from_seed_json(source_path)

    if not candidates:
        raise SystemExit("No candidates loaded. Check input source format/path.")

    try:
        ranked = rank_candidates(
            query_text,
            candidates,
            top_k=args.top_k,
            sbert_model=args.sbert_model,
            sbert_device=args.sbert_device,
            sbert_batch_size=args.sbert_batch_size,
            embed_backend=args.embed_backend,
            embed_api_base=(args.embed_api_base or None),
            embed_api_key=(args.embed_api_key or None),
            embed_model=args.embed_model,
            embed_batch_size=int(args.embed_batch_size),
        )
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

    print("=" * 90)
    print("INPUT QUESTION")
    print(query_text)
    print("=" * 90)
    print(f"Scoring backend: {args.embed_backend}")
    if args.embed_backend == "openai":
        print(f"Embedding model: {args.embed_model}")
        if args.embed_api_base:
            print(f"Embedding API base override: {args.embed_api_base}")
    else:
        print(f"SBERT model: {args.sbert_model}")
        if args.sbert_device:
            print(f"SBERT device override: {args.sbert_device}")
    print(f"Candidates loaded: {len(candidates)}")
    print(f"Top-k: {len(ranked)}")
    print("=" * 90)

    for m in ranked:
        print(f"Rank {m.rank} | sbert_score={m.sbert_score:.4f}")
        print(f"  candidate_id: {m.candidate.candidate_id}")
        print(f"  source: {m.candidate.source}")
        if m.candidate.parent_question:
            print(f"  parent_question: {m.candidate.parent_question}")
        print(f"  candidate_question: {m.candidate.question}")
        print(f"  candidate_sql: {m.candidate.sql}")
        print("-" * 90)

    preview = None
    if args.db_path and ranked:
        best = ranked[0]
        cols, rows, err = execute_sql_preview(Path(args.db_path), best.candidate.sql, max_rows=args.preview_rows)
        preview = {
            "candidate_id": best.candidate.candidate_id,
            "columns": cols,
            "rows": rows,
            "error": err,
        }
        print("TOP-1 SQL PREVIEW")
        print(f"candidate_id: {best.candidate.candidate_id}")
        if err:
            print(f"error: {err}")
        else:
            print(f"columns: {cols}")
            print(f"rows (max {args.preview_rows}):")
            for r in rows:
                print(f"  {r}")
        print("=" * 90)

    if args.output_json:
        payload = {
            "input_question": query_text,
            "retrieval_backend": args.embed_backend,
            "sbert": {
                "model": args.sbert_model,
                "device": args.sbert_device,
                "batch_size": int(args.sbert_batch_size),
            },
            "openai_embedding": {
                "model": args.embed_model,
                "api_base": (args.embed_api_base or "https://api.openai.com/v1"),
                "batch_size": int(args.embed_batch_size),
            },
            "ranked_results": [
                {
                    **asdict(m),
                    "candidate": asdict(m.candidate),
                }
                for m in ranked
            ],
            "top1_sql_preview": preview,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved ranked results to: {out_path}")


if __name__ == "__main__":
    main()
