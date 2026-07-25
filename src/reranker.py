"""
reranker.py

Chapter 10 -- BGE Reranker.

------------------------------------------------------------------------
10.1  Role of the Reranker in the Pipeline
------------------------------------------------------------------------
This module sits directly after Chapter 9's Hybrid Retrieval stage:

    User Query
        |
        v
    Hybrid Retrieval (dense + BM25, merged -- Chapter 9)
        |
        v
    BGE Reranker  (this module)
        |
        v
    Top-N re-ranked chunks --> Prompt Builder (Chapter 11)

Hybrid Retrieval's merge step (Chapter 9.9) compares dense similarity
and BM25 scores only after independently min-max normalizing each list
-- a necessary heuristic for combining two incompatible scales, but it
never lets the query and a candidate chunk interact directly.

The BGE reranker (BAAI/bge-reranker-v2-m3) is a cross-encoder: it takes
the full (query, document) pair as a single joint input and produces
one calibrated relevance score per pair. This is far more accurate than
comparing independently-computed dense/BM25 scores, but also far more
expensive (full attention over query+document for every pair), which is
why it is only ever run over the small merged candidate pool produced
by Hybrid Retrieval (e.g. ~40 candidates), never over the full corpus.

This module intentionally does NOT call ChromaDB, BM25, or the dense
embedding model directly -- it is a pure post-processing step over
whatever candidate list Chapter 9's `hybrid_search()` (or
`merge_candidates()`) already produced. Each input candidate is
expected to be a dict shaped like Chapter 9's merged output, i.e. it
must contain at least:

    {
        "chunk_id": ...,
        "document": ...,   # the chunk text
        "metadata": ...,   # unchanged, passed through as-is
        "score": ...,               # Chapter 9's fused score
        "retrieval_source": ...,    # "dense" / "sparse" / "dense+sparse"
        "dense_score": ...,
        "bm25_score": ...,
    }

Every field already present on a candidate is preserved unchanged; this
module only adds one new field, `reranker_score`, so the output remains
a drop-in replacement for Chapter 9's output when handed to Chapter
11's Prompt Builder.
"""

import argparse
import os
import statistics
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_N = 10       # how many re-ranked results are returned to the caller
DEFAULT_BATCH_SIZE = 16  # matches Section 14.7's recommended reranker batch size
DEFAULT_MAX_LENGTH = 512  # matches Section 5.11's max token budget for a chunk

# TASK 4 -- debug logging flag, read independently here (not passed down
# from app.py) so this module logs its own output right where it's
# computed. Off by default; set RAG_DEBUG=1 to enable.
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

_tokenizer = None
_model = None
_device = None


# ---------------------------------------------------------------------------
# 10.2  Model Loading
# ---------------------------------------------------------------------------

def get_reranker_model():
    """Lazily load and cache the BGE reranker model + tokenizer (loaded
    once per process, mirroring query.py's get_model() pattern for the
    dense embedding model).
    """
    global _tokenizer, _model, _device
    if _model is None:
        print(f"Loading reranker model: {MODEL_NAME} ...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


# ---------------------------------------------------------------------------
# 10.3  Pairwise Scoring
# ---------------------------------------------------------------------------

def _score_pairs(
    query: str,
    documents: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list:
    """Builds (query, document) pairs and computes one relevance score
    per pair using the cross-encoder, processed in batches for GPU
    utilization (Section 14.7).

    Returns a list of floats in [0, 1] (sigmoid of the model's raw
    relevance logit -- BGE reranker's standard scoring convention),
    same length and order as `documents`.
    """
    tokenizer, model, device = get_reranker_model()

    scores = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits.view(-1).float()
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            scores.extend(batch_scores)

    return scores


# ---------------------------------------------------------------------------
# 10.4  Reranking Entry Point
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list,
    top_n: int = DEFAULT_TOP_N,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list:
    """Re-ranks Chapter 9's merged hybrid-retrieval candidates using the
    BGE cross-encoder.

    Parameters
    ----------
    query : str
        The original user query (same string passed into
        hybrid_search()).
    candidates : list[dict]
        Chapter 9's merged candidate list (output of
        hybrid_retriever.hybrid_search() or merge_candidates()). Each
        dict must contain "document"; every other field (chunk_id,
        metadata, score, retrieval_source, dense_score, bm25_score,
        ...) is preserved unchanged on the output.
    top_n : int
        Number of top re-ranked candidates to return (default 10, per
        Chapter 10's spec / Section 14.7's "Re-ranked Documents" = 10).
    batch_size : int
        Cross-encoder batch size (default 16, per Section 14.7).
    max_length : int
        Max token length per (query, document) pair (default 512, per
        Section 5.11's chunk token budget).

    Returns
    -------
    list[dict]
        The input candidates, each with a new "reranker_score" field
        (float, [0, 1]), sorted descending by reranker_score, truncated
        to `top_n`. All existing fields/metadata are preserved as-is.
    """
    if not candidates:
        return []

    documents = [c["document"] for c in candidates]
    scores = _score_pairs(query, documents, batch_size=batch_size, max_length=max_length)

    reranked = []
    for candidate, score in zip(candidates, scores):
        entry = dict(candidate)  # shallow copy -- preserve every existing field/metadata
        entry["reranker_score"] = round(float(score), 4)
        reranked.append(entry)

    reranked.sort(key=lambda c: c["reranker_score"], reverse=True)
    reranked = reranked[:top_n]

    if RAG_DEBUG:
        print("=" * 22)
        print("After Reranker")
        print("=" * 22)
        for c in reranked:
            clause_no = (c.get("metadata") or {}).get("clause_no", "N/A")
            print(f"  {c['chunk_id']}  clause={clause_no}  reranker_score={c['reranker_score']}")

    return reranked


# ---------------------------------------------------------------------------
# TASK 6 -- Distribution-based confidence gate for out-of-domain queries.
#
# Root cause this addresses: has_usable_context() in prompt_engineering.py
# only checks "is the candidate list non-empty" -- it has no notion of
# score quality. rerank() always returns its top_n=12 best-of-a-bad-lot
# candidates even for a completely off-topic query (e.g. "What is
# Artificial Intelligence?"), so has_usable_context() passes, sources get
# built and shown in the UI ("Grounded in 16 retrieved clauses"), even
# though every reranker_score is ~0 and Gemma correctly ignores all of
# them and answers "not found" anyway. The mismatch is a sources/UI
# problem, not an LLM problem.
#
# Fix: gate on the SCORE DISTRIBUTION of this query's own candidate pool,
# not a single hand-picked constant:
#   1. top_score below an absolute floor -> nothing cleared even a low
#      relevance bar.
#   2. top_score not meaningfully separated from the rest of the pool's
#      scores -> the reranker isn't discriminating anything as more
#      relevant than anything else, which is the out-of-domain signature
#      (a real hit normally stands out above its own candidate pool).
#
# Calibration note: the two defaults below are a reasonable starting
# point for BAAI/bge-reranker-v2-m3's sigmoid output, but MUST be
# re-validated against this corpus's real query logs (see
# scripts/calibrate_confidence.py) before being trusted in production --
# override via the env vars rather than editing the constants directly.
# ---------------------------------------------------------------------------

MIN_ABSOLUTE_CONFIDENCE = float(os.environ.get("RAG_MIN_CONFIDENCE", "0.10"))
MIN_SEPARATION_MARGIN = float(os.environ.get("RAG_MIN_SEPARATION", "0.08"))


def evaluate_confidence(reranked: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribution-based low-confidence detector, meant to run on
    rerank()'s output BEFORE sibling expansion / prompt building.

    Parameters
    ----------
    reranked : list[dict]
        rerank()'s output -- each entry must carry "reranker_score".

    Returns
    -------
    dict
        {"confident": bool, "top_score": float | None, "reason": str}
        A dict rather than a bare bool so callers/logs can see *why* a
        query was flagged low-confidence (useful for calibration and
        for debugging false positives/negatives later).
    """
    if not reranked:
        return {"confident": False, "top_score": None, "reason": "no_candidates"}

    scores = sorted(
        (c["reranker_score"] for c in reranked if c.get("reranker_score") is not None),
        reverse=True,
    )
    if not scores:
        return {"confident": False, "top_score": None, "reason": "no_scores"}

    top_score = scores[0]

    if top_score < MIN_ABSOLUTE_CONFIDENCE:
        return {"confident": False, "top_score": top_score, "reason": "below_absolute_floor"}

    rest = scores[1:]
    if rest:
        separation = top_score - statistics.median(rest)
        if separation < MIN_SEPARATION_MARGIN:
            return {"confident": False, "top_score": top_score, "reason": "no_separation_from_pool"}

    return {"confident": True, "top_score": top_score, "reason": "ok"}


# ---------------------------------------------------------------------------
# TASK 3 -- parent_clause sibling expansion.
# ---------------------------------------------------------------------------
#
# Root cause this addresses (from the forensic audit): clause-based
# chunking means a broad question like "what spare parts, tools, and
# test equipment must the contractor provide?" is really asking about
# an entire clause family (6.8, 6.8.1-6.8.8), but dense/BM25/reranker
# all score chunks independently on text similarity -- they have no
# notion of "these 9 chunks are one family." A sibling like 6.8.5
# ("Routine Change") shares almost no vocabulary with the query and
# legitimately scores low on both lexical and semantic grounds, so
# widening top-k alone (Task 2) cannot fully fix this: verified live,
# 6.8.5/6.8.6 rank 58th/59th out of 63 chunks even under BM25 alone.
#
# The fix does NOT change how retrieval scores anything. It runs once,
# after the reranker has already produced its top_n list, and asks a
# narrow question: "does this list already contain multiple members of
# the same clause family?" If so, that is strong evidence the user's
# question is about the family as a whole (not a single sub-topic that
# happens to reuse a clause number), so it is safe to go check whether
# any *other* members of that same family are also relevant enough to
# include -- using the SAME cross-encoder score, not a guess, so
# "relevant enough" is measured the same way the rest of the pipeline
# already measures relevance.

DEFAULT_MAX_EXTRA_PER_PARENT = 2   # cap per clause family
DEFAULT_MAX_TOTAL_EXTRA = 4        # global cap across all families in one answer
DEFAULT_SIBLING_TRIGGER = 2        # need >=N siblings already in reranked output to trigger expansion
DEFAULT_RELEVANCE_MARGIN = 0.15    # a candidate sibling must score within this margin of the
                                    # weakest already-included sibling from the same family


def expand_with_siblings(
    query: str,
    reranked: List[Dict[str, Any]],
    max_extra_per_parent: int = DEFAULT_MAX_EXTRA_PER_PARENT,
    max_total_extra: int = DEFAULT_MAX_TOTAL_EXTRA,
    sibling_trigger: int = DEFAULT_SIBLING_TRIGGER,
    relevance_margin: float = DEFAULT_RELEVANCE_MARGIN,
) -> List[Dict[str, Any]]:
    """Given the reranker's already-finalized output, looks for clause
    families (parent_clause) with multiple members already present, and
    conditionally pulls in additional siblings of that family that the
    cross-encoder confirms are still relevant to the query.

    Parameters
    ----------
    query : str
        The original user query (same string passed to rerank()).
    reranked : list[dict]
        rerank()'s output -- must carry "metadata" (with parent_clause,
        if any) and "reranker_score" on every entry.
    max_extra_per_parent : int
        Hard cap on how many siblings can be added for any one family
        (Task 5: keep prompt size reasonable).
    max_total_extra : int
        Hard cap on total additions across all families in this answer
        (Task 5: bound total token growth regardless of how many
        families the reranked list happens to touch).
    sibling_trigger : int
        Minimum number of a family's members that must already be in
        `reranked` before that family is even considered for expansion.
        Requiring >=2 (not >=1) avoids triggering on every single clause
        that happens to have a parent_clause value -- a lone match is
        normal, focused retrieval, not evidence of a broad "whole family"
        question. This directly protects the case the audit's stated
        constraint calls out: "Focused questions should continue
        performing exactly as they do now."
    relevance_margin : float
        A held-out sibling is only added if its reranker_score is within
        this margin of the *lowest* reranker_score already included from
        the same family. This ties "genuinely relevant" (Task 5) to the
        same score the reranker already uses, rather than an arbitrary
        absolute cutoff that would behave differently across query types.

    Returns
    -------
    list[dict]
        `reranked` with 0 or more additional entries appended, each
        carrying the same fields as a normal reranked entry plus
        "retrieval_source": "sibling_expansion" so callers/logs/UI can
        distinguish an expansion hit from an originally-ranked one.
        Re-sorted by reranker_score descending. Never mutates the input
        list's dicts in place.
    """
    if not reranked:
        return reranked

    # Lazy import: mirrors reranker.py's existing pattern of not taking a
    # hard dependency on the retriever/query module layout at import time
    # (see this file's own main()/hybrid_search import comment above).
    from .query import get_chunks_by_parent_clause

    already_included_ids = {c["chunk_id"] for c in reranked}

    # Group already-reranked entries by parent_clause.
    families: Dict[str, List[Dict[str, Any]]] = {}
    for c in reranked:
        parent = (c.get("metadata") or {}).get("parent_clause")
        if not parent:
            continue
        families.setdefault(parent, []).append(c)

    additions: List[Dict[str, Any]] = []

    for parent, members in families.items():
        if len(additions) >= max_total_extra:
            break
        if len(members) < sibling_trigger:
            continue  # single match in this family -- treat as focused, not broad

        weakest_included_score = min(m["reranker_score"] for m in members)

        siblings = get_chunks_by_parent_clause(parent)
        candidate_siblings = [s for s in siblings if s["chunk_id"] not in already_included_ids]
        if not candidate_siblings:
            continue

        # Score candidate siblings with the SAME cross-encoder used for
        # the main ranking, so "relevant enough" is measured consistently
        # rather than guessed. This is a small extra call (typically 1-7
        # short documents for a clause family in this corpus), not a
        # second retrieval pass.
        sibling_scores = _score_pairs(query, [s["document"] for s in candidate_siblings])

        scored_siblings = sorted(
            zip(candidate_siblings, sibling_scores), key=lambda pair: pair[1], reverse=True
        )

        added_for_this_parent = 0
        for sibling, score in scored_siblings:
            if added_for_this_parent >= max_extra_per_parent:
                break
            if len(additions) >= max_total_extra:
                break
            if score < weakest_included_score - relevance_margin:
                continue  # not close enough to the family's own relevance bar

            entry = dict(sibling)
            entry["reranker_score"] = round(float(score), 4)
            entry["retrieval_source"] = "sibling_expansion"
            entry["score"] = entry["reranker_score"]
            entry["dense_score"] = None
            entry["bm25_score"] = None
            additions.append(entry)
            already_included_ids.add(sibling["chunk_id"])
            added_for_this_parent += 1

    if not additions:
        return reranked

    expanded = list(reranked) + additions
    expanded.sort(key=lambda c: c["reranker_score"], reverse=True)

    if RAG_DEBUG:
        print("=" * 22)
        print("Sibling Expansion")
        print("=" * 22)
        for c in additions:
            clause_no = (c.get("metadata") or {}).get("clause_no", "N/A")
            print(f"  +{c['chunk_id']}  clause={clause_no}  reranker_score={c['reranker_score']}")

    return expanded


# ---------------------------------------------------------------------------
# CLI -- mirrors query.py / hybrid_retriever.py's CLI shape for a
# consistent developer experience. Only imports hybrid_search lazily,
# inside main(), so this module has no hard dependency on the retriever
# package layout and can be unit-tested with plain candidate dicts.
# ---------------------------------------------------------------------------

def print_reranked_results(query: str, results: list) -> None:
    """Pretty-prints reranked results, including the original
    retrieval_source and both underlying scores alongside the new
    reranker_score -- useful for manually validating that reranking is
    actually reordering the merged candidates sensibly.
    """
    print("=" * 70)
    print(f"Reranked Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] reranker_score={result['reranker_score']:.4f}  "
            f"fused_score={result.get('score')}  "
            f"source={result.get('retrieval_source')}  "
            f"chunk_id={result['chunk_id']}"
        )
        print(
            f"    clause_no={metadata.get('clause_no', 'N/A')}  "
            f"heading={metadata.get('heading', 'N/A')}  "
            f"pdf_page={metadata.get('pdf_page', 'N/A')}  "
            f"document_name={metadata.get('document_name', 'N/A')}"
        )
        text_preview = result["document"][:300]
        print(f"    text: {text_preview}{'...' if len(result['document']) > 300 else ''}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="BGE cross-encoder reranking over Chapter 9's hybrid retrieval candidates."
    )
    parser.add_argument("query", type=str, help="Natural language or clause-number query.")
    parser.add_argument("--top_k_dense", type=int, default=10)
    parser.add_argument("--top_k_bm25", type=int, default=10)
    parser.add_argument(
        "--merged_top_k",
        type=int,
        default=40,
        help="Size of the merged candidate pool handed to the reranker (Section 14.7).",
    )
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="Final reranked result count.")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional metadata filter as key=value, e.g. --filter clause_no=6.8.2",
    )
    return parser.parse_args()


def main():
    # Imported here (not at module scope) so reranker.py has no hard
    # dependency on the retriever package's import layout, per the
    # instruction not to modify hybrid_retriever.py.
    from .hybrid_retriever import hybrid_search
    from .query import build_filter

    args = parse_args()
    metadata_filter = build_filter(args.filter)

    candidates = hybrid_search(
        args.query,
        top_k_dense=args.top_k_dense,
        top_k_bm25=args.top_k_bm25,
        final_top_k=args.merged_top_k,
        metadata_filter=metadata_filter,
    )

    results = rerank(args.query, candidates, top_n=args.top_n)
    print_reranked_results(args.query, results)


if __name__ == "__main__":
    main()