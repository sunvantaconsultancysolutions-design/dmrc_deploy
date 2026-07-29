"""Replays a set of real queries through hybrid_search() -> rerank() ->
evaluate_confidence() and reports, per query:

  - the top reranker score and the gate's decision under the CURRENT
    thresholds (RAG_MIN_CONFIDENCE / RAG_MIN_SEPARATION / RAG_HIGH_CONFIDENCE
    env vars, same ones reranker.py itself reads)
  - the gate's decision under the OLD (pre-HIGH_CONFIDENCE_ABSOLUTE)
    logic, so the effect of that bugfix is visible on real data before
    it's trusted
  - whether the top-ranked chunk's clause_no matches an "expected_clause"
    you supply, so mismatched-but-confident answers (a different failure
    mode than "wrongly rejected") are visible too -- this script only
    detects that failure mode, it does not fix it; see the module
    docstring in reranker.py and the accompanying PR notes for why that
    one has no clean automated fix.

This is the calibration script reranker.py's own module comment flags
as missing ("TODO: this calibration should be backed by a repeatable
script ..."). It must be run where the dense embedding model, reranker
model, and ChromaDB collection are actually available (i.e. on the GPU
pod / wherever `uvicorn src.app:app` runs) -- it is not runnable in an
offline sandbox with no model weights.

Usage
-----
    python scripts/calibrate_confidence.py --queries queries.jsonl
    python scripts/calibrate_confidence.py --queries queries.jsonl --csv out.csv
    python scripts/calibrate_confidence.py            # uses the built-in set below

queries.jsonl: one JSON object per line, e.g.
    {"query": "what is the penalty in the contract", "expected_clause": "6.7.2"}
    {"query": "what is the precedence order of documents", "expected_clause": "1.4"}

"expected_clause" is optional -- a substring match against the top
result's clause_no (or, for BOQ chunks, s_no). Omit it for queries
where you just want to see the scores, not check correctness.
"""

import argparse
import csv
import json
import statistics
import sys

sys.path.insert(0, ".")

from src.hybrid_retriever import hybrid_search
from src.reranker import (
    HIGH_CONFIDENCE_ABSOLUTE,
    MIN_ABSOLUTE_CONFIDENCE,
    MIN_SEPARATION_MARGIN,
    evaluate_confidence,
    rerank,
)
from src.prompt_engineering import get_boq_item_number

# The two real queries from the false-negative report this script was
# built to investigate, plus a couple of sanity-check contrasts. Not a
# substitute for real query logs -- replace with actual traffic ASAP.
DEFAULT_QUERIES = [
    {"query": "what is the penalty in the contract", "expected_clause": "6.7.2"},
    {"query": "What penalty applies if equipment stays down too long?", "expected_clause": "6.7.2"},
    {"query": "what is the precedence order of documents", "expected_clause": "1.4"},
    {"query": "what is the contractor required to do within two after notice to proceed"},
    {"query": "what is the contractor required to do within two months of notice to proceed", "expected_clause": "5.1"},
]


def _evaluate_old_gate(scores):
    """Reproduces evaluate_confidence()'s pre-HIGH_CONFIDENCE_ABSOLUTE
    logic against an already-computed score list, so the two gates can
    be compared on the exact same retrieval output (one hybrid_search +
    rerank call per query, not two).
    """
    if not scores:
        return {"confident": False, "reason": "no_scores"}
    top_score = scores[0]
    if top_score < MIN_ABSOLUTE_CONFIDENCE:
        return {"confident": False, "reason": "below_absolute_floor"}
    rest = scores[1:]
    if rest:
        separation = top_score - statistics.median(rest)
        if separation < MIN_SEPARATION_MARGIN:
            return {"confident": False, "reason": "no_separation_from_pool"}
    return {"confident": True, "reason": "ok"}


def _top_clause(candidate):
    metadata = candidate.get("metadata") or {}
    if metadata.get("chunk_type") == "boq":
        return get_boq_item_number(metadata)
    return metadata.get("clause_no")


def run(queries, csv_path=None):
    rows = []
    for item in queries:
        query = item["query"]
        expected = item.get("expected_clause")

        candidates = hybrid_search(query)
        reranked = rerank(query, candidates)
        scores = sorted(
            (c["reranker_score"] for c in reranked if c.get("reranker_score") is not None),
            reverse=True,
        )

        new_gate = evaluate_confidence(reranked)
        old_gate = _evaluate_old_gate(scores)

        top = reranked[0] if reranked else None
        top_clause = _top_clause(top) if top else None
        match = (
            None if expected is None or top_clause is None
            else (str(expected) in str(top_clause))
        )

        row = {
            "query": query,
            "top_score": round(scores[0], 4) if scores else None,
            "runner_up_score": round(scores[1], 4) if len(scores) > 1 else None,
            "old_gate_confident": old_gate["confident"],
            "old_gate_reason": old_gate["reason"],
            "new_gate_confident": new_gate["confident"],
            "new_gate_reason": new_gate["reason"],
            "gate_flipped": old_gate["confident"] != new_gate["confident"],
            "top_clause": top_clause,
            "expected_clause": expected,
            "clause_match": match,
        }
        rows.append(row)

        flag = "FLIPPED " if row["gate_flipped"] else ""
        mismatch_flag = " [MISMATCH]" if match is False else ""
        print(
            f"{flag}{row['new_gate_confident']!s:5} "
            f"top={row['top_score']} runner_up={row['runner_up_score']} "
            f"reason={row['new_gate_reason']:22} clause={top_clause}{mismatch_flag}  "
            f"| {query}"
        )

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")

    flipped = sum(1 for r in rows if r["gate_flipped"])
    mismatched = sum(1 for r in rows if r["clause_match"] is False)
    print(
        f"\n{len(rows)} queries | thresholds: "
        f"MIN_ABSOLUTE_CONFIDENCE={MIN_ABSOLUTE_CONFIDENCE} "
        f"MIN_SEPARATION_MARGIN={MIN_SEPARATION_MARGIN} "
        f"HIGH_CONFIDENCE_ABSOLUTE={HIGH_CONFIDENCE_ABSOLUTE}"
    )
    print(f"gate decision changed (old vs new): {flipped}")
    print(f"confident but top clause != expected: {mismatched}  "
          f"(not fixed by this script -- see module docstring)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", help="Path to a .jsonl file of {query, expected_clause?} objects")
    ap.add_argument("--csv", help="Optional path to write full results as CSV")
    args = ap.parse_args()

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [json.loads(line) for line in f if line.strip()]
    else:
        queries = DEFAULT_QUERIES
        print("(no --queries given -- using the small built-in sanity set)\n")

    run(queries, csv_path=args.csv)


if __name__ == "__main__":
    main()
