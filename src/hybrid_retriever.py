"""
hybrid_retriever.py

Chapter 9.4 / 9.8 / 9.9 / 9.10 / 9.12 -- Hybrid Retrieval Architecture,
Retrieval Workflow, Candidate Merging, Metadata Filtering, and
Retrieval Parameters.

------------------------------------------------------------------------
9.4  Hybrid Retrieval Architecture (documentation)
------------------------------------------------------------------------
This module is the single orchestration point that combines the two
independent retrieval paths already available in this project:

    - Dense retrieval  (Chapter 8, reused as-is): query.py's `search()`,
      which embeds the query with BGE-M3 and does ANN similarity search
      against the existing ChromaDB collection.
    - Sparse retrieval (Chapter 9.6, new): bm25_index.py's `BM25Index`,
      which does lexical term-matching over the same corpus of chunk
      text, read directly out of the same ChromaDB collection.

Neither retrieval path is replaced or weakened by the other -- both run
independently against the same underlying chunk corpus and the same
metadata schema, and their results are only combined at the very end
(see 9.9 Candidate Merging below). This keeps the two retrieval
mechanisms decoupled and independently testable/tunable, which is why
they live in separate modules (bm25_index.py, hybrid_retriever.py)
instead of being interleaved into one function.

------------------------------------------------------------------------
9.7  Dense vs Sparse Retrieval (documentation only, no code)
------------------------------------------------------------------------
Dense retrieval (BGE-M3 + ChromaDB):
  + Captures semantic/paraphrase similarity ("contractor obligations"
    matches a clause about "responsibilities of the Contractor" even
    with no shared words).
  + Robust to synonyms, rephrasing, and cross-lingual-style variation.
  - Can under-rank exact identifiers: a query containing a literal
    clause number ("6.8.2"), a rare part name, or a specific numeric
    value competes for meaning in embedding space against much more
    "semantically typical" chunks, and can lose to a chunk that's
    topically similar but doesn't contain the exact token the user
    actually wanted.
  - Opaque: harder to explain *why* a given chunk ranked where it did.

Sparse retrieval (BM25):
  + Excellent at exact / near-exact term matches -- clause numbers,
    BOQ identifiers, proper nouns, acronyms (e.g. "ECS", "DLP", "SAT").
  + Fast to build, fully explainable (term frequency + IDF), and needs
    no GPU/model inference.
  - Blind to meaning: "contractor duties" will not match a clause about
    "responsibilities of the Contractor" unless the words overlap.
  - Sensitive to exact phrasing/vocabulary choice.

Together: dense retrieval provides recall on meaning, BM25 provides
precision on exact terms -- Chapter 9.9's merge step is what lets a
single query benefit from both simultaneously instead of picking one.

------------------------------------------------------------------------
9.13 Advantages of Hybrid Retrieval (documentation only, no code)
------------------------------------------------------------------------
  - Higher recall than either method alone: a query that dense retrieval
    misses because of vocabulary mismatch can still be caught by BM25
    (and vice versa).
  - Robustness to query style: users who paste an exact clause number
    ("show me 6.10.2") and users who ask a natural-language question
    ("what does the contractor need to submit?") are both served well
    by the same retriever, without the caller needing to know which
    style of query they're issuing.
  - No additional embedding cost: BM25 is built from text already
    embedded and stored -- there is no new encoding step, no GPU work,
    and no duplicate vector storage (see bm25_index.py docstring).
  - Graceful degradation: if the embedding model or ChromaDB ANN index
    were ever unavailable, BM25 alone can still serve lexical queries
    (and symmetrically for dense retrieval), since the two paths do not
    depend on each other.
"""

import argparse
import os
from typing import Optional

from .bm25_index import get_bm25_index
from .query import search as dense_search, build_filter, get_model  # noqa: F401  (get_model re-exported for CLI warm-up parity with query.py)

# TASK 4 -- debug logging flag, read independently here so this module
# logs its own dense/BM25/merge output right where it's computed. Off
# by default; set RAG_DEBUG=1 to enable.
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


def _debug_clause_block(header: str, results: list) -> None:
    print("=" * 22)
    print(header)
    print("=" * 22)
    if not results:
        print("(none)")
        return
    for r in results:
        clause_no = (r.get("metadata") or {}).get("clause_no", "N/A")
        print(f"  {r['chunk_id']}  clause={clause_no}")


# ---------------------------------------------------------------------------
# 9.12 Retrieval Parameters -- configurable, overridable per call.
# ---------------------------------------------------------------------------

TOP_K_DENSE = 10   # how many candidates dense retrieval contributes before merging
TOP_K_BM25 = 10    # how many candidates BM25 retrieval contributes before merging
FINAL_TOP_K = 5    # size of the final merged result list returned to the caller


# ---------------------------------------------------------------------------
# 9.9 Candidate Merging
# ---------------------------------------------------------------------------

def _normalize_scores(results: list, score_key: str) -> list:
    """Min-max normalizes `score_key` across `results` to the [0, 1]
    range, writing the result into a new "normalized_score" field on
    each dict (the original score is left untouched).

    Why this is necessary: dense similarity_score is already in [0, 1]
    (cosine similarity of L2-normalized vectors), but BM25's bm25_score
    is an unbounded, corpus-dependent value that can be far outside that
    range. Comparing them directly ("keep highest score for duplicates")
    would always favor whichever scale happens to produce bigger raw
    numbers rather than whichever result is actually the better match.
    Normalizing each list independently onto a common [0, 1] scale
    before merging is the standard way to make "highest score wins"
    meaningful across two retrieval methods with different scoring
    functions.
    """
    if not results:
        return results

    scores = [r[score_key] for r in results]
    lo, hi = min(scores), max(scores)
    spread = hi - lo

    for r in results:
        if spread == 0:
            # All candidates tied (e.g. a single result, or a query with
            # zero BM25 term overlap for every candidate) -- treat them
            # as equally strong rather than dividing by zero.
            r["normalized_score"] = 1.0
        else:
            r["normalized_score"] = (r[score_key] - lo) / spread

    return results


def _dedupe_by_document_text(candidates: list) -> list:
    """TASK 1 FIX -- BOQ semantic-retrieval recall.

    Root cause: the BOQ corpus reuses the exact same short line-item
    description verbatim under many different parents/panels -- e.g.
    "1 No. Digital Ammeter, CT operated." is a distinct chunk_id under
    dozens of different `parent` BOQ items. Step 2 of the merge above
    ("remove duplicate chunk_ids") does not catch this: each occurrence
    IS a different chunk_id (a different BOQ row), it just happens to
    carry identical text.

    Confirmed live against this project's own ChromaDB collection: BM25
    alone returns "Explain Digital Ammeter"'s top-30 candidates as 17
    copies of the single string "1 No. Digital Ammeter, CT operated."
    (5 unique strings total in 30 hits); "Explain Selector Switch"
    returns 30 hits across only 7 unique strings. Dense retrieval hits
    the same duplication, since near-identical text embeds to
    near-identical vectors.

    Why this breaks natural-language BOQ queries end-to-end (not just
    "weak ranking"): reranker.py's evaluate_confidence() (Task 6,
    unchanged by this fix) flags a query low-confidence when the top
    reranker_score isn't separated from the median of the rest of the
    pool -- a heuristic for "did the top hit actually stand out". When
    15+ of the pool's candidates are the exact same string (and a
    cross-encoder scores identical (query, text) pairs identically),
    the median gets dragged up to the top score, separation collapses
    to ~0, and the query is misclassified as out-of-domain even though
    the right chunk was retrieved correctly. This is why these queries
    fail completely (NO_CONTEXT_ANSWER) rather than just ranking lower.

    Fix: collapse exact-duplicate document text to its single
    highest-scoring occurrence, so one real match is one candidate
    instead of seventeen. Only IDENTICAL text collapses (whitespace-
    trimmed) -- near-duplicates with any wording difference (e.g. this
    corpus's "Inbuilt" vs "Inbuild" typos) are left as distinct
    candidates, so no genuinely different BOQ row is ever dropped.
    Clause chunks are long, unique prose that essentially never
    collides on this check, so clause retrieval is unaffected by
    construction, not just by chance.
    """
    best_by_text: dict = {}
    order: list = []
    for c in candidates:
        key = (c.get("document") or "").strip()
        existing = best_by_text.get(key)
        if existing is None:
            best_by_text[key] = c
            order.append(key)
        elif c["score"] > existing["score"]:
            best_by_text[key] = c
    return [best_by_text[key] for key in order]


def merge_candidates(dense_results: list, sparse_results: list, final_top_k: int = FINAL_TOP_K) -> list:
    """Implements Chapter 9.9's merge rules exactly:

      1. Merge dense and sparse result lists.
      2. Remove duplicate chunk_ids.
      3. For a chunk_id present in both lists, keep the higher score.
      3.5. TASK 1 FIX -- collapse duplicate document TEXT (see
           `_dedupe_by_document_text` above) to its single highest-scoring
           occurrence, so a line item repeated verbatim across many BOQ
           rows doesn't flood the pool with copies of itself.
      4. Return the Top-K merged results.

    "Score" is compared on the normalized_score computed by
    `_normalize_scores` (see that function's docstring for why raw
    dense/BM25 scores cannot be compared directly). Each output entry
    also carries a `retrieval_source` field ("dense", "sparse", or
    "dense+sparse") so callers/UI can show why a result was surfaced.
    """
    dense_results = _normalize_scores(list(dense_results), "similarity_score")
    sparse_results = _normalize_scores(list(sparse_results), "bm25_score")

    merged: dict = {}  # chunk_id -> merged candidate dict

    for r in dense_results:
        merged[r["chunk_id"]] = {
            "chunk_id": r["chunk_id"],
            "document": r["document"],
            "metadata": r["metadata"],
            "score": r["normalized_score"],
            "retrieval_source": "dense",
            "dense_score": r["similarity_score"],
            "bm25_score": None,
        }

    for r in sparse_results:
        existing = merged.get(r["chunk_id"])
        if existing is None:
            merged[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": r["normalized_score"],
                "retrieval_source": "sparse",
                "dense_score": None,
                "bm25_score": r["bm25_score"],
            }
        else:
            # Duplicate chunk_id found in both lists: keep the higher
            # score, and record that it was surfaced by both methods.
            existing["retrieval_source"] = "dense+sparse"
            existing["bm25_score"] = r["bm25_score"]
            if r["normalized_score"] > existing["score"]:
                existing["score"] = r["normalized_score"]

    merged_list = list(merged.values())
    merged_list = _dedupe_by_document_text(merged_list)
    merged_list.sort(key=lambda c: c["score"], reverse=True)
    return merged_list[:final_top_k]


# ---------------------------------------------------------------------------
# 9.8 Retrieval Workflow
#
#     User Query
#         |
#         v
#     Dense Retrieval (ChromaDB)  --\
#                                     >--  Candidate Merge  --> Top-K merged candidates
#     BM25 Retrieval              --/
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Query synonym expansion for BM25 (contract-domain specific)
# ---------------------------------------------------------------------------
# These synonyms expand the BM25 query so that user-friendly natural-language
# terms retrieve chunks that use the contract's own terminology.
# Only applied to BM25 (lexical); dense retrieval handles semantics itself.
_QUERY_SYNONYMS: dict = {
    # PENALTY / DAMAGES
    "penalty":               ["penalty", "liquidated damages", "damages", "fine", "rs. 10,000"],
    "liquidated damages":    ["liquidated damages", "penalty", "penalty clause", "damages", "fine imposed", "rs per day", "maintenance period penalty"],
    "fine":                  ["penalty", "fine", "damages"],
    "compensation":          ["compensation", "liquidated damages", "penalty", "damages for delay"],
    "default":               ["default", "breach", "penalty", "liquidated damages"],
    "breach":                ["breach", "breach of contract", "default", "penalty"],
    "damages":               ["damages", "liquidated damages", "penalty"],
    # DEFECTS / WARRANTY
    "defect":                ["defect", "defects liability", "dlp", "defective"],
    "defects":               ["defects", "defects liability", "dlp"],
    "dlp":                   ["dlp", "defects liability period", "defect liability"],
    "warranty":              ["warranty", "guarantee period", "defects liability"],
    "guarantee period":      ["guarantee period", "warranty", "defects liability"],
    # TESTING / COMMISSIONING
    "testing":               ["testing", "commissioning", "test programme", "acceptance test"],
    "commissioning":         ["commissioning", "testing", "integrated testing", "system acceptance"],
    "system acceptance":     ["system acceptance test", "sat", "acceptance test", "commissioning"],
    "sat":                   ["system acceptance test", "sat", "acceptance test"],
    "integrated testing":    ["integrated testing", "integrated system test", "commissioning"],
    "functional test":       ["functional test", "functional tests", "installation test", "testing"],
    "trial running":         ["trial running", "trial run", "commissioning"],
    "inspection":            ["inspection", "testing", "acceptance test"],
    "handover":              ["handover", "completion", "takeover", "substantial completion"],
    # SCOPE / OBLIGATIONS
    "scope of work":         ["scope of work", "scope of supply", "contractor obligations", "responsibilities"],
    "scope of supply":       ["scope of supply", "scope of work", "supply"],
    "contractor obligations":["contractor obligations", "contractor responsibilities", "contractor shall"],
    "employer obligations":  ["employer obligations", "employer responsibilities", "employer shall"],
    "responsibilities":      ["responsibilities", "obligations", "contractor shall"],
    "obligation":            ["obligation", "obligations", "contractor shall", "responsibilities"],
    # INSTALLATION
    "installation":          ["installation", "erection", "installation plan", "method statement"],
    "method statement":      ["method statement", "installation plan", "programme"],
    "resident staff":        ["resident staff", "contractor staff", "representative"],
    # MAINTENANCE
    "maintenance":           ["maintenance", "operation and maintenance", "dlp", "maintenance period"],
    "operation and maintenance": ["operation and maintenance", "maintenance", "dlp"],
    "annual maintenance":    ["annual maintenance contract", "amc", "maintenance beyond dlp"],
    "amc":                   ["annual maintenance contract", "amc", "maintenance beyond dlp"],
    "routine maintenance":   ["routine maintenance", "corrective maintenance", "maintenance procedures"],
    "operation manual":      ["operation manual", "maintenance manual", "documentation"],
    "maintenance manual":    ["maintenance manual", "operation manual", "documentation"],
    # SPARE PARTS
    "spare parts":           ["spare parts", "spares", "tools and test equipment"],
    "spares":                ["spares", "spare parts", "tools and test equipment"],
    "spares list":           ["spares list", "spare parts", "schedule of spares"],
    "long lead time":        ["long lead time", "spare parts", "lead times"],
    "shelf life":            ["shelf life", "storage requirement", "spare parts"],
    "tools":                 ["tools", "test equipment", "tools and test equipment"],
    # DOCUMENTS / DRAWINGS
    "drawing":               ["drawing", "drawings and records", "as-built", "documentation"],
    "as-built":              ["as-built drawing", "as built drawing", "final drawing"],
    "submission":            ["submission", "drawings and documents", "notice to proceed"],
    "notice to proceed":     ["notice to proceed", "commencement", "start date"],
    # INTERFACES
    "interfacing":           ["interfacing", "interface", "coordination", "interfacing contractor"],
    "clearances":            ["clearances", "certificates", "statutory authorities", "interfacing agencies"],
    "coordination":          ["coordination", "interface", "civil contractor"],
    # SAFETY / ENVIRONMENT
    "safety":                ["safety", "health and safety", "safety requirement"],
    "health and safety":     ["health and safety", "safety requirement", "environmental"],
    "environmental":         ["environmental", "environmental requirement", "safety"],
    # INSURANCE / FINANCIAL SECURITY
    "insurance":             ["insurance", "indemnity", "indemnification"],
    "indemnity":             ["indemnity", "insurance", "indemnification"],
    "retention":             ["retention", "retention money", "payment"],
    "retention money":       ["retention money", "retention", "payment"],
    "bank guarantee":        ["bank guarantee", "performance security", "performance bond"],
    "performance security":  ["performance security", "performance bond", "bank guarantee"],
    "mobilization":          ["mobilization", "mobilisation", "advance payment"],
    "mobilisation":          ["mobilisation", "mobilization", "advance payment"],
    # COMPLETION / DELAY
    "completion":            ["completion", "time for completion", "completion period", "completion date"],
    "delay":                 ["delay", "delay damages", "liquidated damages", "extension of time"],
    "extension of time":     ["extension of time", "eot", "delay", "completion period"],
    "eot":                   ["extension of time", "eot", "delay"],
    # PAYMENT
    "payment":               ["payment", "interim payment", "payment schedule", "retention"],
    "interim payment":       ["interim payment", "payment certificate", "payment schedule"],
    # DISPUTE / ARBITRATION
    "dispute":               ["dispute", "arbitration", "dispute resolution", "adjudication"],
    "arbitration":           ["arbitration", "dispute resolution", "adjudication"],
    "jurisdiction":          ["jurisdiction", "applicable law", "dispute resolution", "arbitration"],
    # VARIATION / CLAIMS
    "variation":             ["variation", "variation order", "change order", "extra work"],
    "change order":          ["change order", "variation", "variation order"],
    "claims":                ["claims", "extra work", "variation", "dispute"],
    # TRAINING
    "training":              ["training", "training requirements", "staff training"],
    # ECS / TVS / SYSTEMS
    "ahu":                   ["air handling unit", "ahu", "air conditioning"],
    "air handling":          ["air handling unit", "ahu", "air conditioning"],
    "cooling tower":         ["cooling tower", "condenser", "chiller"],
    "chiller":               ["chiller", "cooling tower", "refrigeration", "chilled water", "chilled water pump", "feeder motors"],
    "relay":                 ["relay", "preventor relay", "single phasing preventor", "protection relay", "apfcr relay"],
    "contactor":             ["contactor", "ac-3 duty", "auxiliary contact", "pole contactor", "auxiliary contactor"],
    "pump":                  ["pump", "chilled water pump", "condenser water pump"],
    "tvs":                   ["tunnel ventilation", "tvs", "trackway exhaust", "ventilation"],
    "tunnel ventilation":    ["tunnel ventilation", "tvs", "trackway exhaust"],
    "ecs":                   ["environment control system", "ecs", "air conditioning"],
    "bms":                   ["building management system", "bms", "scada"],
    "verification and validation": ["verification", "validation", "design verification"],
    "performance requirements": ["performance requirements", "design conditions", "specifications"],
    # Phase 1 audit additions
    "bid value":             ["bid value", "tender value", "contract value", "total amount"],
    "tender value":          ["tender value", "bid value", "contract value", "tender total"],
    "worth":                 ["worth", "value", "amount", "total"],
    "mccb":                  ["mccb", "circuit breaker", "moulded case circuit breaker"],
    "acb":                   ["acb", "air circuit breaker", "circuit breaker"],
    "ahu":                   ["ahu", "air handling unit", "air conditioning"],
    "metro staff":           ["metro staff", "employer staff", "employer engineers", "employer's engineers"],
    "employer staff":        ["employer staff", "employer engineers", "employer's engineers", "metro staff"],
}


def _expand_query_for_bm25(query: str) -> str:
    """Returns an expanded query string for BM25 by appending synonym terms.

    Only used for BM25 (lexical). Dense retrieval handles semantic similarity
    without expansion. The expansion is additive — original query words are
    always kept so exact matches are never lost.

    Example:
        "what is the penalty" → "what is the penalty liquidated damages damages fine"
    """
    q_lower = query.lower()
    extra_terms = []
    for trigger, synonyms in _QUERY_SYNONYMS.items():
        if trigger in q_lower:
            for syn in synonyms:
                if syn not in q_lower and syn not in extra_terms:
                    extra_terms.append(syn)
    if extra_terms:
        return query + " " + " ".join(extra_terms)
    return query


def hybrid_search(
    query: str,
    top_k_dense: int = TOP_K_DENSE,
    top_k_bm25: int = TOP_K_BM25,
    final_top_k: int = FINAL_TOP_K,
    metadata_filter: Optional[dict] = None,
) -> list:
    """Runs the full Chapter 9.8 pipeline for a single query:

        1. Dense retrieval over ChromaDB (query.py's existing search()).
        2. BM25 retrieval over the same corpus (bm25_index.py).
        3. Candidate merge (see merge_candidates above).

    metadata_filter (9.10): the SAME filter dict is passed to both
    retrieval calls, so a filter like {"clause_no": "6.8.2"} or
    {"chapter": "Chapter 3"} restricts the candidate pool identically
    for dense and sparse retrieval -- neither path can return a chunk
    the other path would have excluded.
    """
    dense_results = dense_search(query, top_k=top_k_dense, metadata_filter=metadata_filter)

    bm25_index = get_bm25_index()
    # Expand query for BM25 with domain synonyms (e.g. "penalty" → adds
    # "liquidated damages", "damages", "fine" for better recall)
    bm25_query = _expand_query_for_bm25(query)
    sparse_results = bm25_index.search(bm25_query, top_k=top_k_bm25, metadata_filter=metadata_filter)

    if RAG_DEBUG:
        _debug_clause_block("Dense Results", dense_results)
        _debug_clause_block("BM25 Results", sparse_results)

    merged = merge_candidates(dense_results, sparse_results, final_top_k=final_top_k)

    if RAG_DEBUG:
        _debug_clause_block("Merged Candidates", merged)

    return merged


# ---------------------------------------------------------------------------
# CLI -- mirrors query.py's CLI shape for a consistent developer experience.
# ---------------------------------------------------------------------------

def print_hybrid_results(query: str, results: list) -> None:
    """Pretty-prints hybrid results, including which retrieval path(s)
    surfaced each result -- useful for validating 9.9's merge behavior
    by eye during manual testing.
    """
    print("=" * 70)
    print(f"Hybrid Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] score={result['score']:.4f}  "
            f"source={result['retrieval_source']}  "
            f"chunk_id={result['chunk_id']}"
        )
        print(
            f"    clause_no={metadata.get('clause_no', 'N/A')}  "
            f"heading={metadata.get('heading', 'N/A')}  "
            f"pdf_page={metadata.get('pdf_page', 'N/A')}  "
            f"document_name={metadata.get('document_name', 'N/A')}"
        )
        if result["dense_score"] is not None:
            print(f"    dense_similarity={result['dense_score']:.4f}", end="  ")
        if result["bm25_score"] is not None:
            print(f"bm25_score={result['bm25_score']:.4f}", end="")
        print()
        text_preview = result["document"][:300]
        print(f"    text: {text_preview}{'...' if len(result['document']) > 300 else ''}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid (dense + BM25) retrieval over the DMRC ChromaDB collection."
    )
    parser.add_argument("query", type=str, help="Natural language or clause-number query.")
    parser.add_argument("--top_k_dense", type=int, default=TOP_K_DENSE)
    parser.add_argument("--top_k_bm25", type=int, default=TOP_K_BM25)
    parser.add_argument("--final_top_k", type=int, default=FINAL_TOP_K)
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional metadata filter as key=value, e.g. --filter clause_no=6.8.2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_filter = build_filter(args.filter)
    results = hybrid_search(
        args.query,
        top_k_dense=args.top_k_dense,
        top_k_bm25=args.top_k_bm25,
        final_top_k=args.final_top_k,
        metadata_filter=metadata_filter,
    )
    print_hybrid_results(args.query, results)


if __name__ == "__main__":
    main()